import os
import itertools
import ROOT
import random
import math
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from operator import itemgetter

import onnxruntime


from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from PhysicsTools.NanoNN.helpers.jetmetCorrector import JetMETCorrector, rndSeed
from PhysicsTools.NanoNN.helpers.triggerHelper import passTrigger
from PhysicsTools.NanoNN.helpers.utils import closest, sumP4, polarP4, configLogger, get_subjets, deltaPhi, deltaR
from PhysicsTools.NanoNN.helpers.nnHelper import convert_prob, ensemble
from PhysicsTools.NanoNN.helpers.massFitter import fitMass
from PhysicsTools.NanoNN.helpers.higgsPairingAlgorithm_v2 import higgsPairingAlgorithm_v2

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

class _NullObject:
    '''An null object which does not store anything, and does not raise exception.'''
    def __bool__(self):
        return False
    def __nonzero__(self):
        return False
    def __getattr__(self, name):
        pass
    def __setattr__(self, name, value):
        pass

class METObject(Object):
    def p4(self):
        return polarP4(self, eta=None, mass=None)

class triggerEfficiency():
    def __init__(self, year):
        self._year = year
        trigger_files = {'data': {"2016APV": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2016.root'),
                                  "2016"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2016.root'),
                                  "2017"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2017.root'),
                                  "2018"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_2018.root'),
                                  "2022"   : '',
                                  "2022EE" : ''}[self._year],
                         'mc': {"2016APV": os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Summer16.root'),
                                "2016"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Summer16.root'),
                                "2017"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall17.root'),
                                "2018"   : os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data/trigger/JetHTTriggerEfficiency_Fall18.root'),
                                "2022"   : '',
                                "2022EE" : ''}[self._year]
                     }

        self.triggerHists = {}
        for key,tfile in trigger_files.items():
            if tfile=="": continue
            triggerFile = ROOT.TFile.Open(tfile)
            self.triggerHists[key]={
                'all': triggerFile.Get("efficiency_ptmass"),
                '0.9': triggerFile.Get("efficiency_ptmass_Xbb0p0To0p9"),
                '0.95': triggerFile.Get("efficiency_ptmass_Xbb0p9To0p95"),
                '0.98': triggerFile.Get("efficiency_ptmass_Xbb0p95To0p98"),
                '1.0': triggerFile.Get("efficiency_ptmass_Xbb0p98To1p0")
            }
            for key,h in self.triggerHists[key].items():
                h.SetDirectory(0)
            triggerFile.Close()
        
    def getEfficiency(self, pt, mass, xbb=-1, mcEff=False):
        if (mcEff and 'mc' not in self.triggerHists) or (not mcEff and 'data' not in self.triggerHists): return 1.0
        triggerHists = self.triggerHists['mc'] if mcEff else self.triggerHists['data']
        if xbb < 0.9 and xbb>=0:
            thist = triggerHists['0.9']
        elif xbb < 0.95 and xbb>=0.9:
            thist = triggerHists['0.95']
        elif xbb < 0.98 and xbb>=0.95:
            thist = triggerHists['0.98']
        elif xbb <= 1.0 and xbb>=0.98:
            thist = triggerHists['1.0']
        else:
            thist = triggerHists['all']

        # constrain to histogram bounds
        if mass > thist.GetXaxis().GetXmax() * 0.999: 
            tmass = thist.GetXaxis().GetXmax() * 0.999
        elif mass < 0: 
            tmass = 0.001
        else: 
            tmass = mass
            
        if pt > thist.GetYaxis().GetXmax() * 0.999:
            tpt = thist.GetYaxis().GetXmax() * 0.999
        elif pt < 0:
            tpt = 0.001
        else:
            tpt  = pt

        trigEff = thist.GetBinContent(thist.GetXaxis().FindFixBin(tmass), 
                                      thist.GetYaxis().FindFixBin(tpt))
        return trigEff
        
class hhh6bProducerPNetAK4(Module):
    
    def __init__(self, year, **kwargs):
        print(year)
        self.year = year
        self.Run = 2 if year in ["2016APV", "2016", "2017", "2018"] else 3

        self.jetType = 'ak8'
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'met_unclustered': None, 'smearMET': False, 'applyHEMUnc': False}
        self._opts = {'run_mass_regression': False, 'mass_regression_versions': ['ak8V01a', 'ak8V01b', 'ak8V01c'],
                      'WRITE_CACHE_FILE': False, 'option': "1", 'allJME': False}
        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'],
                                  self._jmeSysts['jes'],
                                  self._jmeSysts['jer'],
                                  self._jmeSysts['met_unclustered'],
                                  self._jmeSysts['applyHEMUnc']])
        self._allJME = self._opts['allJME']
        if self._allJME: self._needsJMECorr = False

        logger.info('Running %s channel for %s jets with JME systematics %s, other options %s',
                    self._opts['option'], self.jetType, str(self._jmeSysts), str(self._opts))
        
        # set up mass regression
        if self._opts['run_mass_regression']:
            from PhysicsTools.NanoNN.helpers.makeInputs import ParticleNetTagInfoMaker
            from PhysicsTools.NanoNN.helpers.runPrediction import ParticleNetJetTagsProducer
            self.tagInfoMaker = ParticleNetTagInfoMaker(
                fatjet_branch='FatJet', pfcand_branch='PFCands', sv_branch='SV', jetR=self._jetConeSize)
            prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/data')
            self.pnMassRegressions = [ParticleNetJetTagsProducer(
                '%s/MassRegression/%s/{version}/preprocess.json' % (prefix, self.jetType),
                '%s/MassRegression/%s/{version}/particle_net_regression.onnx' % (prefix, self.jetType),
                version=ver, cache_suffix='mass') for ver in self._opts['mass_regression_versions']]

        # Old: https://twiki.cern.ch/twiki/bin/viewauth/CMS/BtagRecommendation
        # New: https://btv-wiki.docs.cern.ch/ScaleFactors/
        if self.Run==2:
            self.DeepCSV_WP_L = {"2016APV": 0.2027, "2016": 0.1918, "2017": 0.1355, "2018": 0.1208}[self.year]
            self.DeepCSV_WP_M = {"2016APV": 0.6001, "2016": 0.5847, "2017": 0.4506, "2018": 0.4168}[self.year]
            self.DeepCSV_WP_T = {"2016APV": 0.8819, "2016": 0.8767, "2017": 0.7738, "2018": 0.7665}[self.year]
        
            self.DeepFlavB_WP_L = {"2016APV": 0.0508, "2016": 0.0480, "2017": 0.0532, "2018": 0.0490}[self.year]
            self.DeepFlavB_WP_M = {"2016APV": 0.2598, "2016": 0.2489, "2017": 0.3040, "2018": 0.2783}[self.year]
            self.DeepFlavB_WP_T = {"2016APV": 0.6502, "2016": 0.6377, "2017": 0.7476, "2018": 0.7100}[self.year]
        else:
            self.DeepFlavB_WP_L = {"2022": 0.0583, "2022EE": 0.0614}[self.year]
            self.DeepFlavB_WP_M = {"2022": 0.3086, "2022EE": 0.3196}[self.year]
            self.DeepFlavB_WP_T = {"2022": 0.7183, "2022EE": 0.7300}[self.year]
        
        # jet met corrections
        # jet mass scale/resolution: https://github.com/cms-nanoAOD/nanoAOD-tools/blob/a4b3c03ca5d8f4b8fbebc145ddcd605c7553d767/python/postprocessing/modules/jme/jetmetHelperRun2.py#L45-L58
        self._jmsValues = {"2016APV": [1.00, 0.9906, 1.0094],
                           "2016"   : [1.00, 0.9906, 1.0094],
                           "2017"   : [1.0016, 0.978, 0.986], # tuned to our top control region
                           "2018"   : [0.997, 0.993, 1.001],
                           "2022"   : [1.0, 1.0, 1.0],
                           "2022EE" : [1.0, 1.0, 1.0]}[self.year]
        self._jmrValues = {"2016APV": [1.00, 1.0, 1.09],  # tuned to our top control region
                           "2016"   : [1.00, 1.0, 1.09],  # tuned to our top control region
                           "2017"   : [1.03, 1.00, 1.07],
                           "2018"   : [1.065, 1.031, 1.099],
                           "2022"   : [0.0, 0.0, 0.0],
                           "2022EE" : [0.0, 0.0, 0.0]}[self.year]

        self._jmsValuesReg = {"2016APV": [1.00, 0.998, 1.002],
                              "2016"   : [1.00, 0.998, 1.002],
                              "2017"   : [1.002, 0.996, 1.008],
                              "2018"   : [0.994, 0.993, 1.001],
                              "2022"   : [1.0, 1.0, 1.0],
                              "2022EE" : [1.0, 1.0, 1.0]}[self.year]
        self._jmrValuesReg = {"2016APV": [1.028, 1.007, 1.063],
                              "2016"   : [1.028, 1.007, 1.063],
                              "2017"   : [1.026, 1.009, 1.059],
                              "2018"   : [1.031, 1.006, 1.075],
                              "2022"   : [1.0, 1.0, 1.0],
                              "2022EE" : [1.0, 1.0, 1.0]}[self.year]

        if self._needsJMECorr:
            self.jetmetCorr = JetMETCorrector(year=self.year, jetType="AK4PFchs", **self._jmeSysts)
            self.fatjetCorr = JetMETCorrector(year=self.year, jetType="AK8PFPuppi", **self._jmeSysts)
            self.subjetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)
            self._allJME = False

        if self._allJME:
            # self.applyHEMUnc = False
            self.applyHEMUnc = self._jmeSysts['applyHEMUnc']
            year_pf = "_%i"%self.year
            self.jetmetCorrectors = {
                'nominal': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                'JERUp': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='up'),
                'JERDown': JetMETCorrector(year=self.year, jetType="AK4PFchs", jer='down'),
                'JESUp': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes='up'),
                'JESDown': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes='down'),
                
                'JESUp_Abs': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_Abs'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs'+year_pf:JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                'JESUp_BBEC1': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                'JESUp_EC2': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_FlavQCD': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_FlavQCD': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_HF': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelBal': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelBal': JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK4PFchs", jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
            }
            # hemunc for 2018 only
            self.fatjetCorrectors = {
                'nominal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                #'HEMDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=True),
                'JERUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='up'),
                'JERDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='down'),
                'JESUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='up'),
                'JESDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='down'),

                'JESUp_Abs': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_Abs'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_Abs'+year_pf:JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_BBEC1': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_EC2': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_FlavQCD': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_FlavQCD': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_HF': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESUp_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelBal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelBal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                'JESUp_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                'JESDown_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
            }
            self._jmeLabels = self.fatjetCorrectors.keys()
        else:
            self._jmeLabels = []

        # selection
        if self._opts['option']=="5": print('Select Events with FatJet1 pT > 200 GeV and PNetXbb > 0.8 only')
        elif self._opts['option']=="10": print('Select FatJets with pT > 200 GeV and tau3/tau2 < 0.54 only')
        elif self._opts['option']=="21": print('Select FatJets with pT > 250 GeV and mass > 30 only')
        else: print('No selection')

        # trigger Efficiency
        self._teff = triggerEfficiency(self.year)

        # SVfit / FastMTT for ditau mass reconstruction
        macropath = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoNN/python/helpers/FastMTT_cc/')
        for fname in ["svFitAuxFunctions", "MeasuredTauLepton", "FastMTT"]:
          ROOT.gROOT.SetMacroPath(os.pathsep.join([ROOT.gROOT.GetMacroPath(), macropath]))
          try:
            #ROOT.gROOT.LoadMacro(macropath + fname+".cc" + " +g") # For some reason this doesn'k work here
            ROOT.gROOT.ProcessLine(".L " + macropath + fname+".cc")
          except RuntimeError:
            ROOT.gROOT.LoadMacro(macropath + fname+".cc" + " ++g")
        self.kUndefinedDecayType, self.kTauToHadDecay,  self.kTauToElecDecay, self.kTauToMuDecay = 0, 1, 2, 3  

    def beginJob(self):
        if self._needsJMECorr:
            self.jetmetCorr.beginJob()
            self.fatjetCorr.beginJob()
            self.subjetCorr.beginJob()
            
        if self._allJME:
            for key,corr in self.jetmetCorrectors.items():
                self.jetmetCorrectors[key].beginJob()

            for key,corr in self.fatjetCorrectors.items():
                self.fatjetCorrectors[key].beginJob()

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))

       
        # remove all possible h5 cache files
        for f in os.listdir('.'):
            if f.endswith('.h5'):
                os.remove(f)
                
        if self._opts['run_mass_regression']:
            #for p in self.pnMassRegressions:
            #    p.load_cache(inputFile)
            self.tagInfoMaker.init_file(inputFile, fetch_step=1000)
                
        self.out = wrappedOutputTree
        
        # weight variables
        self.out.branch("weight", "F")
        #self.out.branch("weightLHEScaleUp", "F")
        #self.out.branch("weightLHEScaleDown", "F")  

        # event variables
        self.out.branch("met", "F")
        self.out.branch("rho", "F")
        self.out.branch("metphi", "F")
        #self.out.branch("npvs", "F")
        self.out.branch("ht", "F")
        self.out.branch("passmetfilters", "O")
        self.out.branch("l1PreFiringWeight", "F")
        self.out.branch("l1PreFiringWeightUp", "F")
        self.out.branch("l1PreFiringWeightDown", "F")
        self.out.branch("triggerEffWeight", "F")
        self.out.branch("triggerEff3DWeight", "F")
        self.out.branch("triggerEffMCWeight", "F")
        self.out.branch("triggerEffMC3DWeight", "F")

        # fatjets
        self.out.branch("nfatjets","I")
        self.out.branch("nprobejets","I")
        self.out.branch("nprobetaus","I")
        self.out.branch("nHiggsMatchedJets","I")

        for idx in ([1, 2, 3, 4]):

            prefix = 'fatJet%i' % idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "MassSD", "F")
            self.out.branch(prefix + "MassSD_noJMS", "F")
            self.out.branch(prefix + "MassSD_UnCorrected", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            self.out.branch(prefix + "PNetXjj", "F")
            if self.Run!=2:
                self.out.branch(prefix + "PNetXtautau", "F")
                self.out.branch(prefix + "PNetXtaumu", "F")
                self.out.branch(prefix + "PNetXtaue", "F")
                self.out.branch(prefix + "PNetXtauany", "F")
            self.out.branch(prefix + "PNetQCD", "F")
            self.out.branch(prefix + "Area", "F")
            self.out.branch(prefix + "n3b1", "F")
            self.out.branch(prefix + "Tau2OverTau1", "F")
            self.out.branch(prefix + "Tau3OverTau2", "F")
            self.out.branch(prefix + "GenMatchIndex", "I")
            self.out.branch(prefix + "HiggsMatchedIndex", "I")
            self.out.branch(prefix + "HiggsMatched", "O")
            self.out.branch(prefix + "HasMuon", "O")
            self.out.branch(prefix + "HasElectron", "O")
            self.out.branch(prefix + "HasBJetCSVLoose", "O")
            self.out.branch(prefix + "HasBJetCSVMedium", "O")
            self.out.branch(prefix + "HasBJetCSVTight", "O")
            self.out.branch(prefix + "OppositeHemisphereHasBJet", "O")
            self.out.branch(prefix + "NSubJets", "I")

            # here we form the MHH system w. mass regressed
            self.out.branch(prefix + "PtOverMHH", "F")
            self.out.branch(prefix + "PtOverMSD", "F")
            self.out.branch(prefix + "PtOverMRegressed", "F")

            if self.isMC:
                self.out.branch(prefix + "MatchedGenPt", "F")

                # uncertainties
                self.out.branch(prefix + "MassSD_JMS_Down", "F")
                self.out.branch(prefix + "MassSD_JMS_Up", "F")
                self.out.branch(prefix + "MassSD_JMR_Down", "F")
                self.out.branch(prefix + "MassSD_JMR_Up", "F")

                self.out.branch(prefix + "PtOverMHH_JMS_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMS_Up", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Up", "F")

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        self.out.branch(prefix + "Pt" + "_" + syst, "F")
                        self.out.branch(prefix + "PtOverMHH" + "_" + syst, "F")

        # tri-higgs resolved variables
        self.out.branch("h1_t3_pt", "F")
        self.out.branch("h1_t3_eta", "F")
        self.out.branch("h1_t3_phi", "F")
        self.out.branch("h1_t3_mass", "F")
        self.out.branch("h1_t3_match", "I")
        self.out.branch("h1_t3_dRjets", "F")

        self.out.branch("h2_t3_pt", "F")
        self.out.branch("h2_t3_eta", "F")
        self.out.branch("h2_t3_phi", "F")
        self.out.branch("h2_t3_mass", "F")
        self.out.branch("h2_t3_match", "I")
        self.out.branch("h2_t3_dRjets", "F")

        self.out.branch("h3_t3_pt", "F")
        self.out.branch("h3_t3_eta", "F")
        self.out.branch("h3_t3_phi", "F")
        self.out.branch("h3_t3_mass", "F")
        self.out.branch("h3_t3_match", "I")
        self.out.branch("h3_t3_dRjets", "F")

        self.out.branch("h_fit_mass", "F")

        self.out.branch("h1_4b2t_pt", "F")
        self.out.branch("h1_4b2t_eta", "F")
        self.out.branch("h1_4b2t_phi", "F")
        self.out.branch("h1_4b2t_mass", "F")
        self.out.branch("h1_4b2t_match", "I")
        self.out.branch("h1_4b2t_dRjets", "F")

        self.out.branch("h2_4b2t_pt", "F")
        self.out.branch("h2_4b2t_eta", "F")
        self.out.branch("h2_4b2t_phi", "F")
        self.out.branch("h2_4b2t_mass", "F")
        self.out.branch("h2_4b2t_match", "I")
        self.out.branch("h2_4b2t_dRjets", "F")

        self.out.branch("h3_4b2t_pt", "F")
        self.out.branch("h3_4b2t_eta", "F")
        self.out.branch("h3_4b2t_phi", "F")
        self.out.branch("h3_4b2t_mass", "F")
        self.out.branch("h3_4b2t_match", "I")
        self.out.branch("h3_4b2t_dRjets", "F")

        self.out.branch("h_fit_mass_4b2t", "F")

        self.out.branch("reco6b_Idx", "I")
        self.out.branch("reco4b2t_Idx", "I")
        self.out.branch("reco4b2t_TauIsBoosted", "I")
        self.out.branch("reco4b2t_TauIsResolved", "I")
        self.out.branch("reco4b2t_TauFinalState", "I")

        # max min
        self.out.branch("max_h_eta", "F")
        self.out.branch("min_h_eta", "F")
        self.out.branch("max_h_dRjets", "F")
        self.out.branch("min_h_dRjets", "F")

        self.out.branch("max_h_eta_4b2t", "F")
        self.out.branch("min_h_eta_4b2t", "F")
        self.out.branch("max_h_dRjets_4b2t", "F")
        self.out.branch("min_h_dRjets_4b2t", "F")

        self.out.branch("ngenvistau", "I")
        #self.out.branch("nsignaltaus","I") # This is the same as "ntaus"
            
        # more small jets
        self.out.branch("nsmalljets", "I")
        self.out.branch("ntaus", "I")
        self.out.branch("nleps", "I")
        self.out.branch("nbtags", "I")
        for idx in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            prefix = 'jet%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "DeepFlavB", "F")
            self.out.branch(prefix + "PNetB", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "Area", "F")

            self.out.branch(prefix + "HasMuon", "O")
            self.out.branch(prefix + "HasElectron", "O")
            self.out.branch(prefix + "JetId", "F")
            if self.Run==2:
                self.out.branch(prefix + "PuId", "F")
                self.out.branch(prefix + "bRegCorr", "F")
                self.out.branch(prefix + "bRegRes", "F")
                self.out.branch(prefix + "cRegCorr", "F")
                self.out.branch(prefix + "cRegRes", "F")
            if self.isMC:
                self.out.branch(prefix + "MatchedGenPt", "F")
                self.out.branch(prefix + "HadronFlavour", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")
                self.out.branch(prefix + "FatJetMatched", "O")
                self.out.branch(prefix + "FatJetMatchedIndex", "I")

        # leptons
        for idx in ([1, 2, 3, 4]):
            prefix = 'lep%i'%idx
            self.out.branch(prefix + "Charge", "F")
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "Id", "I")
            if self.isMC:
                self.out.branch(prefix + "MatchedGenPt", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")
                self.out.branch(prefix + "FatJetMatched", "O")
                self.out.branch(prefix + "FatJetMatchedIndex", "I")

        for idx in ([1, 2, 3, 4]):
            prefix = 'tau%i'%idx
            self.out.branch(prefix + "Charge", "F")
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "Id", "I")
            self.out.branch(prefix + "decayMode", "F")

            if self.Run==2: # TODO: Can switch to v2p5 for Run2UL too, if inputs have branches available
                self.out.branch(prefix + "rawDeepTau2017v2p1VSe", "F")
                self.out.branch(prefix + "rawDeepTau2017v2p1VSjet", "F")
                self.out.branch(prefix + "rawDeepTau2017v2p1VSmu", "F")
            else:
                self.out.branch(prefix + "rawDeepTau2018v2p5VSe", "F")
                self.out.branch(prefix + "rawDeepTau2018v2p5VSjet", "F")
                self.out.branch(prefix + "rawDeepTau2018v2p5VSmu", "F")
            if self.isMC:
                self.out.branch(prefix + "MatchedGenPt", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")
                self.out.branch(prefix + "FatJetMatched", "O")
                self.out.branch(prefix + "FatJetMatchedIndex", "I")

        # gen variables
        for idx in ([1, 2, 3]):
            prefix = 'genHiggs%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Decay", "I")
            self.out.branch(prefix + "RecoPt", "F")
            self.out.branch(prefix + "RecoEta", "F")
            self.out.branch(prefix + "RecoPhi", "F")
            self.out.branch(prefix + "RecoMass", "F")
        for idx in ([1, 2, 3, 4, 5, 6]):
            prefix = 'genHiggsDau%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Id", "I")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        if self._opts['run_mass_regression'] and self._opts['WRITE_CACHE_FILE']:
            for p in self.pnMassRegressions:
                p.update_cache()
                
        # remove all h5 cache files
        if self._opts['run_mass_regression']:
            for f in os.listdir('.'):
                if f.endswith('.h5'):
                    os.remove(f)

        if self.isMC:
            cwd = ROOT.gDirectory
            outputFile.cd()
            cwd.cd()
                    
    def loadGenHistory(self, event, fatjets):
        # gen matching
        if not self.isMC:
            return
            
        try:
            genparts = event.genparts
        except RuntimeError as e:
            genparts = Collection(event, "GenPart")
            for idx, gp in enumerate(genparts):
                if 'dauIdx' not in gp.__dict__:
                    gp.dauIdx = []
                    if gp.genPartIdxMother >= 0:
                        mom = genparts[gp.genPartIdxMother]
                        if 'dauIdx' not in mom.__dict__:
                            mom.dauIdx = [idx]
                        else:
                            mom.dauIdx.append(idx)
            event.genparts = genparts

        def isHadronic(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) < 6:
                    return True
            return False

        def isTau(gp):
            if len(gp.dauIdx) == 0:
                raise ValueError('Particle has no daughters!')
            for idx in gp.dauIdx:
                if abs(genparts[idx].pdgId) == 15:
                    return True
            return False

        def getFinal(gp):
            for idx in gp.dauIdx:
                dau = genparts[idx]
                if dau.pdgId == gp.pdgId:
                    return getFinal(dau)
            return gp
               
        lepGenTops = []
        hadGenTops = []
        tauGenTops = []
        hadGenWs = []
        hadGenZs = []
        hadGenHs = []
        tauGenWs = []
        tauGenZs = []
        tauGenHs = []
        
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            if abs(gp.pdgId) == 6:
                for idx in gp.dauIdx:
                    dau = genparts[idx]
                    if abs(dau.pdgId) == 24:
                        genW = getFinal(dau)
                        gp.genW = genW
                        if isHadronic(genW):
                            hadGenTops.append(gp)
                        if isTau(genW):
                            tauGenTops.append(gp)
                        else:
                            lepGenTops.append(gp)
                    elif abs(dau.pdgId) in (1, 3, 5):
                        gp.genB = dau
            elif abs(gp.pdgId) == 24:
                if isHadronic(gp):
                    hadGenWs.append(gp)
                elif isTau(gp):
                    tauGenWs.append(gp)
            elif abs(gp.pdgId) == 23:
                if isHadronic(gp):
                    hadGenZs.append(gp)
                elif isTau(gp):
                    tauGenZs.append(gp)
            elif abs(gp.pdgId) == 25:
                if isHadronic(gp):
                    hadGenHs.append(gp)
                elif isTau(gp):
                    tauGenHs.append(gp)
                         
        for parton in itertools.chain(lepGenTops, hadGenTops, tauGenTops):
            parton.daus = (parton.genB, genparts[parton.genW.dauIdx[0]], genparts[parton.genW.dauIdx[1]])
            parton.genW.daus = parton.daus[1:]
        for parton in itertools.chain(hadGenWs, hadGenZs, hadGenHs, tauGenWs, tauGenZs, tauGenHs):
            parton.daus = (genparts[parton.dauIdx[0]], genparts[parton.dauIdx[1]])
            
        for fj in fatjets:
            fj.genH, fj.dr_H, fj.genHidx = closest(fj, hadGenHs+tauGenHs)
            fj.genZ, fj.dr_Z, fj.genZidx = closest(fj, hadGenZs+tauGenZs)
            fj.genW, fj.dr_W, fj.genWidx = closest(fj, hadGenWs+tauGenWs)
            fj.genT, fj.dr_T, fj.genTidx = closest(fj, hadGenTops+tauGenTops)
            fj.genLepT, fj.dr_LepT, fj.genLepidx = closest(fj, lepGenTops)

        hadGenHs.sort(key=lambda x: x.pt, reverse = True)
        tauGenHs.sort(key=lambda x: x.pt, reverse = True)
        return hadGenHs+tauGenHs
               
    def selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for lepton counting
        event.looseTaus = [] # store taus
        
        electrons = Collection(event, "Electron")
        for el in electrons:
            el.Id = el.charge * (-11)
            el.kind = self.kTauToElecDecay
            el.mass = 0.000511
            #if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
            if self.Run==2:
                if el.pt > 10 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.045 and abs(el.dz) < 0.2 and el.miniPFRelIso_all <= 0.2 and el.lostHits <= 1 and el.convVeto and el.mvaFall17V2noIso_WP90: #and el.cutBased>3: # cutBased ID: (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                    event.looseLeptons.append(el)
            else:
                if el.pt > 10 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.045 and abs(el.dz) < 0.2 and el.miniPFRelIso_all <= 0.2 and el.lostHits <= 1 and el.convVeto and el.mvaNoIso_WP90: #and el.cutBased>3: # cutBased ID: (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                    event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            mu.Id = mu.charge * (-13)
            mu.kind = self.kTauToMuDecay
            mu.mass = 0.10566
            if mu.pt > 10 and abs(mu.eta) <= 2.4 and abs(mu.dxy) < 0.045 and abs(mu.dz) < 0.2 and mu.mediumId and mu.miniPFRelIso_all <= 0.2: # mu.tightId
                event.looseLeptons.append(mu)

        taus = Collection(event, "Tau")
        for tau in taus:
            tau.Id = tau.charge * (-15)
            tau.kind = self.kTauToHadDecay
            if tau.decayMode==0: tau.mass = 0.13957
            if self.Run==2:
                if tau.pt > 20 and abs(tau.eta) <= 2.3 and abs(tau.dz) < 0.2 and (tau.decayMode in [0,1,2,10,11]) and tau.idDeepTau2017v2p1VSe >= 1 and tau.idDeepTau2017v2p1VSmu >= 1 and tau.idDeepTau2017v2p1VSjet >= 1:
                    event.looseTaus.append(tau) # All loosest WPs. To use later: VVloose VsE (2), VLoose vsMu (1), Loose Vsjet (8)
            else:
                if tau.pt > 20 and abs(tau.eta) <= 2.5 and abs(tau.dz) < 0.2 and (tau.decayMode in [0,1,2,10,11]) and tau.idDeepTau2018v2p5VSe >= 1 and tau.idDeepTau2018v2p5VSmu >= 1 and tau.idDeepTau2018v2p5VSjet >= 1:
                    event.looseTaus.append(tau) # All loosest WPs. To use later: VVloose VsE (2), VLoose vsMu (1), Loose Vsjet (4)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
        if self.Run==2:
            event.looseTaus.sort(key=lambda x: x.rawDeepTau2017v2p1VSjet, reverse=True)
        else:
            event.looseTaus.sort(key=lambda x: x.rawDeepTau2018v2p5VSjet, reverse=True)

        self.nTaus = int(len(event.looseTaus))
        self.nLeps = int(len(event.looseLeptons))

    def correctJetsAndMET(self, event):
        # correct Jets and MET
        event.idx = event._entry if event._tree._entrylist==ROOT.MakeNullPointer(ROOT.TEntryList) else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        #event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
        event.met = METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
        
        # JetMET corrections
        if self._needsJMECorr:
            rho = event.fixedGridRhoFastjetAll
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                             met=event.met, rawMET=METObject(event, "RawMET"),
                                             defaultMET=METObject(event, "MET"),
                                             rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True) 
        
            # correct fatjets
            self.fatjetCorr.setSeed(rndSeed(event, event._allFatJets))
            self.fatjetCorr.correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

            # correct subjets
            self.subjetCorr.setSeed(rndSeed(event, event.subjets))
            self.subjetCorr.correctJetAndMET(jets=event.subjets, met=None, rho=rho,
                                             genjets=Collection(event, self._sj_gen_name) if self.isMC else None,
                                             isMC=self.isMC, runNumber=event.run)

        # all JetMET corrections
        if self._allJME:
            rho = event.fixedGridRhoFastjetAll
            event._AllJets = {}
            event._FatJets = {}
            extra=0
            for key,corr in self.fatjetCorrectors.items():          
                if key=='nominal':
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._allFatJets))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._allFatJets, met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                    # event._FatJets[key] = Collection(event, self._fj_name)
                    # self.fatjetCorrectors[key].setSeed(rndSeed(event, event._FatJets[key], extra))
                    # self.fatjetCorrectors[key].correctJetAndMET(jets=event._FatJets[key], met=None, rho=rho,
                    #                                             genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                    #                                             isMC=self.isMC, runNumber=event.run)
                else:
                    event._FatJets[key] = Collection(event, self._fj_name)
                    self.fatjetCorrectors[key].setSeed(rndSeed(event, event._FatJets[key], extra))
                    self.fatjetCorrectors[key].correctJetAndMET(jets=event._FatJets[key], met=None, rho=rho,
                                                                genjets=Collection(event, self._fj_gen_name) if self.isMC else None,
                                                                isMC=self.isMC, runNumber=event.run)
                    for idx, fj in enumerate(event._FatJets[key]):
                        fj.idx = idx
                        fj.is_qualified = True
                        if self.Run==2:
                            fj.Xbb = (fj.particleNetMD_Xbb/(1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq))
                            #den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCDb + fj.particleNetMD_QCDbb + fj.particleNetMD_QCDc + fj.particleNetMD_QCDcc + fj.particleNetMD_QCDothers
                            den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCD
                            num = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq
                        else:
                            fj.Xbb = fj.particleNet_XbbVsQCD
                            XbbRaw = fj.particleNet_QCD * (fj.particleNet_XbbVsQCD / (1-fj.particleNet_XbbVsQCD))
                            XccRaw = fj.particleNet_QCD * (fj.particleNet_XccVsQCD / (1-fj.particleNet_XccVsQCD))
                            XggRaw = fj.particleNet_QCD * (fj.particleNet_XggVsQCD / (1-fj.particleNet_XggVsQCD))
                            XqqRaw = fj.particleNet_QCD * (fj.particleNet_XqqVsQCD / (1-fj.particleNet_XqqVsQCD))
                            den = XbbRaw + XccRaw + XggRaw + XqqRaw + fj.particleNet_QCD
                            num = XbbRaw + XccRaw + XggRaw + XqqRaw
                        if den>0:
                            fj.Xjj = num/den
                        else:
                            fj.Xjj = -1
                        fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
                        fj.t21 = (fj.tau2/fj.tau1) if fj.tau1 > 0 else -1
                        fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]

            for key,corr in self.jetmetCorrectors.items():
                rho = event.fixedGridRhoFastjetAll
                if key=='nominal':
                    corr.setSeed(rndSeed(event, event._allJets))
                    corr.correctJetAndMET(jets=event._allJets, lowPtJets=Collection(event, "CorrT1METJet"),
                                          met=event.met, rawMET=METObject(event, "RawMET"),
                                          defaultMET=METObject(event, "MET"),
                                          rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                          isMC=self.isMC, runNumber=event.run)
                    event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)
                else:
                    event._AllJets[key] = Collection(event, "Jet")
                    corr.setSeed(rndSeed(event, event._AllJets[key], extra))
                    corr.correctJetAndMET(jets=event._AllJets[key], lowPtJets=Collection(event, "CorrT1METJet"),
                                          met=event.met, rawMET=METObject(event, "RawMET"),
                                          defaultMET=METObject(event, "MET"),
                                          rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                                          isMC=self.isMC, runNumber=event.run)
                    event._AllJets[key] = sorted(event._AllJets[key], key=lambda x: x.pt, reverse=True)

        # link fatjet to subjets 
        for idx, fj in enumerate(event._allFatJets):
            fj.idx = idx
            fj.is_qualified = True
            fj.subjets = get_subjets(fj, event.subjets, ('subJetIdx1', 'subJetIdx2'))
            if self.Run==2:
                if (1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq) > 0:
                    fj.Xbb = (fj.particleNetMD_Xbb/(1. - fj.particleNetMD_Xcc - fj.particleNetMD_Xqq))
                else: 
                    fj.Xbb = -1
                #den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCDb + fj.particleNetMD_QCDbb + fj.particleNetMD_QCDc + fj.particleNetMD_QCDcc + fj.particleNetMD_QCDothers
                den = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq + fj.particleNetMD_QCD
                num = fj.particleNetMD_Xbb + fj.particleNetMD_Xcc + fj.particleNetMD_Xqq
            else:
                fj.Xbb = fj.particleNet_XbbVsQCD
                fj.Xtautau = fj.particleNet_XttVsQCD
                fj.Xtaumu = fj.particleNet_XtmVsQCD
                fj.Xtaue = fj.particleNet_XteVsQCD
                if fj.particleNet_XttVsQCD<1.0 and fj.particleNet_XtmVsQCD<1.0 and fj.particleNet_XteVsQCD<1.0:
                    XttRaw = fj.particleNet_QCD * (fj.particleNet_XttVsQCD / (1-fj.particleNet_XttVsQCD))
                    XtmRaw = fj.particleNet_QCD * (fj.particleNet_XtmVsQCD / (1-fj.particleNet_XtmVsQCD))
                    XteRaw = fj.particleNet_QCD * (fj.particleNet_XteVsQCD / (1-fj.particleNet_XteVsQCD))
                    den = XttRaw + XtmRaw + XteRaw + fj.particleNet_QCD
                    num = XttRaw + XtmRaw + XteRaw
                else: # QCDraw must be negligibly small in this case
                    den = 1.0
                    num = 1.0
                if den>0:
                    fj.Xtauany = num/den
                else:
                    fj.Xtauany = -1
                if fj.particleNet_XbbVsQCD<1.0 and fj.particleNet_XccVsQCD<1.0 and fj.particleNet_XggVsQCD<1.0 and fj.particleNet_XqqVsQCD<1.0:
                    XbbRaw = fj.particleNet_QCD * (fj.particleNet_XbbVsQCD / (1-fj.particleNet_XbbVsQCD))
                    XccRaw = fj.particleNet_QCD * (fj.particleNet_XccVsQCD / (1-fj.particleNet_XccVsQCD))
                    XggRaw = fj.particleNet_QCD * (fj.particleNet_XggVsQCD / (1-fj.particleNet_XggVsQCD))
                    XqqRaw = fj.particleNet_QCD * (fj.particleNet_XqqVsQCD / (1-fj.particleNet_XqqVsQCD))
                    den = XbbRaw + XccRaw + XggRaw + XqqRaw + fj.particleNet_QCD
                    num = XbbRaw + XccRaw + XggRaw + XqqRaw
                else: # QCDraw must be negligibly small in this case
                    den = 1.0
                    num = 1.0
            if den>0:
                fj.Xjj = num/den
            else:
                fj.Xjj = -1
            fj.t32 = (fj.tau3/fj.tau2) if fj.tau2 > 0 else -1
            fj.t21 = (fj.tau2/fj.tau1) if fj.tau1 > 0 else -1
            if self.isMC:
                fj.msoftdropJMS = fj.msoftdrop*self._jmsValues[0]
            else:
                fj.msoftdropJMS = fj.msoftdrop

            # do we need to recompute the softdrop mass?
            # fj.msoftdrop = sumP4(*fj.subjets).M()
            
            corr_mass_JMRUp = random.gauss(0.0, self._jmrValues[2] - 1.)
            corr_mass = max(self._jmrValues[0]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            corr_mass_JMRDown = max(self._jmrValues[1]-1.,0.)/(self._jmrValues[2]-1.) * corr_mass_JMRUp
            fj.msoftdrop_corr = fj.msoftdropJMS*(1.+corr_mass)
            fj.msoftdrop_JMS_Down = fj.msoftdrop_corr*(self._jmsValues[1]/self._jmsValues[0])
            fj.msoftdrop_JMS_Up = fj.msoftdrop_corr*(self._jmsValues[2]/self._jmsValues[0])
            fj.msoftdrop_JMR_Down = fj.msoftdropJMS*(1.+corr_mass_JMRDown)
            fj.msoftdrop_JMR_Up = fj.msoftdropJMS*(1.+corr_mass_JMRUp)


        # sort fat jets
        event._xbbFatJets = sorted(event._allFatJets, key=lambda x: x.Xbb, reverse = True) # sort by PnXbb score
        
        # select jets
        event.fatjets = [fj for fj in event._xbbFatJets if fj.pt > 200 and abs(fj.eta) < 2.5 and (fj.jetId & 2)]
        #event.ak4jets = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and (j.jetId & 2)]

        if self.Run==2:
            puid = 3 if '2016' in self.year  else 6
            # process puid 3(Medium) or 7(Tight) for 2016 and 6(Medium) or 7(Tight) for 2017/18
            ak4jets_unclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 6 and ((j.puId == puid or j.puId == 7) or j.pt > 50)]
        else:
            # No puid in Run3, because "default" jets are PuppiJets
            ak4jets_unclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 2]
        # Clean Jets from Taus and Leptons
        event.ak4jets = []
        for j in ak4jets_unclean:
            goodjet = True
            for l in event.looseLeptons: # +event.looseTaus # Don't clean with Taus yet, when Taus have only very loostest WPs applied -> Might be genuine Jets
                if j.DeltaR(l) < 0.5:
                    goodjet = False
                    break
            if goodjet: event.ak4jets.append(j)
        looseMuons = [l for l in event.looseLeptons if abs(l.Id)==13]
        looseElectrons = [l for l in event.looseLeptons if abs(l.Id)==11]
        for j in event.ak4jets:
            j.hasMuon = True if (closest(j, looseMuons)[1] < 1.0) else False
            j.hasElectron = True if (closest(j, looseElectrons)[1] < 1.0) else False
        for j in event.fatjets:
            j.hasMuon = True if (closest(j, looseMuons)[1] < 1.0) else False
            j.hasElectron = True if (closest(j, looseElectrons)[1] < 1.0) else False

        self.nFatJets = int(len(event.fatjets))
        self.nSmallJets = int(len(event.ak4jets))

        event.ht = sum([j.pt for j in event.ak4jets])

        # b-tag AK4 jet selection - these jets don't have a kinematic selection
        event.bljets = []
        event.bmjets = []
        event.btjets = []
        event.bmjetsCSV = []
        for j in event._allJets:
            #overlap = False
            #for fj in event.fatjets:
            #    if deltaR(fj,j) < 1.0: overlap = True # calculate overlap between small r jets and fatjets
            #if overlap: continue
            if self.Run==2:
                pNetSum = j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc + j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds
                if pNetSum > 0:
                    j.btagPNetB = (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb) / pNetSum
                else:
                    j.btagPNetB = -1
            else:
                pass
                # "btagPNetB" already defined with that exact branch name in NanoAODv12

            if j.btagDeepFlavB > self.DeepFlavB_WP_L:
                event.bljets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_M:
                event.bmjets.append(j)
            if j.btagDeepFlavB > self.DeepFlavB_WP_T:
                event.btjets.append(j)
            if self.Run==2:
                if j.btagDeepB > self.DeepCSV_WP_M:
                    event.bmjetsCSV.append(j)

        jpt_thr = 20; jeta_thr = 2.5;
        if self.year in ["2016APV", "2016"]:
            jpt_thr = 30; jeta_thr = 2.4;
        #event.bmjets = [j for j in event.bmjets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4) and (j.puId >=2)]
        event.bmjets = [j for j in event.bmjets if j.pt > jpt_thr and abs(j.eta) < jeta_thr]
        if self.Run==2:
            event.bljets = [j for j in event.bljets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4) and (j.puId >=2)]
        else:
            event.bljets = [j for j in event.bljets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4)]
        #event.alljets = [j for j in event.alljets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId >= 4) and (j.puId >=2)]
        #event.alljets = [j for j in event.alljets if j.pt > jpt_thr and abs(j.eta) < jeta_thr and (j.jetId == 2) and (j.puId >=2)]

        event.bmjets.sort(key=lambda x : x.pt, reverse = True)
        event.bljets.sort(key=lambda x : x.pt, reverse = True)
        #event.alljets.sort(key=lambda x : x.pt, reverse = True)
        #event.ak4jets.sort(key=lambda x : x.btagDeepFlavB, reverse = True)
        event.ak4jets.sort(key=lambda x : x.btagPNetB, reverse = True)

        self.nBTaggedJets = int(len(event.bmjets))
        #self.nBTaggedJets = int(len(event.bljets))

        # sort and select variations of jets
        if self._allJME:
            event.fatjetsJME = {}
            for syst in self._jmeLabels:
                if syst == 'nominal': 
                    continue
                ptordered = sorted(event._FatJets[syst], key=lambda x: x.pt, reverse=True)
                xbbordered = sorted(event._FatJets[syst], key=lambda x: x.Xbb, reverse = True) 
                event.fatjetsJME[syst] = [fj for fj in xbbordered if fj.pt > 200 and abs(fj.eta) < 2.4 and (fj.jetId & 2)]
                """
                if 'EC2' in syst:
                    for ifj, fj in enumerate(event.fatjets):
                        if fj.pt !=  event.fatjetsJME[syst][ifj].pt:
                            print('%s: ifj %i diff '%(syst,ifj),'nominal pt: ',fj.pt,' eta: ',fj.eta,' JES_syst pt: ',event.fatjetsJME[syst][ifj].pt,' nominal ',event.fatjetsJME['nominal'][ifj].pt)
                if syst == 'nominal':   
                    continue
                """
                
    def evalMassRegression(self, event, jets):
        for i,j in enumerate(jets):
            if self._opts['run_mass_regression']:
                outputs = [p.predict_with_cache(self.tagInfoMaker, event.idx, j.idx, j) for p in self.pnMassRegressions]
                j.regressed_mass = ensemble(outputs, np.median)['mass']
                if self.isMC:
                    j.regressed_massJMS = j.regressed_mass*self._jmsValuesReg[0]
                else:
                    j.regressed_massJMS = j.regressed_mass

                corr_mass_JMRUp = random.gauss(0.0, self._jmrValuesReg[2] - 1.)
                corr_mass = max(self._jmrValuesReg[0]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp
                corr_mass_JMRDown = max(self._jmrValuesReg[1]-1.,0.)/(self._jmrValuesReg[2]-1.) * corr_mass_JMRUp

                j.regressed_mass_corr = j.regressed_massJMS*(1.+corr_mass)
                j.regressed_mass_JMS_Down = j.regressed_mass_corr*(self._jmsValuesReg[1]/self._jmsValuesReg[0])
                j.regressed_mass_JMS_Up = j.regressed_mass_corr*(self._jmsValuesReg[2]/self._jmsValuesReg[0])
                j.regressed_mass_JMR_Down = j.regressed_massJMS*(1.+corr_mass_JMRDown)
                j.regressed_mass_JMR_Up = j.regressed_massJMS*(1.+corr_mass_JMRUp)

                if self._allJME:
                    for syst in self._jmeLabels:
                        if syst == 'nominal': continue
                        if len(event.fatjetsJME[syst])>i:
                            event.fatjetsJME[syst][i].regressed_mass = j.regressed_mass
                            event.fatjetsJME[syst][i].regressed_massJMS = j.regressed_massJMS
                            
            else:
                j.regressed_mass = 0          
                j.regressed_massJMS = 0

    def fillBaseEventInfo(self, event, fatjets, hadGenHs):
        self.out.fillBranch("ht", event.ht)
        if self.Run==2:
            self.out.fillBranch("rho", event.fixedGridRhoFastjetAll)
        else:
            self.out.fillBranch("rho", event.Rho_fixedGridRhoFastjetAll)
        self.out.fillBranch("met", event.met.pt)
        self.out.fillBranch("metphi", event.met.phi)
        self.out.fillBranch("weight", event.gweight)
        #self.out.fillBranch("npvs", event.PV.npvs)

        # qcd weights
        """
        https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio
        ['LHE scale variation weights (w_var / w_nominal)',
        ' [0] is renscfact=0.5d0 facscfact=0.5d0 ',
        ' [1] is renscfact=0.5d0 facscfact=1d0 ',
        ' [2] is renscfact=0.5d0 facscfact=2d0 ',
        ' [3] is renscfact=1d0 facscfact=0.5d0 ',
        ' [4] is renscfact=1d0 facscfact=1d0 ',
        ' [5] is renscfact=1d0 facscfact=2d0 ',
        ' [6] is renscfact=2d0 facscfact=0.5d0 ',
        ' [7] is renscfact=2d0 facscfact=1d0 ',
        ' [8] is renscfact=2d0 facscfact=2d0 ']
        """
        # compute envelope for weights [1,2,3,4,6,8]?

        # for PDF weights
        # need to determine if there are replicas or hessian eigenvectors?
        # 
        # if len(event.LHEPdfWeight)>0:
        # (1) get average of weights
        # (2) then sum ( weight - average )**2
        # (3) then take sqrt(sum/(nweights-1))
        # weight up: 1.0+stddev, down: 1.0-stddev (max and min of 13?)

        met_filters = bool(
            event.Flag_goodVertices and
            event.Flag_globalSuperTightHalo2016Filter and
            event.Flag_EcalDeadCellTriggerPrimitiveFilter and
            event.Flag_BadPFMuonFilter and
            event.Flag_BadPFMuonDzFilter and
            event.Flag_eeBadScFilter
        )
        if self.year not in ["2016APV", "2016"]:
            #met_filters = met_filters and event.Flag_ecalBadCalibFilterV2
            met_filters = met_filters and event.Flag_ecalBadCalibFilter and event.Flag_ecalBadCalibFilter
        if self.Run==2:
            met_filters = met_filters and event.Flag_HBHENoiseFilter and event.Flag_HBHENoiseIsoFilter
        self.out.fillBranch("passmetfilters", met_filters)

        # L1 prefire weights
        if self.isMC and self.Run==2:
            self.out.fillBranch("l1PreFiringWeight", event.L1PreFiringWeight_Nom)
            self.out.fillBranch("l1PreFiringWeightUp", event.L1PreFiringWeight_Up)
            self.out.fillBranch("l1PreFiringWeightDown", event.L1PreFiringWeight_Dn)
        else:
            self.out.fillBranch("l1PreFiringWeight", 1.0)
            self.out.fillBranch("l1PreFiringWeightUp", 1.0)
            self.out.fillBranch("l1PreFiringWeightDown", 1.0)

        # trigger weights
        tweight = 1.0
        tweight_mc = 1.0
        tweight_3d = 1.0
        tweight_3d_mc = 1.0
        if self.isMC:
            if len(fatjets)>1:
                tweight = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS))
                tweight_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, -1, True))
                tweight_3d = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb))
                tweight_3d_mc = 1.0 - (1.0 - self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True))*(1.0 - self._teff.getEfficiency(fatjets[1].pt, fatjets[1].msoftdropJMS, fatjets[1].Xbb, True))
            else:
                if len(fatjets)>0:
                    tweight = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS)
                    tweight_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, -1, True)
                    tweight_3d = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb)
                    tweight_3d_mc = self._teff.getEfficiency(fatjets[0].pt, fatjets[0].msoftdropJMS, fatjets[0].Xbb, True)
        self.out.fillBranch("triggerEffWeight", tweight)
        self.out.fillBranch("triggerEff3DWeight", tweight_3d)
        self.out.fillBranch("triggerEffMCWeight", tweight_mc)
        self.out.fillBranch("triggerEffMC3DWeight", tweight_3d_mc)

        # fill gen higgs info
        def HDecayMode(higgs):
            def FindTauDecay(gp):
                for dau in gp.dauIdx: # First check for other copies
                    if abs(event.genparts[dau].pdgId)==15: return FindTauDecay(event.genparts[dau])
                for dau in gp.dauIdx: # Then look for electrons or muons
                    pdgid = event.genparts[dau].pdgId
                    if abs(pdgid) in [11,13] and event.genparts[dau].statusFlags >> 5 & 1 == 1 and sorted([event.genparts[d].pdgId for d in gp.dauIdx]) == sorted([pdgid, int(-(abs(pdgid)+1)*pdgid/abs(pdgid)), int((abs(gp.pdgId)+1)*gp.pdgId/abs(gp.pdgId))]):
                        return pdgid
                return gp.pdgId # hadronic Tau; return initial pdgId +/-15

            hdecay = 0
            if len(higgs.dauIdx)==2:
                if abs(event.genparts[higgs.dauIdx[0]].pdgId)==abs(event.genparts[higgs.dauIdx[1]].pdgId):
                    hdecay = event.genparts[higgs.dauIdx[0]].pdgId * event.genparts[higgs.dauIdx[1]].pdgId
            if hdecay==-15*15: # Tau decay mode, but hadronic or leptonic?
                hdecay = FindTauDecay(event.genparts[higgs.dauIdx[0]])*FindTauDecay(event.genparts[higgs.dauIdx[1]])
            return hdecay

        if hadGenHs and self.isMC:
            if len(hadGenHs)>0:
                self.out.fillBranch("genHiggs1Pt", hadGenHs[0].pt)
                self.out.fillBranch("genHiggs1Eta", hadGenHs[0].eta)
                self.out.fillBranch("genHiggs1Phi", hadGenHs[0].phi)
                self.out.fillBranch("genHiggs1Decay", HDecayMode(hadGenHs[0]))
                self.out.fillBranch("genHiggsDau1Pt", event.genparts[hadGenHs[0].dauIdx[0]].pt)
                self.out.fillBranch("genHiggsDau1Eta", event.genparts[hadGenHs[0].dauIdx[0]].eta)
                self.out.fillBranch("genHiggsDau1Phi", event.genparts[hadGenHs[0].dauIdx[0]].phi)
                self.out.fillBranch("genHiggsDau1Id", event.genparts[hadGenHs[0].dauIdx[0]].pdgId)
                self.out.fillBranch("genHiggsDau2Pt", event.genparts[hadGenHs[0].dauIdx[1]].pt)
                self.out.fillBranch("genHiggsDau2Eta", event.genparts[hadGenHs[0].dauIdx[1]].eta)
                self.out.fillBranch("genHiggsDau2Phi", event.genparts[hadGenHs[0].dauIdx[1]].phi)
                self.out.fillBranch("genHiggsDau2Id", event.genparts[hadGenHs[0].dauIdx[1]].pdgId)
                if len(hadGenHs)>1:
                    self.out.fillBranch("genHiggs2Pt", hadGenHs[1].pt)
                    self.out.fillBranch("genHiggs2Eta", hadGenHs[1].eta)
                    self.out.fillBranch("genHiggs2Phi", hadGenHs[1].phi)
                    self.out.fillBranch("genHiggs2Decay", HDecayMode(hadGenHs[1]))
                    self.out.fillBranch("genHiggsDau3Pt", event.genparts[hadGenHs[1].dauIdx[0]].pt)
                    self.out.fillBranch("genHiggsDau3Eta", event.genparts[hadGenHs[1].dauIdx[0]].eta)
                    self.out.fillBranch("genHiggsDau3Phi", event.genparts[hadGenHs[1].dauIdx[0]].phi)
                    self.out.fillBranch("genHiggsDau3Id", event.genparts[hadGenHs[1].dauIdx[0]].pdgId)
                    self.out.fillBranch("genHiggsDau4Pt", event.genparts[hadGenHs[1].dauIdx[1]].pt)
                    self.out.fillBranch("genHiggsDau4Eta", event.genparts[hadGenHs[1].dauIdx[1]].eta)
                    self.out.fillBranch("genHiggsDau4Phi", event.genparts[hadGenHs[1].dauIdx[1]].phi)
                    self.out.fillBranch("genHiggsDau4Id", event.genparts[hadGenHs[1].dauIdx[1]].pdgId)

                    if len(hadGenHs)>2:
                        self.out.fillBranch("genHiggs3Pt", hadGenHs[2].pt)
                        self.out.fillBranch("genHiggs3Eta", hadGenHs[2].eta)
                        self.out.fillBranch("genHiggs3Phi", hadGenHs[2].phi)
                        self.out.fillBranch("genHiggs3Decay", HDecayMode(hadGenHs[2]))
                        self.out.fillBranch("genHiggsDau5Pt", event.genparts[hadGenHs[2].dauIdx[0]].pt)
                        self.out.fillBranch("genHiggsDau5Eta", event.genparts[hadGenHs[2].dauIdx[0]].eta)
                        self.out.fillBranch("genHiggsDau5Phi", event.genparts[hadGenHs[2].dauIdx[0]].phi)
                        self.out.fillBranch("genHiggsDau5Id", event.genparts[hadGenHs[2].dauIdx[0]].pdgId)
                        self.out.fillBranch("genHiggsDau6Pt", event.genparts[hadGenHs[2].dauIdx[1]].pt)
                        self.out.fillBranch("genHiggsDau6Eta", event.genparts[hadGenHs[2].dauIdx[1]].eta)
                        self.out.fillBranch("genHiggsDau6Phi", event.genparts[hadGenHs[2].dauIdx[1]].phi)
                        self.out.fillBranch("genHiggsDau6Id", event.genparts[hadGenHs[2].dauIdx[1]].pdgId)

    def _get_filler(self, obj):
        def filler(branch, value, default=0):
            self.out.fillBranch(branch, value if obj else default)
        return filler

    def fillFatJetInfo(self, event, fatjets):
        # hh system
        if len(fatjets) > 0:
            h1Jet = polarP4(fatjets[0],mass='msoftdropJMS')
            h2Jet = polarP4(None)

        if len(fatjets)>1:
            h2Jet = polarP4(fatjets[1],mass='msoftdropJMS')

            if self.isMC:
                h1Jet_JMS_Down = polarP4(fatjets[0],mass='msoftdrop_JMS_Down')
                h2Jet_JMS_Down = polarP4(fatjets[1],mass='msoftdrop_JMS_Down')
                h1Jet_JMS_Up = polarP4(fatjets[0],mass='msoftdrop_JMS_Up')
                h2Jet_JMS_Up = polarP4(fatjets[1],mass='msoftdrop_JMS_Up')

                h1Jet_JMR_Down = polarP4(fatjets[0],mass='msoftdrop_JMR_Down')
                h2Jet_JMR_Down = polarP4(fatjets[1],mass='msoftdrop_JMR_Down')
                h1Jet_JMR_Up = polarP4(fatjets[0],mass='msoftdrop_JMR_Up')
                h2Jet_JMR_Up = polarP4(fatjets[1],mass='msoftdrop_JMR_Up')

        if len(fatjets)>2:
            h3Jet = polarP4(fatjets[2],mass='msoftdropJMS')

            if self.isMC:
                h1Jet_JMS_Down = polarP4(fatjets[0],mass='msoftdrop_JMS_Down')
                h2Jet_JMS_Down = polarP4(fatjets[1],mass='msoftdrop_JMS_Down')
                h3Jet_JMS_Down = polarP4(fatjets[2],mass='msoftdrop_JMS_Down')

                h1Jet_JMS_Up = polarP4(fatjets[0],mass='msoftdrop_JMS_Up')
                h2Jet_JMS_Up = polarP4(fatjets[1],mass='msoftdrop_JMS_Up')
                h3Jet_JMS_Up = polarP4(fatjets[2],mass='msoftdrop_JMS_Up')

                h1Jet_JMR_Down = polarP4(fatjets[0],mass='msoftdrop_JMR_Down')
                h2Jet_JMR_Down = polarP4(fatjets[1],mass='msoftdrop_JMR_Down')
                h3Jet_JMR_Down = polarP4(fatjets[2],mass='msoftdrop_JMR_Down')

                h1Jet_JMR_Up = polarP4(fatjets[0],mass='msoftdrop_JMR_Up')
                h2Jet_JMR_Up = polarP4(fatjets[1],mass='msoftdrop_JMR_Up')
                h3Jet_JMR_Up = polarP4(fatjets[2],mass='msoftdrop_JMR_Up')


        for idx in ([1, 2, 3, 4]):
            prefix = 'fatJet%i' % idx
            fj = fatjets[idx-1] if len(fatjets)>idx-1 else _NullObject()
            if idx==4 and len(fatjets)>3:
                # Make sure to have the FatJet with highest Xtautau score
                # If we have it already, 4. FatJet is just the 4th-best Xbb
                # Otherwise, the 4th is the Tau-Higgs candidate
                if self.Run!=2:
                    probetau = sorted(fatjets, key=lambda x: x.Xtautau, reverse = True)[0]
                    if probetau!=fatjets[0] and probetau!=fatjets[1] and probetau!=fatjets[2]:
                        fj = probetau
            fill_fj = self._get_filler(fj)
            fill_fj(prefix + "Pt", fj.pt)
            fill_fj(prefix + "Eta", fj.eta)
            fill_fj(prefix + "Phi", fj.phi)
            fill_fj(prefix + "RawFactor", fj.rawFactor)
            if self.Run==2:
                fill_fj(prefix + "Mass", fj.particleNet_mass)
            else:
                if fj.mass is not None:
                    fill_fj(prefix + "Mass", fj.mass*fj.particleNet_massCorr)
                else: # Crash when trying to do "None*None"
                    fill_fj(prefix + "Mass", fj.mass)
            fill_fj(prefix + "MassSD_UnCorrected", fj.msoftdrop)
            fill_fj(prefix + "PNetXbb", fj.Xbb)
            fill_fj(prefix + "PNetXjj", fj.Xjj)
            if self.Run==2:
                fill_fj(prefix + "PNetQCD", fj.particleNetMD_QCD)
            else:
                fill_fj(prefix + "PNetXtautau", fj.Xtautau)
                fill_fj(prefix + "PNetXtaumu", fj.Xtaumu)
                fill_fj(prefix + "PNetXtaue", fj.Xtaue)
                fill_fj(prefix + "PNetXtauany", fj.Xtauany)
                fill_fj(prefix + "PNetQCD", fj.particleNet_QCD)
            fill_fj(prefix + "Area", fj.area)
            if self.isMC:
                fill_fj(prefix + "HiggsMatched", fj.HiggsMatch)
                fill_fj(prefix + "HiggsMatchedIndex", fj.HiggsMatchIndex)
                fill_fj(prefix + "MatchedGenPt", fj.MatchedGenPt)

            fill_fj(prefix + "n3b1", fj.n3b1)
            fill_fj(prefix + "Tau2OverTau1", fj.t21)
            fill_fj(prefix + "Tau3OverTau2", fj.t32)
            
            # uncertainties
            if self.isMC:
                fill_fj(prefix + "MassSD_noJMS", fj.msoftdrop)
                fill_fj(prefix + "MassSD", fj.msoftdrop_corr)
                fill_fj(prefix + "MassSD_JMS_Down", fj.msoftdrop_JMS_Down)
                fill_fj(prefix + "MassSD_JMS_Up",  fj.msoftdrop_JMS_Up)
                fill_fj(prefix + "MassSD_JMR_Down", fj.msoftdrop_JMR_Down)
                fill_fj(prefix + "MassSD_JMR_Up",  fj.msoftdrop_JMR_Up)
            else:
                fill_fj(prefix + "MassSD_noJMS", fj.msoftdrop)
                fill_fj(prefix + "MassSD", fj.msoftdropJMS)
            
            # overlap variables
            if fj:
                hasBJetCSVLoose = True if (closest(fj, event.bljets)[1] < 1.0) else False
                hasBJetCSVMedium = True if (closest(fj, event.bmjetsCSV)[1] < 1.0) else False
                hasBJetCSVTight = True if (closest(fj, event.btjets)[1] < 1.0) else False
            else:
                hasBJetCSVLoose = False
                hasBJetCSVMedium = False
                hasBJetCSVTight = False
            fill_fj(prefix + "HasMuon", fj.hasMuon)
            fill_fj(prefix + "HasElectron", fj.hasElectron)
            fill_fj(prefix + "HasBJetCSVLoose", hasBJetCSVLoose)
            fill_fj(prefix + "HasBJetCSVMedium", hasBJetCSVMedium)
            fill_fj(prefix + "HasBJetCSVTight", hasBJetCSVTight)

            nb_fj_opp_ = 0
            for j in event.bmjetsCSV:
                if fj:
                    if abs(deltaPhi(j, fj)) > 2.5 and j.pt>25:
                        nb_fj_opp_ += 1
            hasBJetOpp = True if (nb_fj_opp_>0) else False
            fill_fj(prefix + "OppositeHemisphereHasBJet", hasBJetOpp)
            if fj:
                fill_fj(prefix + "NSubJets", len(fj.subjets))

            # hh variables
            ptovermsd = -1 
            ptovermregressed = -1 
            if fj:
                ptovermsd = -1 if fj.msoftdropJMS<=0 else fj.pt/fj.msoftdropJMS
                ptovermregressed = -1 if fj.regressed_massJMS<=0 else fj.pt/fj.regressed_massJMS
                if (h1Jet+h2Jet).M()>0:
                    fill_fj(prefix + "PtOverMHH", fj.pt/(h1Jet+h2Jet).M())
                else:
                    # print('hh mass 0?',(h1Jet+h2Jet).M())
                    fill_fj(prefix + "PtOverMHH", -1)
            else:
                fill_fj(prefix + "PtOverMHH", -1)
            fill_fj(prefix + "PtOverMSD", ptovermsd)
            fill_fj(prefix + "PtOverMRegressed", ptovermregressed)

            if self.isMC:
                if len(fatjets)>1 and fj:
                    fill_fj(prefix + "PtOverMHH_JMS_Down", fj.pt/(h1Jet_JMS_Down+h2Jet_JMS_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMS_Up", fj.pt/(h1Jet_JMS_Up+h2Jet_JMS_Up).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Down", fj.pt/(h1Jet_JMR_Down+h2Jet_JMR_Down).M())
                    fill_fj(prefix + "PtOverMHH_JMR_Up", fj.pt/(h1Jet_JMR_Up+h2Jet_JMR_Up).M())
                else:
                    fill_fj(prefix + "PtOverMHH_JMS_Down",0)
                    fill_fj(prefix + "PtOverMHH_JMS_Up", 0)
                    fill_fj(prefix + "PtOverMHH_JMR_Down", 0)
                    fill_fj(prefix + "PtOverMHH_JMR_Up",0)

            # matching variables
            if self.isMC:
                # info of the closest genH
                fill_fj(prefix + "GenMatchIndex", fj.genHidx if fj.genHidx else -1)

    def fillFatJetInfoJME(self, event, fatjets):
        if not self._allJME or not self.isMC: return
        for syst in self._jmeLabels:
            if syst == 'nominal': continue
            if len(event.fatjetsJME[syst]) < 2 or len(fatjets)<2: 
                for idx in ([1, 2, 3, 4]):
                    prefix = 'fatJet%i' % idx
                    self.out.fillBranch(prefix + "Pt" + "_" + syst, 0)
                    self.out.fillBranch(prefix + "PtOverMHH" + "_" + syst, 0)
            else:
                h1Jet = polarP4(event.fatjetsJME[syst][0],mass='msoftdropJMS')
                h2Jet = polarP4(event.fatjetsJME[syst][1],mass='msoftdropJMS')

                """
                if 'EC2' in syst and ((event.fatjetsJME[syst][0].pt!=fatjets[0].pt) or (event.fatjetsJME[syst][1].pt!=fatjets[1].pt)):
                    h1Jet_nom = polarP4(fatjets[0],mass='msoftdropJMS') 
                    h2Jet_nom = polarP4(fatjets[1],mass='msoftdropJMS')
                    print('EC2 hh different! %s'%syst)
                    print('hh_mass, nominal: %.4f, syst: %.4f'%((h1Jet_nom+h2Jet_nom).M(),(h1Jet+h2Jet).M()))
                    print('fj1pt, nominal: %.4f, syst: %.4f'%(fatjets[0].pt,event.fatjetsJME[syst][0].pt))
                    print('fj2pt, nominal: %.4f, syst: %.4f'%(fatjets[1].pt,event.fatjetsJME[syst][1].pt))
                """

                for idx in ([1, 2, 3, 4]):
                    prefix = 'fatJet%i' % idx
                    fj = event.fatjetsJME[syst][idx - 1]
                    fill_fj = self._get_filler(fj)
                    fill_fj(prefix + "Pt" + "_" + syst, fj.pt)
                    fill_fj(prefix + "PtOverMHH" + "_" + syst, fj.pt/(h1Jet+h2Jet).M())

    def fillJetInfo(self, event, jets, fatjets, XbbWP, taus, XtautauWP, leptons):
        self.out.fillBranch("nbtags", self.nBTaggedJets)
        self.out.fillBranch("nsmalljets",self.nSmallJets)
        self.out.fillBranch("ntaus",self.nTaus)
        self.out.fillBranch("nleps",self.nLeps)
        self.out.fillBranch("nfatjets", self.nFatJets)
        for idx in ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):
            j = jets[idx-1] if len(jets)>idx-1 else _NullObject()
            prefix = 'jet%i'%(idx)
            fillBranch = self._get_filler(j)
            fillBranch(prefix + "Pt", j.pt)
            fillBranch(prefix + "Eta", j.eta)
            fillBranch(prefix + "Phi", j.phi)
            fillBranch(prefix + "DeepFlavB", j.btagDeepFlavB)
            fillBranch(prefix + "PNetB", j.btagPNetB)
            fillBranch(prefix + "JetId", j.jetId)
            fillBranch(prefix + "Mass", j.mass)
            fillBranch(prefix + "RawFactor", j.rawFactor)
            fillBranch(prefix + "Area", j.area)
            if self.Run==2:
                fillBranch(prefix + "PuId", j.puId)
                fillBranch(prefix + "bRegCorr", j.bRegCorr)
                fillBranch(prefix + "bRegRes", j.bRegRes)
                fillBranch(prefix + "cRegCorr", j.cRegCorr)
                fillBranch(prefix + "cRegRes", j.cRegRes)
            if self.isMC:
                fillBranch(prefix + "HadronFlavour", j.hadronFlavour)
                fillBranch(prefix + "HiggsMatched", j.HiggsMatch)
                fillBranch(prefix + "HiggsMatchedIndex", j.HiggsMatchIndex)
                fillBranch(prefix + "FatJetMatched", j.FatJetMatch)
                fillBranch(prefix + "FatJetMatchedIndex", j.FatJetMatchIndex)
                fillBranch(prefix + "MatchedGenPt", j.MatchedGenPt)
            fillBranch(prefix + "HasMuon", j.hasMuon)
            fillBranch(prefix + "HasElectron", j.hasElectron)

        if self.isMC:
            hadGenH_4vec = [polarP4(h) for h in self.hadGenHs]
            genHdaughter_4vec = [polarP4(d) for d in self.genHdaughter]

        event.reco6b_Idx = -1
        event.reco4b2t_Idx = -1
        event.reco4b2t_TauIsBoosted = 0
        event.reco4b2t_TauIsResolved = 0
        event.reco4b2t_TauFinalState = 0

        #if len(jets)+2*len(fatjets) > 5:
        if True:
            # Technique 3: mass fitter
            
            #m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5 = self.higgsPairingAlgorithm(event,jets,fatjets,XbbWP)
            AK4PNetBWP = {"2022": {"L": 0.047, "M": 0.245, "T": 0.6734, "XT": 0.7862, "XXT": 0.961}, "2022EE": {"L": 0.0499, "M": 0.2605, "T": 0.6915, "XT": 0.8033, "XXT": 0.9664}, "2023": {"L": 0.0358, "M": 0.1917, "T": 0.6172, "XT": 0.7515, "XXT": 0.9659}, "2023BPix": {"L": 0.0359, "M": 0.1919, "T": 0.6133, "XT": 0.7544, "XXT": 0.9688}}
            if self.year in AK4PNetBWP:
                jetWP = AK4PNetBWP[self.year]["T"]
                jetdiscr="btagPNetB"
            else:
                jetWP = 0.0
                jetdiscr=""
            if self.Run==2:
                TauVsEl=2
                TauVsMu=1
                TauVsJet=8
            else:
                TauVsEl=2
                TauVsMu=1
                TauVsJet=4
            event.reco6b_Idx,m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5,bla1,bla2,bla3 = higgsPairingAlgorithm_v2(event,jets,fatjets,XbbWP,self.isMC,self.Run,jetdicsr=jetdiscr,jetWP=jetWP,taus=taus,TauVsEl=TauVsEl,TauVsMu=TauVsMu,TauVsJet=TauVsJet,XtautauWP=XtautauWP)
                       
            self.out.fillBranch("h1_t3_mass", h1.Mass)
            self.out.fillBranch("h1_t3_pt", h1.pt)
            self.out.fillBranch("h1_t3_eta", h1.eta)
            self.out.fillBranch("h1_t3_phi", h1.phi)
            self.out.fillBranch("h1_t3_match", h1.matchH)
            self.out.fillBranch("h1_t3_dRjets", h1.dRjets)

            self.out.fillBranch("h2_t3_mass", h2.Mass)
            self.out.fillBranch("h2_t3_pt", h2.pt)
            self.out.fillBranch("h2_t3_eta", h2.eta)
            self.out.fillBranch("h2_t3_phi", h2.phi)
            self.out.fillBranch("h2_t3_match", h2.matchH)
            self.out.fillBranch("h2_t3_dRjets", h2.dRjets)

            self.out.fillBranch("h3_t3_mass", h3.Mass)
            self.out.fillBranch("h3_t3_pt", h3.pt)
            self.out.fillBranch("h3_t3_eta", h3.eta)
            self.out.fillBranch("h3_t3_phi", h3.phi)
            self.out.fillBranch("h3_t3_match", h3.matchH)
            self.out.fillBranch("h3_t3_dRjets", h3.dRjets)

            self.out.fillBranch("max_h_eta", max(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("min_h_eta", min(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("max_h_dRjets", max(h1.dRjets,h2.dRjets,h3.dRjets))
            self.out.fillBranch("min_h_dRjets", min(h1.dRjets,h2.dRjets,h3.dRjets))

            self.out.fillBranch("h_fit_mass", m_fit)


            # Technique 3: mass fitter for Taus
            
            #m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5 = self.higgsPairingAlgorithm(event,jets,fatjets,XbbWP,dotaus=True,taus=taus,XtautauWP=XtautauWP)
            event.reco4b2t_Idx,m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5,event.reco4b2t_TauIsBoosted,event.reco4b2t_TauIsResolved,event.reco4b2t_TauFinalState = higgsPairingAlgorithm_v2(event,jets,fatjets,XbbWP,self.isMC,self.Run,jetdicsr=jetdiscr,jetWP=jetWP,dotaus=True,taus=taus,TauVsEl=TauVsEl,TauVsMu=TauVsMu,TauVsJet=TauVsJet,XtautauWP=XtautauWP,leptons=leptons,METvars=[event.PuppiMET_pt, event.PuppiMET_phi, event.MET_covXX, event.MET_covXY, event.MET_covYY])
                       
            self.out.fillBranch("h1_4b2t_mass", h1.Mass)
            self.out.fillBranch("h1_4b2t_pt", h1.pt)
            self.out.fillBranch("h1_4b2t_eta", h1.eta)
            self.out.fillBranch("h1_4b2t_phi", h1.phi)
            self.out.fillBranch("h1_4b2t_match", h1.matchH)
            self.out.fillBranch("h1_4b2t_dRjets", h1.dRjets)

            self.out.fillBranch("h2_4b2t_mass", h2.Mass)
            self.out.fillBranch("h2_4b2t_pt", h2.pt)
            self.out.fillBranch("h2_4b2t_eta", h2.eta)
            self.out.fillBranch("h2_4b2t_phi", h2.phi)
            self.out.fillBranch("h2_4b2t_match", h2.matchH)
            self.out.fillBranch("h2_4b2t_dRjets", h2.dRjets)

            self.out.fillBranch("h3_4b2t_mass", h3.Mass)
            self.out.fillBranch("h3_4b2t_pt", h3.pt)
            self.out.fillBranch("h3_4b2t_eta", h3.eta)
            self.out.fillBranch("h3_4b2t_phi", h3.phi)
            self.out.fillBranch("h3_4b2t_match", h3.matchH)
            self.out.fillBranch("h3_4b2t_dRjets", h3.dRjets)

            self.out.fillBranch("max_h_eta_4b2t", max(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("min_h_eta_4b2t", min(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("max_h_dRjets_4b2t", max(h1.dRjets,h2.dRjets,h3.dRjets))
            self.out.fillBranch("min_h_dRjets_4b2t", min(h1.dRjets,h2.dRjets,h3.dRjets))

            self.out.fillBranch("h_fit_mass_4b2t", m_fit)

        self.out.fillBranch("reco6b_Idx", event.reco6b_Idx)
        self.out.fillBranch("reco4b2t_Idx", event.reco4b2t_Idx)
        self.out.fillBranch("reco4b2t_TauIsBoosted", event.reco4b2t_TauIsBoosted)
        self.out.fillBranch("reco4b2t_TauIsResolved", event.reco4b2t_TauIsResolved)
        self.out.fillBranch("reco4b2t_TauFinalState", event.reco4b2t_TauFinalState)

            
    def fillLeptonInfo(self, event, leptons):
        for idx in ([1, 2, 3, 4]):
            lep = leptons[idx-1]if len(leptons)>idx-1 else _NullObject()
            prefix = 'lep%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Charge", lep.charge)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Mass", lep.mass)
            fillBranch(prefix + "Id", lep.Id)
            if self.isMC:
                fillBranch(prefix + "HiggsMatched", lep.HiggsMatch)
                fillBranch(prefix + "HiggsMatchedIndex", lep.HiggsMatchIndex)
                fillBranch(prefix + "FatJetMatched", lep.FatJetMatch)
                fillBranch(prefix + "FatJetMatchedIndex", lep.FatJetMatchIndex)
                fillBranch(prefix + "MatchedGenPt", lep.MatchedGenPt)
    def fillTauInfo(self, event, leptons):
        for idx in ([1, 2, 3, 4]):
            lep = leptons[idx-1] if len(leptons)>idx-1 else _NullObject()
            prefix = 'tau%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Charge", lep.charge)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Mass", lep.mass)
            fillBranch(prefix + "Id", lep.Id)
            fillBranch(prefix + "decayMode", lep.decayMode)
            if self.Run==2: # TODO: Can switch to v2p5 for Run2UL too, if inputs have branches available
                fillBranch(prefix + "rawDeepTau2017v2p1VSe", lep.rawDeepTau2017v2p1VSe)
                fillBranch(prefix + "rawDeepTau2017v2p1VSmu", lep.rawDeepTau2017v2p1VSmu)
                fillBranch(prefix + "rawDeepTau2017v2p1VSjet", lep.rawDeepTau2017v2p1VSjet)
            else:
                fillBranch(prefix + "rawDeepTau2018v2p5VSe", lep.rawDeepTau2018v2p5VSe)
                fillBranch(prefix + "rawDeepTau2018v2p5VSmu", lep.rawDeepTau2018v2p5VSmu)
                fillBranch(prefix + "rawDeepTau2018v2p5VSjet", lep.rawDeepTau2018v2p5VSjet)
            if self.isMC:
                fillBranch(prefix + "HiggsMatched", lep.HiggsMatch)
                fillBranch(prefix + "HiggsMatchedIndex", lep.HiggsMatchIndex)
                fillBranch(prefix + "FatJetMatched", lep.FatJetMatch)
                fillBranch(prefix + "FatJetMatchedIndex", lep.FatJetMatchIndex)
                fillBranch(prefix + "MatchedGenPt", lep.MatchedGenPt)


    '''
    def higgsPairingAlgorithm(self, event, jets, fatjets, XbbWP, dotaus=False, taus=[], XtautauWP=0.0):
        # save jets properties

        dummyJet = polarP4()
        dummyJet.HiggsMatch = False
        dummyJet.HiggsMatchIndex = -1
        dummyJet.FatJetMatch = False
        dummyJet.btagDeepFlavB = -1
        dummyJet.btagPNetB = -1
        dummyJet.DeepTauVsJet = -1
        dummyJet.hadronFlavour = -1
        dummyJet.jetId = -1
        dummyJet.puId = -1
        dummyJet.rawFactor = -1
        dummyJet.bRegCorr = -1
        dummyJet.bRegRes = -1
        dummyJet.cRegCorr = -1
        dummyJet.cRegRes = -1
        dummyJet.MatchedGenPt = 0
        dummyJet.mass = 0.

        dummyHiggs = polarP4()
        dummyHiggs.matchH1 = False
        dummyHiggs.matchH2 = False
        dummyHiggs.matchH3 = False
        dummyHiggs.mass = 0.
        dummyHiggs.Mass = 0.
        dummyHiggs.pt = -1
        dummyHiggs.eta = -1
        dummyHiggs.phi = -1
        dummyHiggs.dRjets = 0.

        probejets = [fj for fj in fatjets]
        probetau = []
        if dotaus:
          probetau = sorted([fj for fj in fatjets and fj.Xtautau > XtautauWP], key=lambda x: x.Xtautau, reverse = True)
          if len(probetau)>0:
            probetau = probetau[0]

        jets_4vec = []
        for j in jets:
            overlap = False
            for fj in probejets:
                if fj!=probetau and deltaR(j,fj) < 1.0: overlap = True
            if overlap == False:
                j_tmp = polarP4(j)
                j_tmp.HiggsMatch = j.HiggsMatch
                j_tmp.HiggsMatchIndex = j.HiggsMatchIndex
                j_tmp.FatJetMatch = j.FatJetMatch
                j_tmp.btagDeepFlavB = j.btagDeepFlavB
                j_tmp.btagPNetB = j.btagPNetB
                j_tmp.DeepTauVsJet = -1
                if self.isMC:
                    j_tmp.hadronFlavour = j.hadronFlavour
                j_tmp.jetId = j.jetId
                j_tmp.rawFactor = j.rawFactor
                j_tmp.mass = j.mass
                j_tmp.MatchedGenPt = j.MatchedGenPt
                if self.Run==2:
                    j_tmp.puId = j.puId
                    j_tmp.bRegCorr = j.bRegCorr
                    j_tmp.bRegRes = j.bRegRes
                    j_tmp.cRegCorr = j.cRegCorr
                    j_tmp.cRegRes = j.cRegRes

                jets_4vec.append(j_tmp)

        taus_4vec = []
        for t in taus:
            overlap = False
            if probetau!=[]:
                if deltaR(t,probetau) < 1.0: overlap = True
            if overlap == False:
                t_tmp = polarP4(t)
                t_tmp.HiggsMatch = t.HiggsMatch
                t_tmp.HiggsMatchIndex = t.HiggsMatchIndex
                t_tmp.FatJetMatch = t.FatJetMatch
                t_tmp.charge = t.charge
                t_tmp.btagDeepFlavB = -1
                t_tmp.btagPNetB = -1
                if self.Run==2:
                    t_tmp.DeepTauVsJet = t.rawDeepTau2017v2p1VSjet
                else:
                    t_tmp.DeepTauVsJet = t.rawDeepTau2018v2p5VSjet
                if self.isMC:
                    t_tmp.hadronFlavour = -1
                t_tmp.jetId = -1
                t_tmp.rawFactor = -1
                t_tmp.mass = t.mass
                t_tmp.MatchedGenPt = t.MatchedGenPt
                if self.Run==2:
                    t_tmp.puId = -1
                    t_tmp.bRegCorr = -1
                    t_tmp.bRegRes = -1
                    t_tmp.cRegCorr = -1
                    t_tmp.cRegRes = -1

                taus_4vec.append(t_tmp)


        if len(jets_4vec) > 5:
            jets_4vec = jets_4vec[:6]

        j0 = dummyJet
        j1 = dummyJet
        j2 = dummyJet
        j3 = dummyJet
        j4 = dummyJet
        j5 = dummyJet

        if self.isMC:
            hadGenH_4vec = [polarP4(h) for h in self.hadGenHs]
            genHdaughter_4vec = [polarP4(d) for d in self.genHdaughter]

        # include boosted categories

        # 3 AK8 jets
        if len(probejets) > 2:
            h1 = probejets[0]
            h2 = probejets[1]
            h3 = probejets[2]
            if probetau!=[]:
              h3 = probetau
              event.reco4b2t_TauIsBoosted = 3
              if h1==h3:
                h1 = probejets[1]
                h2 = probejets[2]
              elif h2==h3:
                h2 = probejets[2]

            if self.Run==2:
                m_fit = (h1.particleNet_mass + h2.particleNet_mass + h3.particleNet_mass) / 3.
            else:
                m_fit = (h1.mass*h1.particleNet_massCorr + h2.mass*h2.particleNet_massCorr + h3.mass*h3.particleNet_massCorr) / 3.
            h1.matchH1 = h1.HiggsMatch
            h2.matchH2 = h2.HiggsMatch
            h3.matchH3 = h3.HiggsMatch
            if self.Run==2:
                h1.Mass = h1.particleNet_mass
                h2.Mass = h2.particleNet_mass
                h3.Mass = h3.particleNet_mass
            else:
                h1.Mass = h1.mass*h1.particleNet_massCorr
                h2.Mass = h2.mass*h2.particleNet_massCorr
                h3.Mass = h3.mass*h3.particleNet_massCorr

            if len(jets_4vec) > 0:
                j0 = jets_4vec[0]
            if len(jets_4vec) > 1:
                j1 = jets_4vec[1]
            if len(jets_4vec) > 2:
                j2 = jets_4vec[2]
            if len(jets_4vec) > 3:
                j3 = jets_4vec[3]
            if len(jets_4vec) > 4:
                j4 = jets_4vec[4]
            if len(jets_4vec) > 5:
                j5 = jets_4vec[5]

            if not dotaus:
                event.reco6b_3bh0h = True
                event.reco6b_Idx = 1
            else:
                event.reco4b2t_3bh0h = True
                event.reco4b2t_Idx = 1

        # 2 AK8 jets
        elif len(probejets) == 2:
            h1 = probejets[0]
            h2 = probejets[1]
            if probetau!=[]:
                h2 = probetau
                event.reco4b2t_TauIsBoosted = 2
                if h1==h2:
                    h1 = probejets[1]

            if self.Run==2:
                h1.Mass = h1.particleNet_mass
                h2.Mass = h2.particleNet_mass
            else:
                h1.Mass = h1.mass*h1.particleNet_massCorr
                h2.Mass = h2.mass*h2.particleNet_massCorr
            h1.matchH1 = h1.HiggsMatch
            h2.matchH2 = h2.HiggsMatch

            if (not dotaus) or probetau!=[]: # All 6b, or 4b2tau if boosted Tau already selected
                permutations = list(itertools.permutations(jets_4vec))
                permutations = [el[:6] for el in permutations]
                permutations = list(set(permutations))
                if len(permutations)<2:
                    if not dotaus:
                        event.reco6b_2bh0h = True
                        event.reco6b_Idx = 5
                    else:
                        event.reco4b2t_2bh0h = True
                        event.reco4b2t_Idx = 5
                    return (h1.Mass + h2.Mass)/2.,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5

                min_chi2 = 1000000000000000
                for permutation in permutations:
                    j0_tmp = permutation[0]
                    j1_tmp = permutation[1]

                    h3_tmp = j0_tmp + j1_tmp

                    fitted_mass = (h1.Mass + h2.Mass + h3_tmp.M())/3.
                    chi2 = (h1.Mass - fitted_mass)**2 + (h2.Mass - fitted_mass)**2 + (h3_tmp.M() - fitted_mass)**2

                    if chi2 < min_chi2:
                        m_fit = fitted_mass
                        min_chi2 = chi2

                        if j0_tmp.Pt() > j1_tmp.Pt():
                            j0 = j0_tmp
                            j1 = j1_tmp
                        else:
                            j0 = j1_tmp
                            j1 = j0_tmp
                        if len(jets_4vec) > 2:    
                            j2 = permutation[2] 
                        if len(jets_4vec) > 3:
                            j3 = permutation[3] 
                        if len(jets_4vec) > 4:
                            j4 = permutation[4] 
                        if len(jets_4vec) > 5:
                            j5 = permutation[5] 

                        h3 = h3_tmp
                if not dotaus:
                    event.reco6b_2bh1h = True
                    event.reco6b_Idx = 2
                else:
                    event.reco4b2t_2bh1h = True
                    event.reco4b2t_Idx = 2

            else: # 4b2tau, third resolved Higgs must be made from two taus
                if len(taus_4vec)<2:
                    event.reco4b2t_2bh0h = True
                    event.reco4b2t_Idx = 5
                    return (h1.Mass + h2.Mass)/2.,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5
                permutations = list(itertools.permutations(taus_4vec))
                permutations = [el[:2] for el in permutations]
                permutations = list(set(permutations))

                min_chi2 = 1000000000000000
                for permutation in permutations:
                    t0_tmp = permutation[0]
                    t1_tmp = permutation[1]
                    if t0_tmp.charge * t1_tmp.charge >= 0: continue

                    h3_tmp = t0_tmp + t1_tmp

                    fitted_mass = (h1.Mass + h2.Mass + h3_tmp.M())/3.
                    chi2 = (h1.Mass - fitted_mass)**2 + (h2.Mass - fitted_mass)**2 + (h3_tmp.M() - fitted_mass)**2

                    if chi2 < min_chi2:
                        m_fit = fitted_mass
                        min_chi2 = chi2

                        if t0_tmp.Pt() > t1_tmp.Pt():
                            j0 = t0_tmp
                            j1 = t1_tmp
                        else:
                            j0 = t1_tmp
                            j1 = t0_tmp
                        if len(jets_4vec) > 0:    
                            j2 = permutation[0] 
                        if len(jets_4vec) > 1:
                            j3 = permutation[1] 
                        if len(jets_4vec) > 2:
                            j4 = permutation[2] 
                        if len(jets_4vec) > 3:
                            j5 = permutation[3] 

                        h3 = h3_tmp

                if min_chi2==1000000000000000: # Happens if opposite sign requirement not fulfilled
                    event.reco4b2t_2bh0h = True
                    event.reco4b2t_Idx = 5
                    return (h1.Mass + h2.Mass)/2.,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5
                event.reco4b2t_2bh1h = True
                event.reco4b2t_Idx = 2
                event.reco4b2t_TauIsResolved = 3

            h3.Mass = h3.M()
            h3.pt = h3.Pt()
            h3.eta = h3.Eta()
            h3.phi = h3.Phi()            
            h3.matchH3 = False
            if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                h3.matchH3 = True


        elif len(probejets) == 1:
            h1 = probejets[0]
            if probetau!=[]:
                h1 = probetau
                event.reco4b2t_TauIsBoosted = 1
            if self.Run==2:
                h1.Mass = h1.particleNet_mass
            else:
                h1.Mass = h1.mass*h1.particleNet_massCorr
            h1.matchH1 = h1.HiggsMatch

            if (not dotaus) or probetau!=[]: # All 6b, or 4b2tau if boosted Tau already selected
                permutations = list(itertools.permutations(jets_4vec))
                permutations = [el[:6] for el in permutations]
                permutations = list(set(permutations))
                if len(jets_4vec)<2:
                    if not dotaus:
                        event.reco6b_1bh0h = True
                        event.reco6b_Idx = 8
                    else:
                        event.reco4b2t_1bh0h = True
                        event.reco4b2t_Idx = 8
                    return h1.Mass,h1,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
                elif len(jets_4vec)<4:
                    min_chi2 = 1000000000000000
                    for permutation in permutations:
                        j0_tmp = permutation[0]
                        j1_tmp = permutation[1]

                        h2_tmp = j0_tmp + j1_tmp

                        fitted_mass = (h1.Mass + h2_tmp.M())/2.
                        chi2 = (h1.Mass - fitted_mass)**2 + (h2_tmp.M() - fitted_mass)**2

                        if chi2 < min_chi2:
                            m_fit = fitted_mass
                            min_chi2 = chi2

                            if j0_tmp.Pt() > j1_tmp.Pt():
                                j0 = j0_tmp
                                j1 = j1_tmp
                            else:
                                j0 = j1_tmp
                                j1 = j0_tmp
                            if len(jets_4vec) > 2:    
                                j2 = permutation[2] 
                            if len(jets_4vec) > 3:
                                j3 = permutation[3] 
                            if len(jets_4vec) > 4:
                                j4 = permutation[4] 
                            if len(jets_4vec) > 5:
                                j5 = permutation[5] 

                            h2 = h2_tmp
                    h2.Mass = h2.M()
                    h2.pt = h2.Pt()
                    h2.eta = h2.Eta()
                    h2.phi = h2.Phi()
                    h2.matchH2 = False
                    if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                        h2.matchH2 = True
                    if not dotaus:
                        event.reco6b_1bh1h = True
                        event.reco6b_Idx = 6
                    else:
                        event.reco4b2t_1bh1h = True
                        event.reco4b2t_Idx = 6
                    return m_fit,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5

                min_chi2 = 1000000000000000
                for permutation in permutations:
                    j0_tmp = permutation[0]
                    j1_tmp = permutation[1]
                    j2_tmp = permutation[2]
                    j3_tmp = permutation[3]

                    h2_tmp = j0_tmp + j1_tmp
                    h3_tmp = j2_tmp + j3_tmp

                    fitted_mass = (h1.Mass + h2_tmp.M() + h3_tmp.M())/3.
                    chi2 = (h1.Mass - fitted_mass)**2 + (h2_tmp.M() - fitted_mass)**2 + (h3_tmp.M() - fitted_mass)**2

                    if chi2 < min_chi2:
                        m_fit = fitted_mass
                        min_chi2 = chi2
                        if h2_tmp.Pt() > h3_tmp.Pt():
                            h2 = h2_tmp
                            h3 = h3_tmp
                            if j0_tmp.Pt() > j1_tmp.Pt():
                                j0 = j0_tmp
                                j1 = j1_tmp
                            else:
                                j0 = j1_tmp
                                j1 = j0_tmp

                            if j2_tmp.Pt() > j3_tmp.Pt():
                                j2 = j2_tmp 
                                j3 = j3_tmp
                            else:
                                j2 = j3_tmp
                                j3 = j2_tmp
                        else:
                            h2 = h3_tmp
                            h3 = h2_tmp
                            if j2_tmp.Pt() > j3_tmp.Pt():
                                j0 = j2_tmp 
                                j1 = j3_tmp
                            else:
                                j1 = j3_tmp
                                j0 = j2_tmp
                            if j0_tmp.Pt() > j1_tmp.Pt():
                                j2 = j0_tmp 
                                j3 = j1_tmp
                            else:
                                j2 = j1_tmp
                                j3 = j0_tmp

                        if len(jets_4vec) > 4:
                            j4 = permutation[4]
                        if len(jets_4vec) > 5:
                            j5 = permutation[5]
                if not dotaus:
                    event.reco6b_1bh2h = True
                    event.reco6b_Idx = 3
                else:
                    event.reco4b2t_1bh2h = True
                    event.reco4b2t_Idx = 3

            else: # 4b2tau, one resolved Higgs must be made from two taus, and need to find the other one
                if len(taus_4vec)<2 and len(jets_4vec)<2:
                    event.reco4b2t_1bh0h = True
                    event.reco4b2t_Idx = 8
                    return h1.Mass,h1,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5

                tpermutations = list(itertools.permutations(taus_4vec))
                tpermutations = [el[:2] for el in tpermutations]
                tpermutations = list(set(tpermutations))

                jpermutations = list(itertools.permutations(jets_4vec))
                jpermutations = [el[:6] for el in jpermutations]
                jpermutations = list(set(jpermutations))

                if len(taus_4vec)<2 or all([perm[0].charge*perm[1].charge>=0 for perm in tpermutations]):
                    min_chi2 = 1000000000000000
                    for permutation in jpermutations:
                        j0_tmp = permutation[0]
                        j1_tmp = permutation[1]

                        h2_tmp = j0_tmp + j1_tmp

                        fitted_mass = (h1.Mass + h2_tmp.M())/2.
                        chi2 = (h1.Mass - fitted_mass)**2 + (h2_tmp.M() - fitted_mass)**2

                        if chi2 < min_chi2:
                            m_fit = fitted_mass
                            min_chi2 = chi2

                            if j0_tmp.Pt() > j1_tmp.Pt():
                                j0 = j0_tmp
                                j1 = j1_tmp
                            else:
                                j0 = j1_tmp
                                j1 = j0_tmp
                            if len(jets_4vec) > 2:    
                                j2 = permutation[2] 
                            if len(jets_4vec) > 3:
                                j3 = permutation[3] 
                            if len(jets_4vec) > 4:
                                j4 = permutation[4] 
                            if len(jets_4vec) > 5:
                                j5 = permutation[5] 

                            h2 = h2_tmp
                    h2.Mass = h2.M()
                    h2.pt = h2.Pt()
                    h2.eta = h2.Eta()
                    h2.phi = h2.Phi()
                    h2.matchH2 = False
                    if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                        h2.matchH2 = True
                    event.reco4b2t_1bh1h = True
                    event.reco4b2t_Idx = 6
                    return m_fit,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5

                if len(jets_4vec)<2:
                    min_chi2 = 1000000000000000
                    for permutation in tpermutations:
                        t0_tmp = permutation[0]
                        t1_tmp = permutation[1]
                        if t0_tmp.charge * t1_tmp.charge >= 0: continue

                        h2_tmp = t0_tmp + t1_tmp

                        fitted_mass = (h1.Mass + h2_tmp.M())/3.
                        chi2 = (h1.Mass - fitted_mass)**2 + (h2_tmp.M() - fitted_mass)**2

                        if chi2 < min_chi2:
                            m_fit = fitted_mass
                            min_chi2 = chi2

                            if t0_tmp.Pt() > t1_tmp.Pt():
                                j0 = t0_tmp
                                j1 = t1_tmp
                            else:
                                j0 = t1_tmp
                                j1 = t0_tmp
                            if len(jets_4vec) > 0:    
                                j2 = permutation[0] 
                            if len(jets_4vec) > 1:
                                j3 = permutation[1] 
                            if len(jets_4vec) > 2:
                                j4 = permutation[2] 
                            if len(jets_4vec) > 3:
                                j5 = permutation[3] 

                        h2 = h2_tmp

                    if min_chi2==1000000000000000: # Happens if opposite sign requirement not fulfilled
                        event.reco4b2t_1bh0h = True
                        event.reco4b2t_Idx = 8
                        return h1.Mass,h1,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
                    h2.Mass = h2.M()
                    h2.pt = h2.Pt()
                    h2.eta = h2.Eta()
                    h2.phi = h2.Phi()
                    h2.matchH2 = False
                    if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                        h2.matchH2 = True
                    event.reco4b2t_1bh1h = True
                    event.reco4b2t_Idx = 6
                    event.reco4b2t_TauIsResolved = 2
                    return m_fit,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5

                fullpermutations = list(itertools.product(jpermutations, tpermutations))
                min_chi2 = 1000000000000000
                for permutation in fullpermutations:
                    t0_tmp = permutation[1][0]
                    t1_tmp = permutation[1][1]
                    if t0_tmp.charge * t1_tmp.charge >= 0: continue
                    j0_tmp = permutation[0][0]
                    j1_tmp = permutation[0][1]

                    h2_tmp = j0_tmp + j1_tmp
                    h3_tmp = t0_tmp + t1_tmp

                    fitted_mass = (h1.Mass + h2_tmp.M() + h3_tmp.M())/3.
                    chi2 = (h1.Mass - fitted_mass)**2 + (h2_tmp.M() - fitted_mass)**2 + (h3_tmp.M() - fitted_mass)**2

                    if chi2 < min_chi2:
                        m_fit = fitted_mass
                        min_chi2 = chi2
                        h2 = h2_tmp
                        h3 = h3_tmp # Have the Tau-Higgs be always "as last as possible"
                        if j0_tmp.Pt() > j1_tmp.Pt():
                            j0 = j0_tmp
                            j1 = j1_tmp
                        else:
                            j0 = j1_tmp
                            j1 = j0_tmp

                        if t0_tmp.Pt() > t1_tmp.Pt():
                            j2 = t0_tmp 
                            j3 = t1_tmp
                        else:
                            j2 = t1_tmp
                            j3 = t0_tmp

                        if len(jets_4vec) > 4:
                            j4 = permutation[0][2]
                        if len(jets_4vec) > 5:
                            j5 = permutation[0][3]


                if min_chi2==1000000000000000:
                    print("This shouldn't happen! We checked for valid Tau pairs before.")
                    return h1.Mass,h1,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
                event.reco4b2t_1bh2h = True
                event.reco4b2t_Idx = 3
                event.reco4b2t_TauIsResolved = 3
            h2.Mass = h2.M()
            h2.pt = h2.Pt()
            h2.eta = h2.Eta()
            h2.phi = h2.Phi()
            h2.matchH2 = False
            if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                h2.matchH2 = True

            h3.Mass = h3.M()
            h3.pt = h3.Pt()
            h3.eta = h3.Eta()
            h3.phi = h3.Phi()
            h3.matchH3 = False
            if j2.HiggsMatch == True and j3.HiggsMatch == True and j2.HiggsMatchIndex == j3.HiggsMatchIndex:
                h3.matchH3 = True

        else: 

            if len(jets_4vec) < 2 and len(taus_4vec) < 2:
                if not dotaus:
                    event.reco6b_0bh0h = True
                    event.reco6b_Idx = 0
                else:
                    event.reco4b2t_0bh0h = True
                    event.reco4b2t_Idx = 0
                return 0,dummyHiggs,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
            else:

                # Technique 3: mass fit
                jpermutations = list(itertools.permutations(jets_4vec))
                jpermutations = [el[:6] for el in jpermutations]
                jpermutations = list(set(jpermutations))
                tpermutations = list(itertools.permutations(taus_4vec))
                tpermutations = [el[:2] for el in tpermutations]
                tpermutations = list(set(tpermutations))
                fullpermutations = list(itertools.product(jpermutations, tpermutations))

                min_chi2 = 1000000000000000
                h1 = dummyHiggs
                h2 = dummyHiggs
                h3 = dummyHiggs
                for permutation in fullpermutations:
                    h_tmp = []
                    h_tmpgood = []
                    if len(permutation[0])>1:
                        j0_tmp = permutation[0][0]
                        j1_tmp = permutation[0][1]
                        h_tmp.append(j0_tmp + j1_tmp)
                        h_tmpgood.append(j0_tmp + j1_tmp)
                    else:
                        j0_tmp = dummyJet
                        j1_tmp = dummyJet
                        h_tmp.append(dummyHiggs)

                    if len(permutation[0])>3:
                        j2_tmp = permutation[0][2]
                        j3_tmp = permutation[0][3]
                        h_tmp.append(j2_tmp + j3_tmp)
                        h_tmpgood.append(j2_tmp + j3_tmp)
                    else:
                        j2_tmp = dummyJet
                        j3_tmp = dummyJet
                        h_tmp.append(dummyHiggs)

                    if not dotaus: # 6b
                        if len(permutation[0])>5:
                            j4_tmp = permutation[0][4]
                            j5_tmp = permutation[0][5]
                            h_tmp.append(j4_tmp + j5_tmp)
                            h_tmpgood.append(j4_tmp + j5_tmp)
                        else:
                            j4_tmp = dummyJet
                            j5_tmp = dummyJet
                            h_tmp.append(dummyHiggs)
                    else: # 4b2tau
                        if len(permutation[1])>1:
                            j4_tmp = permutation[1][0]
                            j5_tmp = permutation[1][1]
                            if j4_tmp.charge * j5_tmp.charge >= 0:
                                j4_tmp = dummyJet
                                j5_tmp = dummyJet
                                h_tmp.append(dummyHiggs)
                            else:
                                h_tmp.append(j4_tmp + j5_tmp)
                                h_tmpgood.append(j4_tmp + j5_tmp)
                        else:
                            j4_tmp = dummyJet
                            j5_tmp = dummyJet
                            h_tmp.append(dummyHiggs)

                    if len(h_tmpgood)==0: continue # Can still happen if we have only Taus and charge requirement is failed

                    fitted_mass = 0.0
                    for h in h_tmpgood:
                        fitted_mass += h.M()
                    fitted_mass = fitted_mass/len(h_tmpgood)
                    chi2 = 0.0
                    for h in h_tmpgood:
                        chi2 += (h.M() - fitted_mass)**2

                    if chi2 < min_chi2:
                        m_fit = fitted_mass
                        min_chi2 = chi2
                        if h_tmp[0].Pt() > h_tmp[1].Pt():
                            if h_tmp[0].Pt() > h_tmp[2].Pt():# or dotaus:
                                h1 = h_tmp[0]
                                if j0_tmp.Pt() > j1_tmp.Pt():
                                    j0 = j0_tmp
                                    j1 = j1_tmp
                                else:
                                    j0 = j1_tmp
                                    j1 = j0_tmp

                                if h_tmp[1].Pt() > h_tmp[2].Pt():
                                    h2 = h_tmp[1]
                                    if j2_tmp.Pt() > j3_tmp.Pt():
                                        j2 = j2_tmp 
                                        j3 = j3_tmp
                                    else:
                                        j2 = j3_tmp
                                        j3 = j2_tmp

                                    h3 = h_tmp[2]
                                    if j4_tmp.Pt() > j5_tmp.Pt():
                                        j4 = j4_tmp
                                        j5 = j5_tmp
                                    else:
                                        j4 = j5_tmp
                                        j5 = j4_tmp
                                else:
                                    h2 = h_tmp[2]
                                    if j4_tmp.Pt() > j5_tmp.Pt():
                                        j2 = j4_tmp
                                        j3 = j5_tmp
                                    else:
                                        j2 = j5_tmp
                                        j3 = j4_tmp

                                    h3 = h_tmp[1]
                                    if j2_tmp.Pt() > j3_tmp.Pt():
                                        j4 = j2_tmp
                                        j5 = j3_tmp
                                    else:
                                        j4 = j3_tmp
                                        j5 = j2_tmp
                            else:
                                h1 = h_tmp[2]
                                if j4_tmp.Pt() > j5_tmp.Pt():
                                    j0 = j4_tmp
                                    j1 = j5_tmp
                                else:
                                    j0 = j5_tmp
                                    j1 = j4_tmp

                                h2 = h_tmp[0]
                                if j0_tmp.Pt() > j1_tmp.Pt():
                                    j2 = j0_tmp
                                    j3 = j1_tmp
                                else:
                                    j2 = j1_tmp
                                    j3 = j0_tmp
                                h3 = h_tmp[1]
                                if j2_tmp.Pt() > j3_tmp.Pt():
                                    j4 = j2_tmp
                                    j5 = j3_tmp
                                else:
                                    j4 = j3_tmp
                                    j5 = j2_tmp
                        else:
                            if h_tmp[0].Pt() > h_tmp[2].Pt():
                                h1 = h_tmp[1]
                                if j2_tmp.Pt() > j3_tmp.Pt():
                                    j0 = j2_tmp
                                    j1 = j3_tmp
                                else:
                                    j0 = j3_tmp
                                    j1 = j2_tmp

                                h2 = h_tmp[0]
                                if j0_tmp.Pt() > j1_tmp.Pt():
                                    j2 = j0_tmp
                                    j3 = j1_tmp
                                else:
                                    j2 = j1_tmp
                                    j3 = j0_tmp

                                h3 = h_tmp[2]
                                if j4_tmp.Pt() > j5_tmp.Pt():
                                    j4 = j4_tmp
                                    j5 = j5_tmp
                                else:
                                    j4 = j5_tmp
                                    j5 = j4_tmp
                            else:
                                h3 = h_tmp[0]
                                if j0_tmp.Pt() > j1_tmp.Pt():
                                    j4 = j0_tmp
                                    j5 = j1_tmp
                                else:
                                    j4 = j1_tmp
                                    j5 = j0_tmp

                                if h_tmp[1].Pt() > h_tmp[2].Pt():
                                    h1 = h_tmp[1]
                                    if j2_tmp.Pt() > j3_tmp.Pt():
                                        j0 = j2_tmp
                                        j1 = j3_tmp
                                    else:
                                        j0 = j3_tmp
                                        j1 = j2_tmp
                                    h2 = h_tmp[2]
                                    if j4_tmp.Pt() > j5_tmp.Pt():
                                        j2 = j4_tmp
                                        j3 = j5_tmp
                                    else:
                                        j2 = j5_tmp
                                        j3 = j4_tmp
                                else:
                                    h1 = h_tmp[2]
                                    if j4_tmp.Pt() > j5_tmp.Pt():
                                        j0 = j4_tmp
                                        j1 = j5_tmp
                                    else:
                                        j0 = j5_tmp
                                        j1 = j4_tmp

                                    h2 = h_tmp[1]
                                    if j2_tmp.Pt() > j3_tmp.Pt():
                                        j2 = j2_tmp
                                        j3 = j3_tmp
                                    else:
                                        j2 = j3_tmp
                                        j3 = j2_tmp

                if h1==dummyHiggs:
                    if not dotaus:
                        event.reco6b_0bh0h = True
                        event.reco6b_Idx = 0
                    else:
                        event.reco4b2t_0bh0h = True
                        event.reco4b2t_Idx = 0
                    return 0,dummyHiggs,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
                else:
                    h1.Mass = h1.M()
                    h1.pt = h1.Pt()
                    h1.eta = h1.Eta()
                    h1.phi = h1.Phi()
                    h1.matchH1 = False
                    if j0.HiggsMatch == True and j1.HiggsMatch == True and j0.HiggsMatchIndex == j1.HiggsMatchIndex:
                        h1.matchH1 = True

                if h2==dummyHiggs:
                    if not dotaus:
                        event.reco6b_0bh1h = True
                        event.reco6b_Idx = 9
                    else:
                        event.reco4b2t_0bh1h = True
                        event.reco4b2t_Idx = 9
                        if j0.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 1
                    return m_fit,h1,dummyHiggs,dummyHiggs,j0,j1,j2,j3,j4,j5
                else:
                    h2.Mass = h2.M()
                    h2.pt = h2.Pt()
                    h2.eta = h2.Eta()
                    h2.phi = h2.Phi()
                    h2.matchH2 = False
                    if j2.HiggsMatch == True and j3.HiggsMatch == True and j2.HiggsMatchIndex == j3.HiggsMatchIndex:
                        h2.matchH2 = True

                if h3==dummyHiggs:
                    if not dotaus:
                        event.reco6b_0bh2h = True
                        event.reco6b_Idx = 7
                    else:
                        event.reco4b2t_0bh2h = True
                        event.reco4b2t_Idx = 7
                        if j0.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 1
                        elif j2.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 2
                    return m_fit,h1,h2,dummyHiggs,j0,j1,j2,j3,j4,j5
                else:
                    h3.Mass = h3.M()
                    h3.pt = h3.Pt()
                    h3.eta = h3.Eta()
                    h3.phi = h3.Phi()
                    h3.matchH3 = False
                    if j4.HiggsMatch == True and j5.HiggsMatch == True and j4.HiggsMatchIndex == j5.HiggsMatchIndex:
                        h3.matchH3 = True

                    if not dotaus:
                        event.reco6b_0bh3h = True
                        event.reco6b_Idx = 4
                    else:
                        event.reco4b2t_0bh3h = True
                        event.reco4b2t_Idx = 4
                        if j0.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 1
                        elif j2.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 2
                        elif j4.DeepTauVsJet != -1: event.reco4b2t_TauIsResolved = 3

        return m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5
    '''

    def fillTriggerFilters(self, event):

        triggerFilters = {'hlt4PixelOnlyPFCentralJetTightIDPt20' :0 ,
                          'hlt3PixelOnlyPFCentralJetTightIDPt30' :1 ,
                          'hltPFJetFilterTwoC30' :2 ,
                          'hlt4PFCentralJetTightIDPt30' : 3,
                          'hlt4PFCentralJetTightIDPt35' : 4,
                          'hltQuadCentralJet30' : 5,
                          'hlt2PixelOnlyPFCentralJetTightIDPt40' : 6,
                          'hltL1sTripleJet1008572VBFIorHTTIorDoubleJetCIorSingleJet' : 7,
                          'hlt3PFCentralJetTightIDPt40' : 8,
                          'hlt3PFCentralJetTightIDPt45' : 9,
                          'hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet' : 10,
                          'hltBTagCaloDeepCSVp17Double' : 11,
                          'hltPFCentralJetLooseIDQuad30' : 12,
                          'hlt1PFCentralJetLooseID75' : 13,
                          'hlt2PFCentralJetLooseID60' : 14,
                          'hlt3PFCentralJetLooseID45' : 15,
                          'hlt4PFCentralJetLooseID40' : 16,
                          'hltBTagPFDeepCSV4p5Triple' : 17,
               
                }

        triggerFiltersHT = {'hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet' : 0, 
                            'hltL1sQuadJetC50IorQuadJetC60IorHTT280IorHTT300IorHTT320IorTripleJet846848VBFIorTripleJet887256VBFIorTripleJet927664VBF' : 1,
                            'hltL1sQuadJetC60IorHTT380IorHTT280QuadJetIorHTT300QuadJet' : 2, 
                            'hltCaloQuadJet30HT300' : 3, 
                            'hltPFCentralJetsLooseIDQuad30HT300' : 4, 
                            }
        
        #hlt = Object(event, "HLT")
        trigobj_ref = Collection(event,"TrigObj")
        trigobj = [obj for obj in trigobj_ref if obj.id == 1] # get jet objects
        trigobjHT = [obj for obj in trigobj_ref if obj.id == 3] # get jet objects

        for key in triggerFilters:            
            self.out.branch("n%s"%key, "I")
        for key in triggerFiltersHT:            
            self.out.branch("n%s"%key, "I")
        
        for key in triggerFilters:
            name = 'n%s'%(key)
            numberOfObjects = len([el for el in trigobj if el.filterBits & (1 << triggerFilters[key])])
            self.out.fillBranch(name,numberOfObjects)

        for key in triggerFiltersHT:
            name = 'n%s'%(key)
            numberOfObjects = len([el for el in trigobjHT if el.filterBits & (1 << triggerFiltersHT[key])])
            self.out.fillBranch(name,numberOfObjects)

    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        # fill histograms
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)

        # select leptons and correct jets
        self.selectLeptons(event)
        self.correctJetsAndMET(event)          
        
        # basic jet selection 
        #probe_jets = [fj for fj in event.fatjets if fj.pt > 300 and fj.Xbb > 0.8]
        probe_jets = [fj for fj in event.fatjets if fj.pt > 265 and abs(fj.eta) < 2.5 and fj.jetId >= 2] # 215 GeV cut good for PNet scores # 265 GeV for cut to use n3b1 to filter AK4 contamination
        
        #probe_jets.sort(key=lambda x: x.pt, reverse=True)
        probe_jets.sort(key=lambda x: x.Xbb, reverse=True)



        if self._opts['option'] == "10":
            probe_jets = [fj for fj in event.fatjets if (fj.pt > 200 and fj.t32<0.54)]
            if len(probe_jets) < 1:
                return False
        #elif self._opts['option'] == "21":
        #    probe_jets = [fj for fj in event.vbffatjets if fj.pt > 200]
        #    if len(probe_jets) < 1:
        #        return False
        #else:
        #    if len(probe_jets) < 2:
        #        return False

        # evaluate regression
        self.evalMassRegression(event, probe_jets)

        # apply selection
        passSel = False
        if self._opts['option'] == "5":
            if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250 and ((probe_jets[0].msoftdropJMS>50 and probe_jets[1].msoftdropJMS>50) or (probe_jets[0].regressed_massJMS>50 and probe_jets[1].regressed_massJMS>50)) and probe_jets[0].Xbb>0.8): passSel = True
        elif self._opts['option'] == "10":
            if len(probe_jets) >= 2:
                if(probe_jets[0].pt > 250 and probe_jets[1].pt > 250): passSel = True
            if(probe_jets[0].pt > 250 and len(event.looseLeptons)>0): passSel = True
        elif self._opts['option'] == "21":
            if(probe_jets[0].pt > 250 and (probe_jets[0].msoftdropJMS >30 or probe_jets[0].regressed_massJMS > 30)): passSel=True
        elif self._opts['option'] == "8":
            if(probe_jets[0].pt > 300 and abs(probe_jets[0].eta)<2.5 and probe_jets[1].pt > 300 and abs(probe_jets[1].eta)<2.5):
                if ((probe_jets[0].msoftdropJMS>30 and probe_jets[1].msoftdropJMS>30) or (probe_jets[0].regressed_massJMS>30 and probe_jets[1].regressed_massJMS>30) or (probe_jets[0].msoftdrop>30 and probe_jets[1].msoftdrop>30)):
                    passSel=True
        elif self._opts['option'] == "0":
            if (self.nFatJets == 0 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "1":
            if (self.nFatJets == 1 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "2":
            if (self.nFatJets == 2 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "3":
            if (self.nFatJets == 3 and self.nSmallJets > 5 ): passSel = True
        elif self._opts['option'] == "4":
            #if (self.nSmallJets > 5 and self.nBTaggedJets > 2): passSel = True
            if (self.nSmallJets > -1): passSel = True

        if not passSel: return False

        # load gen history
        hadGenHs = self.loadGenHistory(event, probe_jets)
        self.hadGenHs = hadGenHs

        for j in event.ak4jets+event.looseTaus+event.looseLeptons:
            j.HiggsMatch = False
            j.FatJetMatch = False
            j.HiggsMatchIndex = -1
            j.FatJetMatchIndex = -1
            j.MatchedGenPt = 0.

        for fj in probe_jets:
            fj.HiggsMatch = False
            fj.HiggsMatchIndex = -1
            fj.MatchedGenPt = 0.

        if self.isMC:
            daughters = []
            matched = 0
            for index_h, higgs_gen in enumerate(hadGenHs):
                matchedthishiggs = []
                matchedthisdau = []
                for idx in higgs_gen.dauIdx:
                    dau = event.genparts[idx]
                    daughters.append(dau)
                    for j in event.ak4jets+event.looseTaus+event.looseLeptons:
                        if j in event.ak4jets and abs(dau.pdgId)!=5: continue
                        if j in event.looseTaus+event.looseLeptons and abs(dau.pdgId)==5: continue
                        if deltaR(j,dau) < 0.5:
                            j.HiggsMatch = True
                            j.HiggsMatchIndex = index_h+1
                            j.MatchedGenPt = dau.pt
                            matched += 1
                            matchedthishiggs.append(j)
                            matchedthisdau.append(dau)
                # Get the invariant mass of gen-matched objects, fot both Hbb and Htautau
                if len(matchedthishiggs)==2 and matchedthisdau[0]!=matchedthisdau[1] and deltaR(matchedthishiggs[0],matchedthishiggs[1])>0.5:
                    combgen = polarP4(matchedthishiggs[0])+polarP4(matchedthishiggs[1])
                    dau1pdgid = daughters[-2].pdgId
                    dau2pdgid = daughters[-1].pdgId
                    if dau1pdgid*dau2pdgid == -5*5 and matchedthishiggs[0] in event.ak4jets and matchedthishiggs[1] in event.ak4jets:
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPt", combgen.Pt())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoEta", combgen.Eta())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPhi", combgen.Phi())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoMass", combgen.M())
                    elif dau1pdgid*dau2pdgid != -5*5 and (matchedthishiggs[0] not in event.ak4jets) and (matchedthishiggs[1] not in event.ak4jets): # FastMTT for HTauTau
                        dm1 = matchedthishiggs[0].decayMode if matchedthishiggs[0] in event.looseTaus else -1
                        tau1 = ROOT.MeasuredTauLepton(matchedthishiggs[0].kind, matchedthishiggs[0].pt, matchedthishiggs[0].eta, matchedthishiggs[0].phi, matchedthishiggs[0].mass, dm1)
                        dm2 = matchedthishiggs[1].decayMode if matchedthishiggs[1] in event.looseTaus else -1
                        tau2 = ROOT.MeasuredTauLepton(matchedthishiggs[1].kind, matchedthishiggs[1].pt, matchedthishiggs[1].eta, matchedthishiggs[1].phi, matchedthishiggs[1].mass, dm2)
                        VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
                        bothtaus = VectorOfTaus()
                        bothtaus.push_back(tau1)
                        bothtaus.push_back(tau2)
                        MET_x = event.PuppiMET_pt*math.cos(event.PuppiMET_phi)
                        MET_y = event.PuppiMET_pt*math.sin(event.PuppiMET_phi)
                        covMET = ROOT.TMatrixD(2,2)
                        covMET[0][0] = event.MET_covXX
                        covMET[1][0] = event.MET_covXY
                        covMET[0][1] = event.MET_covXY
                        covMET[1][1] = event.MET_covYY
                        FMTT = ROOT.FastMTT()
                        FMTT.run(bothtaus, MET_x, MET_y, covMET)
                        FMTToutput = FMTT.getBestP4()
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPt", combgen.Pt())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoEta", combgen.Eta())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPhi", combgen.Phi())
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoMass", FMTToutput.M())
                        if combgen.M()<1:
                          for idxX in higgs_gen.dauIdx:
                             dauX = event.genparts[idxX]
                    else:
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPt", -1)
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoEta", -1)
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoPhi", -1)
                        self.out.fillBranch(f"genHiggs{index_h+1}RecoMass", -1)
                else:
                    self.out.fillBranch(f"genHiggs{index_h+1}RecoPt", -1)
                    self.out.fillBranch(f"genHiggs{index_h+1}RecoEta", -1)
                    self.out.fillBranch(f"genHiggs{index_h+1}RecoPhi", -1)
                    self.out.fillBranch(f"genHiggs{index_h+1}RecoMass", -1)
                for fj in probe_jets:
                    if deltaR(higgs_gen, fj) < 1.0:
                        fj.HiggsMatch = True
                        fj.HiggsMatchIndex = index_h+1
                        fj.MatchedGenPt = higgs_gen.pt

            self.out.fillBranch("nHiggsMatchedJets", matched)

        #print("Matched outside fillJetInfo", matched)
        if self.isMC:
            self.genHdaughter = daughters
        index_fj = 0
        for fj in probe_jets:
            index_fj += 1
            for j in event.ak4jets+event.looseTaus+event.looseLeptons:
                if deltaR(fj,j) < 1.0:
                    j.FatJetMatch = True
                    j.FatJetMatchIndex = index_fj


        # fill output branches
        self.fillBaseEventInfo(event, probe_jets, hadGenHs)

        # Low purity WP (https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2021/005)
        XbbWP = {"2016APV": 0.9088,
                 "2016"   : 0.9137,
                 "2017"   : 0.9105,
                 "2018"   : 0.9172,
                 "2022"   : 0.5, # 2022: Test with low cut
                 "2022EE" : 0.5}[self.year]
                 #"2022"   : 0.91255, # 2022: Temporary average value from Run2
                 #"2022EE" : 0.91255}[self.year]
        XtautauWP = 0.9
        self.out.fillBranch("nprobejets", len([fj for fj in probe_jets if fj.pt > 200 and fj.Xbb > XbbWP]))
        self.out.fillBranch("nprobetaus", len([fj for fj in probe_jets if fj.pt > 200 and fj.Xtauany > XtautauWP]))
        #print(len(probe_jets))
        #if len(probe_jets) > 0:
        self.fillFatJetInfo(event, probe_jets)
          
        # for ak4 jets we only fill the b-tagged medium jets
        #self.fillJetInfo(event, event.bmjets)
        #self.fillJetInfo(event, event.bljets)
        try:
            self.fillJetInfo(event, event.ak4jets, probe_jets, XbbWP, event.looseTaus, XtautauWP, event.looseLeptons)
        except IndexError:
            return False

        self.fillLeptonInfo(event, event.looseLeptons)
        self.fillTauInfo(event, event.looseTaus)
 
        # for all jme systs
        if self._allJME and self.isMC:
            self.fillFatJetInfoJME(event, probe_jets)

        #self.fillTriggerFilters(event) 
        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
def hhh6bProducerPNetAK4FromConfig():
    import sys
    #sys.path.remove('/usr/lib64/python2.7/site-packages')
    import yaml
    with open('hhh6b_cfg.json') as f:
        cfg = yaml.safe_load(f)
        year = cfg['year']
        return hhh6bProducerPNetAK4(**cfg)

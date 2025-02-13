import os
import itertools
from xml.sax.saxutils import prepare_input_source
import ROOT
import random
ROOT.PyConfig.IgnoreCommandLineOptions = True
import numpy as np
from collections import Counter
from operator import itemgetter

import correctionlib

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
        

# Flavour tagging PNet calibrations from Huilin and ttHcc analysis

class FlavTagSFProducer():
    def __init__(self, year, ftag_tag_dict):
        era = {'2016APV': '2016preVFP_UL', '2016': '2016postVFP_UL', '2017': '2017_UL', '2018': '2018_UL','2022': '2018_UL','2022EE':'2018_UL','2023':'2018_UL'}[year]
        correction_file = os.path.expandvars(
            f'$CMSSW_BASE/src/PhysicsTools/NanoAODTools/data/flavTagSF/flavTaggingSF_{era}.json.gz')
        self.corr = correctionlib.CorrectionSet.from_file(correction_file)['particleNetAK4_shape']
        self.ftag_tag_dict = ftag_tag_dict
    def get_sf(self, j, syst='central'):
        return self.corr.evaluate(syst, j.hadronFlavour, self.ftag_tag_dict[j.tag], abs(j.eta), j.pt)


class hhh6bProducerPNetAK4(Module):
    
    def __init__(self, year, **kwargs):
        print(year)
        self.year = year
        self.Run = 2 if year in ["2016APV", "2016", "2017", "2018"] else 3

        if self.Run == 2: 
            self._jet_algo = 'AK4PFchs'
        else:
            self._jet_algo = 'AK4PFPuppi'


        self.jetType = 'ak8'
        self._jetConeSize = 0.8
        self._fj_name = 'FatJet'
        self._sj_name = 'SubJet'
        self._fj_gen_name = 'GenJetAK8'
        self._sj_gen_name = 'SubGenJetAK8'
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None, 'met_unclustered': None, 'smearMET': False, 'applyHEMUnc': False, 'jmr': None}
        
        if self.Run == 3: # Before reading the config
            self._jmeSysts['jec'] = True
            self._jmeSysts['jes'] = 'nominal'
            self._jmeSysts['jer'] = 'nominal'
            self._jmeSysts['jmr'] = 'nominal'
            self._jmeSysts['smearMET'] = True
            self._jmeSysts['applyHEMUnc'] = True
        
        self._opts = {'run_mass_regression': False, 'mass_regression_versions': ['ak8V01a', 'ak8V01b', 'ak8V01c'],
                      'WRITE_CACHE_FILE': False, 'option': "1", 'allJME': False}

        if self.Run == 3:
            self._opts['allJME'] = True # Before reading the config
        
        self._opts['allJME'] = True

        for k in kwargs:
            if k in self._jmeSysts:
                self._jmeSysts[k] = kwargs[k]
            else:
                self._opts[k] = kwargs[k]
        self._needsJMECorr = any([self._jmeSysts['jec'],
                                  self._jmeSysts['jes'],
                                  self._jmeSysts['jer'],
                                  self._jmeSysts['jmr'],
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
            self.jetmetCorr = JetMETCorrector(year=self.year, jetType=self._jet_algo, **self._jmeSysts)
            self.fatjetCorr = JetMETCorrector(year=self.year, jetType="AK8PFPuppi", **self._jmeSysts)
            self.subjetCorr = JetMETCorrector(year=self.year, jetType="AK4PFPuppi", **self._jmeSysts)
            self._allJME = False

        if self._allJME:
            # self.applyHEMUnc = False
            self.applyHEMUnc = self._jmeSysts['applyHEMUnc']
            year_pf = "_%s"%self.year
            self.jetmetCorrectors = {
                'nominal': JetMETCorrector(year=self.year, jetType=self._jet_algo, jer='nominal', applyHEMUnc=self.applyHEMUnc),
                'JERUp': JetMETCorrector(year=self.year, jetType=self._jet_algo, jer='up'),
                'JERDown': JetMETCorrector(year=self.year, jetType=self._jet_algo, jer='down'),
                'JMRUp': JetMETCorrector(year=self.year, jetType=self._jet_algo, jmr='up'),
                'JMRDown': JetMETCorrector(year=self.year, jetType=self._jet_algo, jmr='down'),
                'JESUp': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes='up'),
                'JESDown': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes='down'),

                
                #'JESUp_Abs': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_Abs': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_Abs'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_Abs'+year_pf:JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                #'JESUp_BBEC1': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_BBEC1': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                
                #'JESUp_EC2': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_EC2': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_EC2'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_EC2'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_FlavQCD': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_FlavQCD': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_HF': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_HF': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_HF'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_HF'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_RelBal': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_RelBal': JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType=self._jet_algo, jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
            }
            # hemunc for 2018 only
            self.fatjetCorrectors = {
                'nominal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=self.applyHEMUnc),
                #'HEMDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='nominal', applyHEMUnc=True),
                'JERUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='up'),
                'JERDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jer='down'),
                'JMRUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jmr='up'),
                'JMRDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jmr='down'),
                'JESUp': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='up'),
                'JESDown': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes='down'),
                

                #'JESUp_Abs': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_Abs': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_Abs'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_Abs'+year_pf:JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='Absolute'+year_pf,jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_BBEC1': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_BBEC1': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_BBEC1'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='BBEC1'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_EC2': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_EC2': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_EC2'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='EC2'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_FlavQCD': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_FlavQCD': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='FlavorQCD', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_HF': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_HF': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESUp_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_HF'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='HF'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_RelBal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_RelBal': JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeBal', jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),

                #'JESUp_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='up', jes_uncertainty_file_prefix="RegroupedV2_"),
                #'JESDown_RelSample'+year_pf: JetMETCorrector(year=self.year, jetType="AK8PFPuppi", jes_source='RelativeSample'+year_pf, jes='down', jes_uncertainty_file_prefix="RegroupedV2_"),
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


        #spanet inference 
        prefix_tools = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/NanoAODTools/data')
        #sess_options = onnxruntime.SessionOptions()
        #sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL

        #classification_path =  "spanet/spanet_pnet_all_vars_v0_no_log.onnx"
        #if 'no_log' in classification_path:
        #    self.SpanetONNXNeedLogTransform = True
        #    print("Using %s"%classification_path)
        #    print("Need to transform inputs with log(x + 1) before inference")

        #categorisation_path = "spanet/spanet_categorisation_v6_no_log.onnx"

        #if 'no_log' in classification_path and 'no_log' not in categorisation_path:
        #    print("Error: categorisation and classificaton inference requires both the same inputs")
        #    exit()


        #self.session_classification = onnxruntime.InferenceSession(prefix_tools + '/' + classification_path,sess_options) # HHH vs HH vs QCD vs ...
        #self.session_categorisation = onnxruntime.InferenceSession(prefix_tools + '/' + categorisation_path,sess_options) # 3bh0h, 2bh1h, ...
        
        
        #self.output_nodes_classification = [node.name for node in self.session_classification.get_outputs()]
        #self.output_nodes_categorisation = [node.name for node in self.session_categorisation.get_outputs()]

        # FTAG calibrations from Huilin 
        self.ftag_tag_dict = {
            0: 'L0',
            40: 'C0', 41: 'C1', 42: 'C2', 43: 'C3', 44: 'C4',
            50: 'B0', 51: 'B1', 52: 'B2', 53: 'B3', 54: 'B4',
        }
        self.ftag_mapping = {
            0: 0,
            40: 1, 41: 2, 42: 3, 43: 4, 44: 5,
            50: 6, 51: 7, 52: 8, 53: 9, 54: 10,
        }

        # FIXME: systematics list
        self.ftag_systematics = [
            'Stat',
            'LHEScaleWeight_muF_ttbar',
            'LHEScaleWeight_muF_wjets',
            'LHEScaleWeight_muF_zjets',
            'LHEScaleWeight_muR_ttbar',
            'LHEScaleWeight_muR_wjets',
            'LHEScaleWeight_muR_zjets',
            'PSWeightISR',
            'PSWeightFSR',
            'XSec_WJets_c',
            'XSec_WJets_b',
            'XSec_ZJets_c',
            'XSec_ZJets_b',
            # 'JER',
            # 'JES',
            'PUWeight',
            'PUJetID'
        ]

        self.ftagSF = FlavTagSFProducer(self.year, self.ftag_tag_dict)

        # AK8 PNet calibrations in a pseudo-continuous way, taken from HIG-23-011
        self.fatjet_flavtag_sf = [0.98,1.3,0.87] # SF_tight, SF_medium, SF_fail (unc 0.13,0.19,0.22) to be varied correlated
        self.fatjet_flavtag_sf_up = [0.98+0.13,1.3+0.19,0.87-0.22] # UP UP DOWN
        self.fatjet_flavtag_sf_down = [0.98-0.13,1.3-0.19,0.87+0.22] # DOWN DOWN UP

        self.fatjet_flavtag_wps = {'2016APV' : [0.9883,0.9737], 
                                   '2016'    : [0.9883, 0.9735], 
                                   '2017'    : [0.9870, 0.9714],
                                   '2018'    : [0.9880, 0.9734],
        } # tight, medium, and below is fail 

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
        self.out.branch("npvs", "F")
        self.out.branch("npvsGood", "F")

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
            self.out.branch(prefix + "MatchedGenPt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "MassSD", "F")
            self.out.branch(prefix + "MassSD_noJMS", "F")
            self.out.branch(prefix + "MassSD_UnCorrected", "F")
            self.out.branch(prefix + "PNetXbb", "F")
            self.out.branch(prefix + "PNetXbbTagCat", "I")
            self.out.branch(prefix + "PNetXjj", "F")
            if self.Run!=2:
                self.out.branch(prefix + "PNetXtautau", "F")
                self.out.branch(prefix + "PNetXtaumu", "F")
                self.out.branch(prefix + "PNetXtaue", "F")
                self.out.branch(prefix + "PNetXtauany", "F")
            self.out.branch(prefix + "PNetQCD", "F")
            self.out.branch(prefix + "Area", "F")
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

            # uncertainties
            if self.isMC:
                self.out.branch(prefix + "MassSD_JMS_Down", "F")
                self.out.branch(prefix + "MassSD_JMS_Up", "F")
                self.out.branch(prefix + "MassSD_JMR_Down", "F")
                self.out.branch(prefix + "MassSD_JMR_Up", "F")

                self.out.branch(prefix + "PtOverMHH_JMS_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMS_Up", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Down", "F")
                self.out.branch(prefix + "PtOverMHH_JMR_Up", "F")
                self.out.branch(prefix + "PNetSF", "F")


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
        self.out.branch("h1_t3_match", "O")
        self.out.branch("h1_t3_dRjets", "F")

        self.out.branch("h2_t3_pt", "F")
        self.out.branch("h2_t3_eta", "F")
        self.out.branch("h2_t3_phi", "F")
        self.out.branch("h2_t3_mass", "F")
        self.out.branch("h2_t3_match", "O")
        self.out.branch("h2_t3_dRjets", "F")

        self.out.branch("h3_t3_pt", "F")
        self.out.branch("h3_t3_eta", "F")
        self.out.branch("h3_t3_phi", "F")
        self.out.branch("h3_t3_mass", "F")
        self.out.branch("h3_t3_match", "O")
        self.out.branch("h3_t3_dRjets", "F")

        self.out.branch("h_fit_mass", "F")

        self.out.branch("h1_4b2t_pt", "F")
        self.out.branch("h1_4b2t_eta", "F")
        self.out.branch("h1_4b2t_phi", "F")
        self.out.branch("h1_4b2t_mass", "F")
        self.out.branch("h1_4b2t_match", "O")
        self.out.branch("h1_4b2t_dRjets", "F")

        self.out.branch("h2_4b2t_pt", "F")
        self.out.branch("h2_4b2t_eta", "F")
        self.out.branch("h2_4b2t_phi", "F")
        self.out.branch("h2_4b2t_mass", "F")
        self.out.branch("h2_4b2t_match", "O")
        self.out.branch("h2_4b2t_dRjets", "F")

        self.out.branch("h3_4b2t_pt", "F")
        self.out.branch("h3_4b2t_eta", "F")
        self.out.branch("h3_4b2t_phi", "F")
        self.out.branch("h3_4b2t_mass", "F")
        self.out.branch("h3_4b2t_match", "O")
        self.out.branch("h3_4b2t_dRjets", "F")

        self.out.branch("h_fit_mass_4b2t", "F")

        self.out.branch("reco6b_Idx", "I")
        self.out.branch("reco4b2t_Idx", "I")
        self.out.branch("reco4b2t_TauIsBoosted", "I")
        self.out.branch("reco4b2t_TauIsResolved", "I")

        # SPANET variables
        self.out.branch("probHHH","F")
        self.out.branch("probQCD","F")
        self.out.branch("probTT", "F")
        self.out.branch("probVJets","F")
        self.out.branch("probVV","F")
        self.out.branch("probHHH4b2tau","F")
        self.out.branch("probHH4b","F")
        self.out.branch("probHH2b2tau","F")

        self.out.branch("prob3bh0h", "F")
        self.out.branch("prob2bh1h", "F")
        self.out.branch("prob1bh2h", "F")
        self.out.branch("prob0bh3h", "F")
        self.out.branch("prob2bh0h", "F")
        self.out.branch("prob1bh1h", "F")
        self.out.branch("prob0bh2h", "F")
        self.out.branch("prob1bh0h", "F")
        self.out.branch("prob0bh1h", "F")
        self.out.branch("prob0bh0h", "F")

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
            self.out.branch(prefix + "PNetSF", "F")

            if self.Run == 2:
                self.out.branch(prefix + "PNetC", "F")
                self.out.branch(prefix + "PNetBPlusC", "F")
                self.out.branch(prefix + "PNetBVsC", "F")
                self.out.branch(prefix + "PNetTagCat","I")
            else:
                self.out.branch(prefix + "PNetCvB", "F")
                self.out.branch(prefix + "PNetCvL", "F")

            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "RawFactor", "F")
            self.out.branch(prefix + "MatchedGenPt", "F")
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
                self.out.branch(prefix + "FatJetMatched", "O")
                self.out.branch(prefix + "FatJetMatchedIndex", "I")
            if self.isMC:
                self.out.branch(prefix + "HadronFlavour", "F")
                self.out.branch(prefix + "HiggsMatched", "O")
                self.out.branch(prefix + "HiggsMatchedIndex", "I")

                self.out.branch(prefix + "Charge", "I")
                self.out.branch(prefix + "PdgId", "I")
                self.out.branch(prefix + "DRGenQuark", "F")


        # leptons
        for idx in ([1, 2]):
            prefix = 'lep%i'%idx
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Id", "I")

        for idx in ([1, 2, 3, 4]):
            prefix = 'tau%i'%idx
            self.out.branch(prefix + "Charge", "F")
            self.out.branch(prefix + "Pt", "F")
            self.out.branch(prefix + "Eta", "F")
            self.out.branch(prefix + "Phi", "F")
            self.out.branch(prefix + "Mass", "F")
            self.out.branch(prefix + "Id", "I")
            self.out.branch(prefix + "decayMode", "F")
            self.out.branch(prefix + "MatchedGenPt", "F")

            if self.Run==2: # TODO: Can switch to v2p5 for Run2UL too, if inputs have branches available
                self.out.branch(prefix + "rawDeepTau2017v2p1VSe", "F")
                self.out.branch(prefix + "rawDeepTau2017v2p1VSjet", "F")
                self.out.branch(prefix + "rawDeepTau2017v2p1VSmu", "F")
            else:
                self.out.branch(prefix + "rawDeepTau2018v2p5VSe", "F")
                self.out.branch(prefix + "rawDeepTau2018v2p5VSjet", "F")
                self.out.branch(prefix + "rawDeepTau2018v2p5VSmu", "F")
            if self.isMC:
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

        # Flavour tagging up and down variations
        self.ftagbasewgts = {'flavTagWeight': 1}
        self.fj_ftagbasewgts = {'fatJetFlavTagWeight': 1, 'fatJetFlavTagWeight_UP': 1, 'fatJetFlavTagWeight_DOWN': 1,}

        if self.isMC:
            for syst in self.ftag_systematics:
                self.ftagbasewgts[f'flavTagWeight_{syst}_UP'] = 1
                self.ftagbasewgts[f'flavTagWeight_{syst}_DOWN'] = 1
            for name in self.ftagbasewgts.keys():
                self.out.branch(name, "F")

            for name in self.fj_ftagbasewgts.keys():
                self.out.branch(name, "F")
                

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
                    
    def loadGenHistory(self, event, fatjets, ak4jets):
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
        quarksGen = []
        
        for gp in genparts:
            if gp.statusFlags & (1 << 13) == 0:
                continue
            #print(gp.pdgId)

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
            elif abs(gp.pdgId) == 5 or abs(gp.pdgId) == 4 or abs(gp.pdgId) == 3 or abs(gp.pdgId) == 2 or abs(gp.pdgId) == 1 or abs(gp.pdgId) == 0:
                quarksGen.append(gp)
                         
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
        quarksGen.sort(key = lambda x: x.pdgId, reverse = True)

        #for q in quarksGen:
            #print(q.pdgId)

        for jet in ak4jets:
            if len(quarksGen) > 0:
                jet.genQuark, jet.dr_genQuark, jet.genQuarkIdx  = closest(jet,quarksGen)
                #print(jet.genQuark.pdgId, jet.dr_genQuark, jet.genQuarkIdx)
            else:
                jet.genQuark = 0
                jet.pdgId = 0
                jet.dr_genQuark = -999.
                jet.genQuarkIdx = 0

        #print()

        return hadGenHs+tauGenHs
               
    def selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for lepton counting
        event.cleaningElectrons = []
        event.cleaningMuons = []
        event.looseTaus = [] # store taus
        
        electrons = Collection(event, "Electron")
        for el in electrons:
            el.Id = el.charge * (-11)
            #if el.pt > 35 and abs(el.eta) <= 2.5 and el.miniPFRelIso_all <= 0.2 and el.cutBased:
            if self.Run==2:
                if el.pt > 15 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.045 and abs(el.dz) < 0.2 and el.pfRelIso03_all <= 0.15 and el.lostHits <= 1 and el.convVeto and el.mvaFall17V2Iso_WP90: #and el.cutBased>3: # cutBased ID: (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                    event.looseLeptons.append(el)
            else:
                if el.pt > 10 and abs(el.eta) <= 2.5 and abs(el.dxy) < 0.045 and abs(el.dz) < 0.2 and el.miniPFRelIso_all <= 0.2 and el.lostHits <= 1 and el.convVeto and el.mvaIso_WP90: #and el.cutBased>3: # cutBased ID: (0:fail, 1:veto, 2:loose, 3:medium, 4:tight)
                    event.looseLeptons.append(el)
            if self.Run==2:
                if el.pt > 30 and el.mvaFall17V2Iso_WP90:
                    event.cleaningElectrons.append(el)
            else:
                if el.pt > 30 and el.mvaIso_WP90:
                    event.cleaningElectrons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            mu.Id = mu.charge * (-13)
            if mu.pt > 10 and abs(mu.eta) <= 2.4 and abs(mu.dxy) < 0.045 and abs(mu.dz) < 0.2 and mu.mediumId and mu.pfRelIso03_all <= 0.15: # mu.tightId
                event.looseLeptons.append(mu)
            if mu.pt > 30 and mu.looseId:
                event.cleaningMuons.append(mu)

        taus = Collection(event, "Tau")
        for tau in taus:
            tau.Id = tau.charge * (-15)
            tau.kind = self.kTauToHadDecay
            if tau.decayMode==0: tau.mass = 0.13957
            if self.Run==2:
                if tau.pt > 20 and abs(tau.eta) <= 2.3 and abs(tau.dz) < 0.2 and (tau.decayMode in [0,1,2,10,11]) and tau.idDeepTau2017v2p1VSe >= 2 and tau.idDeepTau2017v2p1VSmu >= 1 and tau.idDeepTau2017v2p1VSjet >= 8:
                    event.looseTaus.append(tau) # VVloose VsE, VLoose vsMu, Loose Vsjet
            else:
                if tau.pt > 20 and abs(tau.eta) <= 2.5 and abs(tau.dz) < 0.2 and (tau.decayMode in [0,1,2,10,11]) and tau.idDeepTau2018v2p5VSe >= 2 and tau.idDeepTau2018v2p5VSmu >= 1 and tau.idDeepTau2018v2p5VSjet >= 4:
                    event.looseTaus.append(tau) # VVloose VsE, VLoose vsMu, Loose Vsjet

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
        if self.Run == 3:
            event.rho = Object(event,'Rho')
        #event.met = METObject(event, "METFixEE2017") if self.year == 2017 else METObject(event, "MET")
        event.met = METObject(event, "MET")
        event._allFatJets = Collection(event, self._fj_name)
        event.subjets = Collection(event, self._sj_name)  # do not sort subjets after updating!!
        
        # JetMET corrections
        if self._needsJMECorr:
            if self.Run == 2:
                rho = event.fixedGridRhoFastjetAll
            else:
                rho = event.rho.fixedGridRhoFastjetAll
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

        if self._jmeSysts['jes']:
            if "up" in self._jmeSysts['jes']:
                event._allFatJets = event._FatJets['JESUp']
                event._allJets = event._AllJets['JESUp']
            elif "down" in self._jmeSysts['jes']:
                event._allFatJets = event._FatJets['JESDown']
                event._allJets = event._AllJets['JESDown']

        if self._jmeSysts['jer']:
            if "up" in self._jmeSysts['jer']:
                event._allFatJets = event._FatJets['JERUp']
                event._allJets = event._AllJets['JERUp']
            elif "down" in self._jmeSysts['jer']:
                event._allFatJets = event._FatJets['JERDown']
                event._allJets = event._AllJets['JERDown']
        
        if self._jmeSysts['jmr']:
            if "up" in self._jmeSysts['jmr']:
                event._allFatJets = event._FatJets['JMRUp']
                event._allJets = event._AllJets['JMRDown']
            elif "down" in self._jmeSysts['jmr']:
                event._allFatJets = event._FatJets['JMRDown']
                event._allJets = event._AllJets['JMRDown']


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

        # Apply calibrations for AK8 PNet score
        if self.Run == 2:
            for fj in event.fatjets:
                if fj.Xbb > self.fatjet_flavtag_wps[self.year][0]:  # Tight working point
                    fj.tag = 2 # Tight working point
                    if self.isMC:
                        fj.ftagSF = self.fatjet_flavtag_sf[0]
                        self.fj_ftagwgts['fatJetFlavTagWeight'] *= fj.ftagSF
                        self.fj_ftagwgts['fatJetFlavTagWeight_UP'] *= self.fatjet_flavtag_sf_up[0]
                        self.fj_ftagwgts['fatJetFlavTagWeight_DOWN'] *= self.fatjet_flavtag_sf_down[0]
                elif fj.Xbb < self.fatjet_flavtag_wps[self.year][0] and fj.Xbb > self.fatjet_flavtag_wps[self.year][1]: # Medium working point
                    fj.tag = 1
                    if self.isMC:
                        fj.ftagSF = self.fatjet_flavtag_sf[1]
                        self.fj_ftagwgts['fatJetFlavTagWeight'] *= fj.ftagSF
                        self.fj_ftagwgts['fatJetFlavTagWeight_UP'] *= self.fatjet_flavtag_sf_up[1]
                        self.fj_ftagwgts['fatJetFlavTagWeight_DOWN'] *= self.fatjet_flavtag_sf_down[1]
                elif fj.Xbb < self.fatjet_flavtag_wps[self.year][1] : # Inefficiency scale factor
                    fj.tag = 0
                    if self.isMC:
                        fj.ftagSF = self.fatjet_flavtag_sf[2]
                        self.fj_ftagwgts['fatJetFlavTagWeight'] *= fj.ftagSF
                        self.fj_ftagwgts['fatJetFlavTagWeight_UP'] *= self.fatjet_flavtag_sf_up[2]
                        self.fj_ftagwgts['fatJetFlavTagWeight_DOWN'] *= self.fatjet_flavtag_sf_down[2]


        #event.ak4jets = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and (j.jetId & 2)]

        # FatJet calibrations
        


        if self.Run==2:
            puid = 3 if '2016' in self.year  else 6
            # process puid 3(Medium) or 7(Tight) for 2016 and 6(Medium) or 7(Tight) for 2017/18
            ak4jets_unclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 6 and ((j.puId == puid or j.puId == 7) or j.pt > 50)]
        else:
            # No puid in Run3, because "default" jets are PuppiJets
            AK4PNetBWP = {"2022": {"L": 0.047, "M": 0.245, "T": 0.6734, "XT": 0.7862, "XXT": 0.961}, "2022EE": {"L": 0.0499, "M": 0.2605, "T": 0.6915, "XT": 0.8033, "XXT": 0.9664}, "2023": {"L": 0.0358, "M": 0.1917, "T": 0.6172, "XT": 0.7515, "XXT": 0.9659}, "2023BPix": {"L": 0.0359, "M": 0.1919, "T": 0.6133, "XT": 0.7544, "XXT": 0.9688}}
            ak4jets_unclean = [j for j in event._allJets if j.pt > 20 and abs(j.eta) < 2.5 and j.jetId >= 6 and j.btagPNetB > AK4PNetBWP[self.year]["T"]]
        # Clean Jets from Taus and Leptons
        event.ak4jets = []
        for j in ak4jets_unclean:
            goodjet = True
            for l in event.looseLeptons+event.looseTaus:
                if j.DeltaR(l) < 0.5:
                    goodjet = False
                    break
            if goodjet: event.ak4jets.append(j)

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
            #    if deltaR(fj,j) < 0.8: overlap = True # calculate overlap between small r jets and fatjets
            #if overlap: continue
            if self.Run==2:
                pNetSum = j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc + j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds
                if pNetSum > 0:
                    j.btagPNetB = (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb) / pNetSum
                    j.btagPNetC = (j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc) / (j.ParticleNetAK4_probb + j.ParticleNetAK4_probbb + j.ParticleNetAK4_probc + j.ParticleNetAK4_probcc + j.ParticleNetAK4_probg + j.ParticleNetAK4_probuds)
                    j.btagPNetBPlusC = j.btagPNetB + j.btagPNetC
                    j.btagPNetBVsC = j.btagPNetB / j.btagPNetBPlusC

                    # Calibrations + uncertainties (working points hardcoded)
                    #if self.isMC:
                    if '2017' in self.year or '2018' in self.year:
                        wps_pnet_b_plus_c = [0.5, 0.2, 0.1]
                        wps_pnet_b_vs_c = [0.99,0.96, 0.88, 0.7,0.4,0.15,0.05]
                    elif '2016' in self.year:
                        wps_pnet_b_plus_c = [0.35, 0.17, 0.1]
                        wps_pnet_b_vs_c = [0.99,0.96, 0.88, 0.7, 0.4, 0.15, 0.05]

                    # pseudo-continuous tagging in b+c and b vs c define tag
                    # need to define SF after the jet pt cuts
                    if j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC > wps_pnet_b_vs_c[0]: j.tag = 54
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[0] and j.btagPNetBVsC > wps_pnet_b_vs_c[1]: j.tag = 53
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[1] and j.btagPNetBVsC > wps_pnet_b_vs_c[2]: j.tag = 52
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[2] and j.btagPNetBVsC > wps_pnet_b_vs_c[3]: j.tag = 51
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[3] and j.btagPNetBVsC > wps_pnet_b_vs_c[4]: j.tag = 50

                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[6]: j.tag = 44
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[5] and j.btagPNetBVsC > wps_pnet_b_vs_c[6]: j.tag = 43
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[0] and j.btagPNetBVsC < wps_pnet_b_vs_c[4] and j.btagPNetBVsC > wps_pnet_b_vs_c[5]: j.tag = 42

                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[1] and j.btagPNetBPlusC < wps_pnet_b_plus_c[0] : j.tag = 41
                    elif j.btagPNetBPlusC > wps_pnet_b_plus_c[2] and j.btagPNetBPlusC < wps_pnet_b_plus_c[1] : j.tag = 40
                    elif j.btagPNetBPlusC < wps_pnet_b_plus_c[2]: j.tag = 0

                else:
                    j.btagPNetB = -1
                    j.btagPNetC = -1
                    j.btagPNetBPlusC = -1
                    j.btagPNetBVsC = -1
                    j.tag = -1
            
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

        for j in event.ak4jets:
            j.FatJetMatch = False
            j.FatJetMatchedIndex = -1

        index_fj = 0
        for fj in event.fatjets:
            index_fj += 1
            for j in event.ak4jets:
                if deltaR(fj,j) < 0.8:
                    j.FatJetMatch = True
                    j.FatJetMatchIndex = index_fj

        # Apply FTAG calibrations
        if self.Run ==2:

            if self.isMC: 
                for j in event.ak4jets:
                    #print("Jet properties",j.btagPNetBPlusC, j.btagPNetBVsC, j.pt, abs(j.eta))
                    if j.tag > -1:
                        j.ftagSF = self.ftagSF.get_sf(j)
                    else:
                        j.ftagSF = 1
                    if j.FatJetMatch == False: 
                        self.ftagwgts['flavTagWeight'] *= j.ftagSF

                    for syst in self.ftag_systematics:
                        if j.FatJetMatch == False: 
                            self.ftagwgts[f'flavTagWeight_{syst}_UP'] *= self.ftagSF.get_sf(j, 'up_' + syst)
                            self.ftagwgts[f'flavTagWeight_{syst}_DOWN'] *= self.ftagSF.get_sf(j, 'down_' + syst)


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

        pv = Object(event,"PV")
        self.out.fillBranch("npvs", pv.npvs)
        self.out.fillBranch("npvsGood", pv.npvsGood)


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
                if len(hadGenHs)>1:
                    self.out.fillBranch("genHiggs2Pt", hadGenHs[1].pt)
                    self.out.fillBranch("genHiggs2Eta", hadGenHs[1].eta)
                    self.out.fillBranch("genHiggs2Phi", hadGenHs[1].phi)
                    self.out.fillBranch("genHiggs2Decay", HDecayMode(hadGenHs[1]))

                    if len(hadGenHs)>2:
                        self.out.fillBranch("genHiggs3Pt", hadGenHs[2].pt)
                        self.out.fillBranch("genHiggs3Eta", hadGenHs[2].eta)
                        self.out.fillBranch("genHiggs3Phi", hadGenHs[2].phi)
                        self.out.fillBranch("genHiggs3Decay", HDecayMode(hadGenHs[2]))

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
            fill_fj(prefix + "PNetXbbTagCat", fj.tag)

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
                fill_fj(prefix + "PNetSF", fj.ftagSF)


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
            
            # lepton variables
            if fj:
                hasMuon = True if (closest(fj, event.cleaningMuons)[1] < 1.0) else False
                hasElectron = True if (closest(fj, event.cleaningElectrons)[1] < 1.0) else False
                hasBJetCSVLoose = True if (closest(fj, event.bljets)[1] < 1.0) else False
                hasBJetCSVMedium = True if (closest(fj, event.bmjetsCSV)[1] < 1.0) else False
                hasBJetCSVTight = True if (closest(fj, event.btjets)[1] < 1.0) else False
            else:
                hasMuon = False
                hasElectron = False
                hasBJetCSVLoose = False
                hasBJetCSVMedium = False
                hasBJetCSVTight = False
            fill_fj(prefix + "HasMuon", hasMuon)
            fill_fj(prefix + "HasElectron", hasElectron)
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
                for idx in ([1, 2]):
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

                for idx in ([1, 2]):
                    prefix = 'fatJet%i' % idx
                    fj = event.fatjetsJME[syst][idx - 1]
                    fill_fj = self._get_filler(fj)
                    fill_fj(prefix + "Pt" + "_" + syst, fj.pt)
                    fill_fj(prefix + "PtOverMHH" + "_" + syst, fj.pt/(h1Jet+h2Jet).M())

    def fillJetInfo(self, event, jets, fatjets, XbbWP, taus, XtautauWP):
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
            if self.Run == 2:
                fillBranch(prefix + "PNetC", j.btagPNetC)
                fillBranch(prefix + "PNetBPlusC", j.btagPNetBPlusC)
                fillBranch(prefix + "PNetBVsC", j.btagPNetBVsC)
                if j.tag and j.tag > -1:   
                    fillBranch(prefix + "PNetTagCat", self.ftag_mapping[j.tag])
                else: 
                    fillBranch(prefix + "PNetTagCat", -1)



            else:
                fillBranch(prefix + "PNetCvB", j.btagPNetCvB)
                fillBranch(prefix + "PNetCvL", j.btagPNetCvL)
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
                fillBranch(prefix + "FatJetMatched", j.FatJetMatch) # Fill for Data too
                fillBranch(prefix + "FatJetMatchedIndex", j.FatJetMatchIndex)

            if self.isMC:
                fillBranch(prefix + "HadronFlavour", j.hadronFlavour)
                fillBranch(prefix + "HiggsMatched", j.HiggsMatch)
                fillBranch(prefix + "Charge", j.Charge)
                fillBranch(prefix + "PdgId", j.pdgId)
                fillBranch(prefix + "DRGenQuark", j.drGen)
                fillBranch(prefix + "HiggsMatchedIndex", j.HiggsMatchIndex)
                fillBranch(prefix + "MatchedGenPt", j.MatchedGenPt)
                fillBranch(prefix + "PNetSF", j.ftagSF)

                
            if j:
                hasMuon = True if (closest(j, event.cleaningMuons)[1] < 0.5) else False
                hasElectron = True if (closest(j, event.cleaningElectrons)[1] < 0.5) else False
            else:
                hasMuon = False
                hasElectron = False

            fillBranch(prefix + "HasMuon", hasMuon)
            fillBranch(prefix + "HasElectron", hasElectron)

        if self.isMC:
            hadGenH_4vec = [polarP4(h) for h in self.hadGenHs]
            genHdaughter_4vec = [polarP4(d) for d in self.genHdaughter]

        event.reco6b_Idx = -1
        event.reco4b2t_Idx = -1
        event.reco4b2t_TauIsBoosted = 0
        event.reco4b2t_TauIsResolved = 0

        #if len(jets)+2*len(fatjets) > 5:
        if True:
            # Technique 3: mass fitter
            
            #m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5 = self.higgsPairingAlgorithm(event,jets,fatjets,XbbWP)
            event.reco6b_Idx,m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5,bla1,bla2 = higgsPairingAlgorithm_v2(event,jets,fatjets,XbbWP,self.isMC,self.Run)
                       
            self.out.fillBranch("h1_t3_mass", h1.Mass)
            self.out.fillBranch("h1_t3_pt", h1.pt)
            self.out.fillBranch("h1_t3_eta", abs(h1.eta))
            self.out.fillBranch("h1_t3_phi", h1.phi)
            self.out.fillBranch("h1_t3_match", h1.matchH1)
            self.out.fillBranch("h1_t3_dRjets", h1.dRjets)

            self.out.fillBranch("h2_t3_mass", h2.Mass)
            self.out.fillBranch("h2_t3_pt", h2.pt)
            self.out.fillBranch("h2_t3_eta", abs(h2.eta))
            self.out.fillBranch("h2_t3_phi", h2.phi)
            self.out.fillBranch("h2_t3_match", h2.matchH2)
            self.out.fillBranch("h2_t3_dRjets", h2.dRjets)

            self.out.fillBranch("h3_t3_mass", h3.Mass)
            self.out.fillBranch("h3_t3_pt", h3.pt)
            self.out.fillBranch("h3_t3_eta", abs(h3.eta))
            self.out.fillBranch("h3_t3_phi", h3.phi)
            self.out.fillBranch("h3_t3_match", h3.matchH3)
            self.out.fillBranch("h3_t3_dRjets", h3.dRjets)

            self.out.fillBranch("max_h_eta", max(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("min_h_eta", min(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
            self.out.fillBranch("max_h_dRjets", max(h1.dRjets,h2.dRjets,h3.dRjets))
            self.out.fillBranch("min_h_dRjets", min(h1.dRjets,h2.dRjets,h3.dRjets))

            self.out.fillBranch("h_fit_mass", m_fit)


            # Technique 3: mass fitter for Taus
            
            #m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5 = self.higgsPairingAlgorithm_v2(event,jets,fatjets,XbbWP,dotaus=True,taus=taus,XtautauWP=XtautauWP)
            if False:
                event.reco4b2t_Idx,m_fit,h1,h2,h3,j0,j1,j2,j3,j4,j5,event.reco4b2t_TauIsBoosted,event.reco4b2t_TauIsResolved = higgsPairingAlgorithm_v2(event,jets,fatjets,XbbWP,self.isMC,self.Run,dotaus=True,taus=taus,XtautauWP=XtautauWP,METvars=[event.PuppiMET_pt, event.PuppiMET_phi, event.MET_covXX, event.MET_covXY, event.MET_covYY])
                        
                self.out.fillBranch("h1_4b2t_mass", h1.Mass)
                self.out.fillBranch("h1_4b2t_pt", h1.pt)
                self.out.fillBranch("h1_4b2t_eta", abs(h1.eta))
                self.out.fillBranch("h1_4b2t_phi", h1.phi)
                self.out.fillBranch("h1_4b2t_match", h1.matchH1)
                self.out.fillBranch("h1_4b2t_dRjets", h1.dRjets)

                self.out.fillBranch("h2_4b2t_mass", h2.Mass)
                self.out.fillBranch("h2_4b2t_pt", h2.pt)
                self.out.fillBranch("h2_4b2t_eta", abs(h2.eta))
                self.out.fillBranch("h2_4b2t_phi", h2.phi)
                self.out.fillBranch("h2_4b2t_match", h2.matchH2)
                self.out.fillBranch("h2_4b2t_dRjets", h2.dRjets)

                self.out.fillBranch("h3_4b2t_mass", h3.Mass)
                self.out.fillBranch("h3_4b2t_pt", h3.pt)
                self.out.fillBranch("h3_4b2t_eta", abs(h3.eta))
                self.out.fillBranch("h3_4b2t_phi", h3.phi)
                self.out.fillBranch("h3_4b2t_match", h3.matchH3)
                self.out.fillBranch("h3_4b2t_dRjets", h3.dRjets)

                self.out.fillBranch("max_h_eta_4b2t", max(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
                self.out.fillBranch("min_h_eta_4b2t", min(abs(h1.eta), abs(h2.eta), abs(h3.eta)))
                self.out.fillBranch("max_h_dRjets_4b2t", max(h1.dRjets,h2.dRjets,h3.dRjets))
                self.out.fillBranch("min_h_dRjets_4b2t", min(h1.dRjets,h2.dRjets,h3.dRjets))

                self.out.fillBranch("h_fit_mass_4b2t", m_fit)

                self.out.fillBranch("h3_4b2t_phi", -1)
                self.out.fillBranch("reco6b_Idx", event.reco6b_Idx)
                self.out.fillBranch("reco4b2t_Idx", event.reco4b2t_Idx)
                self.out.fillBranch("reco4b2t_TauIsBoosted", event.reco4b2t_TauIsBoosted)
                self.out.fillBranch("reco4b2t_TauIsResolved", event.reco4b2t_TauIsResolved)

    def fillFTAGSF(self):

        for name, val in self.ftagwgts.items():
            self.out.fillBranch(name,val)
        for name, val in self.fj_ftagwgts.items():
            self.out.fillBranch(name,val)
            
    def fillLeptonInfo(self, event, leptons):
        for idx in ([1, 2]):
            lep = leptons[idx-1]if len(leptons)>idx-1 else _NullObject()
            prefix = 'lep%i'%(idx)
            fillBranch = self._get_filler(lep)
            fillBranch(prefix + "Pt", lep.pt)
            fillBranch(prefix + "Eta", lep.eta)
            fillBranch(prefix + "Phi", lep.phi)
            fillBranch(prefix + "Id", lep.Id)
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


    def fillClassification(self, output_values):

        self.out.fillBranch("probHHH", output_values[12][0][1] )
        self.out.fillBranch("probQCD", output_values[12][0][2] )
        self.out.fillBranch("probTT", output_values[12][0][3] )
        self.out.fillBranch("probVJets", output_values[12][0][4] )
        self.out.fillBranch("probVV", output_values[12][0][5] )
        self.out.fillBranch("probHHH4b2tau", output_values[12][0][6] )
        self.out.fillBranch("probHH4b", output_values[12][0][7] )
        self.out.fillBranch("probHH2b2tau", output_values[12][0][8] )


    def fillCategorisation(self, output_values):

        self.out.fillBranch("prob3bh0h", output_values[12][0][1] )
        self.out.fillBranch("prob2bh1h", output_values[12][0][2] )
        self.out.fillBranch("prob1bh2h", output_values[12][0][3] )
        self.out.fillBranch("prob0bh3h", output_values[12][0][4] )
        self.out.fillBranch("prob2bh0h", output_values[12][0][5] )
        self.out.fillBranch("prob1bh1h", output_values[12][0][6] )
        self.out.fillBranch("prob0bh2h", output_values[12][0][7] )
        self.out.fillBranch("prob1bh0h", output_values[12][0][8] )
        self.out.fillBranch("prob0bh1h", output_values[12][0][9] )
        self.out.fillBranch("prob0bh0h", output_values[12][0][0] )


        

    def prepare_inputs_spanet(self,event): # for SPANET ONNX inference
        jets = event.ak4jets[:10]
        fatjets = event.fatjets[0:3]
        met = float(event.met.pt)
        ht = float(event.ht)
        leptons = event.looseLeptons[:2]
        taus = event.looseTaus[:2] # hardcoded to be changed later

        # depends on if the inputs need a log transform from spanet

        # jet arrays
        array = []
        for i  in range(10):
            
            try:
                j = jets[i]
                if self.SpanetONNXNeedLogTransform:                     
                    arr = np.array([float(j.pt * j.bRegCorr), float(j.eta), float(np.sin(j.phi)),float(np.cos(j.phi)), float(j.btagPNetB),float(np.log(j.mass+1)),]).astype(np.float32) # order matters here
                else:
                    arr = np.array([float(j.pt * j.bRegCorr), float(j.eta), float(np.sin(j.phi)),float(np.cos(j.phi)), float(j.btagPNetB),float(j.mass),]).astype(np.float32) # order matters here
            except:
                arr = np.array([0.,0.,0.,0.,0.,0.]).astype(np.float32)
            array.append([arr])
        # fatjet arrays
        boosted_array = []
        for i in range(3):
            try:
                fj = fatjets[i]
                if self.SpanetONNXNeedLogTransform:  
                    boost_arr = np.array([float(np.log(fj.pt+1)),float(fj.eta),float(np.sin(fj.phi)), float(np.cos(fj.phi)), float(fj.Xbb), float(fj.Xjj), float(fj.particleNetMD_QCD), float(fj.particleNet_mass)]).astype(np.float32)    
                else:
                    boost_arr = np.array([float(fj.pt),float(fj.eta),float(np.sin(fj.phi)), float(np.cos(fj.phi)), float(fj.Xbb), float(fj.Xjj), float(fj.particleNetMD_QCD), float(fj.particleNet_mass)]).astype(np.float32)
            except:
                boost_arr = np.array([0.,0.,0.,0.,0.,0.,0.,0.]).astype(np.float32)
                
            boosted_array.append([boost_arr])

        # electrons and muons
        lep_array = []
        for i in range(2):
            try:
                lep = leptons[i]
                if self.SpanetONNXNeedLogTransform: 
                    lep_arr = np.array([float(np.log(lep.pt+1)), float(lep.eta), float(np.sin(lep.phi)),float(np.cos(lep.phi))]).astype(np.float32)
                else:
                    lep_arr = np.array([float(lep.pt), float(lep.eta), float(np.sin(lep.phi)),float(np.cos(lep.phi))]).astype(np.float32)
            except:
                lep_arr = np.array([0.,0.,0.,0.]).astype(np.float32)
            lep_array.append([lep_arr])

        # taus
        tau_array = []
        for i in range(2):
            try:
                tau = taus[i]
                if self.SpanetONNXNeedLogTransform: 
                    tau_arr = np.array([float(np.log(tau.pt+1)), float(tau.eta), float(np.sin(tau.phi)),float(np.cos(tau.phi)) ]).astype(np.float32)
                else: 
                    tau_arr = np.array([float(tau.pt), float(tau.eta), float(np.sin(tau.phi)),float(np.cos(tau.phi)) ]).astype(np.float32)
            except:
                tau_arr = np.array([0.,0.,0.,0.]).astype(np.float32)
            tau_array.append([tau_arr])

        # Higgses candidates from AK4
        higgs_array = {}
        for i in range(10):
            higgs_list = []
            name = 'Jet%d'%(i+1)
            try:
                jet1 = polarP4(jets[i])
                
                for j in range(10):
                    try:
                        jet2 = polarP4(jets[j])
                        if i == j: continue 
                        if j < i : continue 
                        higgs = jet1+jet2
                        if self.SpanetONNXNeedLogTransform: 
                            higgs_arr = np.array([ float(np.log(higgs.M()+1)), float(np.log(higgs.Pt()+1)),float(higgs.Eta()), float(np.sin(higgs.Phi())), float(np.cos(higgs.Phi())), float(deltaR(jets[i],jets[j])) ]).astype(np.float32)
                        else: 
                            higgs_arr = np.array([ float(higgs.M()), float(higgs.Pt()),float(higgs.Eta()), float(np.sin(higgs.Phi())), float(np.cos(higgs.Phi())), float(deltaR(jets[i],jets[j])) ]).astype(np.float32)
                        higgs_list.append(higgs_arr)
                    except:
                        higgs_arr = np.array([0.,0.,0.,0.,0.,0.]).astype(np.float32)
                        higgs_list.append(higgs_arr)

            except:
                for j in range(i+1, 10):
                    higgs_arr = np.array([0.,0.,0.,0.,0.,0.]).astype(np.float32)
                    higgs_list.append(higgs_arr)

            higgs_array[name] = [higgs_list]

        if self.SpanetONNXNeedLogTransform: 
            met_array = [np.array([np.log(met+1)])]
            ht_array = [np.array([np.log(ht+1)])]
            MIN_PT = np.log(20 + 1)
            MIN_FJPT = np.log(200+1)
            MIN_MASS = np.log(20+1)
        else:
            met_array = [np.array([met])]
            ht_array = [np.array([ht])]
            MIN_PT = 20
            MIN_FJPT = 200
            MIN_MASS = 20
        
        

        Jets_data = np.transpose(array,(1,0,2)).astype(np.float32)
        Jets_mask = Jets_data[:,:,0] > MIN_PT

        BoostedJets_data = np.transpose(boosted_array,(1,0,2)).astype(np.float32)
        BoostedJets_mask = BoostedJets_data[:,:,0] > MIN_FJPT

        Leptons_data = np.transpose(lep_array,(1,0,2)).astype(np.float32)
        Leptons_mask = Leptons_data[:,:,0] > MIN_PT

        Taus_data = np.transpose(tau_array,(1,0,2)).astype(np.float32)
        Taus_mask = Taus_data[:,:,0] > MIN_PT

        MET_data = np.transpose([met_array],(1,0,2)).astype(np.float32)
        MET_mask = MET_data[:,:,0] > 0

        HT_data = np.transpose([ht_array],(1,0,2)).astype(np.float32)
        HT_mask = MET_data[:,:,0] > 0

        Jet1_data = np.array(higgs_array['Jet1']).astype(np.float32)
        Jet1_mask = Jet1_data[:,:,0] > MIN_MASS

        Jet2_data = np.array(higgs_array['Jet2']).astype(np.float32)
        Jet2_mask = Jet2_data[:,:,0] > MIN_MASS

        Jet3_data = np.array(higgs_array['Jet3']).astype(np.float32)
        Jet3_mask = Jet3_data[:,:,0] > MIN_MASS

        Jet4_data = np.array(higgs_array['Jet4']).astype(np.float32)
        Jet4_mask = Jet4_data[:,:,0] > MIN_MASS

        Jet5_data = np.array(higgs_array['Jet5']).astype(np.float32)
        Jet5_mask = Jet5_data[:,:,0] > MIN_MASS

        Jet6_data = np.array(higgs_array['Jet6']).astype(np.float32)
        Jet6_mask = Jet6_data[:,:,0] > MIN_MASS

        Jet7_data = np.array(higgs_array['Jet7']).astype(np.float32)
        Jet7_mask = Jet7_data[:,:,0] > MIN_MASS

        Jet8_data = np.array(higgs_array['Jet8']).astype(np.float32)
        Jet8_mask = Jet8_data[:,:,0] > MIN_MASS

        Jet9_data = np.array(higgs_array['Jet9']).astype(np.float32)
        Jet9_mask = Jet9_data[:,:,0] > MIN_MASS

        #Jet10_data = np.array(higgs_array['Jet10']).astype(np.float32)
        #Jet10_mask = Jet10_data[:,:,0] > 20
        input_dict = {"Jets_data": Jets_data, "Jets_mask": Jets_mask, "BoostedJets_data":BoostedJets_data, "BoostedJets_mask": BoostedJets_mask, "Leptons_data" : Leptons_data, "Leptons_mask" : Leptons_mask, 'Taus_data' : Taus_data, 'Taus_mask': Taus_mask, "MET_data" : MET_data, "MET_mask": MET_mask, 'HT_data': HT_data, "HT_mask" : HT_mask, 'Jet1_data' : Jet1_data, 'Jet1_mask': Jet1_mask, 'Jet2_data' : Jet2_data, 'Jet2_mask': Jet2_mask, 'Jet3_data' : Jet3_data, 'Jet3_mask': Jet3_mask, 'Jet4_data' : Jet4_data, 'Jet4_mask': Jet4_mask, 'Jet5_data' : Jet5_data, 'Jet5_mask': Jet5_mask, 'Jet6_data' : Jet6_data, 'Jet6_mask': Jet6_mask, 'Jet7_data' : Jet7_data, 'Jet7_mask': Jet7_mask, 'Jet8_data' : Jet8_data, 'Jet8_mask': Jet8_mask,'Jet9_data' : Jet9_data, 'Jet9_mask': Jet9_mask}
        #for key, value in input_dict.items():
        #    print(key, value.shape)
        return input_dict
    
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        
        # First pre-selection - makes run 5x faster
        allJets = Collection(event, "Jet")
        loose_jets = [j for j in allJets if j.pt > 20 and abs(j.eta) < 2.5]
        
        if len(loose_jets) < 4 : return False # Match trigger and stop loop here

        # fill histograms
        event.gweight = 1
        if self.isMC:
            event.gweight = event.genWeight / abs(event.genWeight)



        # select leptons and correct jets
        self.selectLeptons(event)

        if len(event.looseLeptons) > 0: return False # Veto loose leptons (electrons and muons)

        # Iniatialise ftag weight for each event
        self.ftagwgts = self.ftagbasewgts.copy()
        self.fj_ftagwgts = self.fj_ftagbasewgts.copy()

        self.correctJetsAndMET(event)          
        
        # basic jet selection 
        #probe_jets = [fj for fj in event.fatjets if fj.pt > 300 and fj.Xbb > 0.8]
        probe_jets = [fj for fj in event.fatjets if fj.pt > 215 and abs(fj.eta) < 2.5 and fj.jetId >= 2] # 215 GeV cut good for PNet scores
        
        #probe_jets.sort(key=lambda x: x.pt, reverse=True)
        probe_jets.sort(key=lambda x: x.Xbb, reverse=True)
        pass1AK8PNet = False
        if len(probe_jets) > 0:
            if probe_jets[0].Xbb > 0.8:
                pass1AK8PNet = True



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
            if (self.nSmallJets > 3 or pass1AK8PNet): passSel = True

        if not passSel: return False

        # load gen history
        hadGenHs = self.loadGenHistory(event, probe_jets, event.ak4jets)
        self.hadGenHs = hadGenHs

        for j in event.ak4jets+event.looseTaus:
            j.HiggsMatch = False
            j.FatJetMatch = False
            j.HiggsMatchIndex = -1
            j.FatJetMatchIndex = -1
            j.MatchedGenPt = 0.
            j.Charge = 0
            j.pdgId = -999
            j.drGen = 0.
            

        for fj in probe_jets:
            fj.HiggsMatch = False
            fj.HiggsMatchIndex = -1
            fj.MatchedGenPt = 0.

        if self.isMC:
            daughters = []
            matched = 0
            for index_h, higgs_gen in enumerate(hadGenHs):
                for idx in higgs_gen.dauIdx:
                    dau = event.genparts[idx]
                    daughters.append(dau)
                    for j in event.ak4jets+event.looseTaus:
                        if deltaR(j,dau) < 0.4:
                            j.HiggsMatch = True
                            j.HiggsMatchIndex = index_h+1
                            j.MatchedGenPt = dau.pt
                            matched += 1
                for fj in probe_jets:
                    if deltaR(higgs_gen, fj) < 0.8:
                        fj.HiggsMatch = True
                        fj.HiggsMatchIndex = index_h+1
                        fj.MatchedGenPt = higgs_gen.pt
                
                for jet in event.ak4jets:
                    if abs(jet.dr_genQuark) < 0.4:
                        jet.Charge = int(jet.genQuark.pdgId/abs(jet.genQuark.pdgId))
                        jet.drGen = jet.dr_genQuark
                        jet.pdgId = jet.genQuark.pdgId
                    else:
                        jet.Charge = int(0)


            self.out.fillBranch("nHiggsMatchedJets", matched)

        #print("Matched outside fillJetInfo", matched)
        if self.isMC:
            self.genHdaughter = daughters
        index_fj = 0
        for fj in probe_jets:
            index_fj += 1
            for j in event.ak4jets+event.looseTaus:
                if deltaR(fj,j) < 0.8:
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
        if self.Run == 3:
            self.out.fillBranch("nprobetaus", len([fj for fj in probe_jets if fj.pt > 200 and fj.Xtautau > XtautauWP]))
        #print(len(probe_jets))
        #if len(probe_jets) > 0:
        self.fillFatJetInfo(event, probe_jets)
          
        # for ak4 jets we only fill the b-tagged medium jets
        #self.fillJetInfo(event, event.bmjets)
        #self.fillJetInfo(event, event.bljets)
        try:
            self.fillJetInfo(event, event.ak4jets, probe_jets, XbbWP, event.looseTaus, XtautauWP)
        except IndexError:
            return False

        self.fillLeptonInfo(event, event.looseLeptons)
        self.fillTauInfo(event, event.looseTaus)
 
        # for all jme systs
        if self._allJME and self.isMC:
            self.fillFatJetInfoJME(event, probe_jets)


        if self.isMC:
            self.fillFTAGSF()
        #self.fillTriggerFilters(event) 

        # SPANET inference

        doSpanet = False
        if doSpanet:
            input_dict = self.prepare_inputs_spanet(event)
            output_classification = self.session_classification.run(self.output_nodes_classification, input_dict)

            output_categorisation = self.session_categorisation.run(self.output_nodes_categorisation, input_dict)

            self.fillClassification(output_classification)
            self.fillCategorisation(output_categorisation)

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

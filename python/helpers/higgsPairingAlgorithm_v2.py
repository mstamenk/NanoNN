import ROOT
import math
import itertools
from PhysicsTools.NanoNN.helpers.utils import polarP4, deltaR

def manualPermutation(input1, input2, input3=[0]):
  output = []
  for i1,v1 in enumerate(input1):
    for i2,v2 in enumerate(input2):
      if v1==v2 or (input1==input2 and i2<=i1): continue
      if v1.startswith("JP") and v2.startswith("JP") and len(list(set(v1.split("_")[1:]+v2.split("_")[1:])))<4: continue
      if v1.startswith("TauP") and v2.startswith("TauP") and len(list(set(v1.split("_")[1:]+v2.split("_")[1:])))<4: continue
      for i3,v3 in enumerate(input3):
        if v3!=0:
          if v1==v3 or (input1==input3 and i3<=i1): continue
          if v2==v3 or (input2==input3 and i3<=i2): continue
          if v1.startswith("JP") and v3.startswith("JP") and len(list(set(v1.split("_")[1:]+v3.split("_")[1:])))<4: continue
          if v1.startswith("TauP") and v3.startswith("TauP") and len(list(set(v1.split("_")[1:]+v3.split("_")[1:])))<4: continue
          if v2.startswith("JP") and v3.startswith("JP") and len(list(set(v2.split("_")[1:]+v3.split("_")[1:])))<4: continue
          if v2.startswith("TauP") and v3.startswith("TauP") and len(list(set(v2.split("_")[1:]+v3.split("_")[1:])))<4: continue
          output.append([v1,v2,v3])
        else:
          output.append([v1,v2])
  return output

def higgsPairingAlgorithm_v2(event, jets, fatjets, XbbWP, isMC, Run, jetdicsr, jetWP, dotaus=False, taus=[], TauVsEl=1, TauVsMu=1, TauVsJet=1, XtautauWP=0.0, leptons=[], METvars=[]):

    # save jets properties
    dummyJet = polarP4()
    dummyJet.HiggsMatch = False
    dummyJet.HiggsMatchIndex = -1
    dummyJet.FatJetMatch = False
    dummyJet.btagDeepFlavB = -1
    dummyJet.btagPNetB = -1
    dummyJet.DM = -1
    dummyJet.kind = -1
    dummyJet.DeepTauVsJet = -1
    dummyJet.hadronFlavour = -1
    dummyJet.jetId = -1
    dummyJet.puId = -1
    dummyJet.pdgId = -1
    dummyJet.rawFactor = -1
    dummyJet.bRegCorr = -1
    dummyJet.bRegRes = -1
    dummyJet.cRegCorr = -1
    dummyJet.cRegRes = -1
    dummyJet.MatchedGenPt = 0
    dummyJet.mass = 0.

    dummyHiggs = polarP4()
    dummyHiggs.matchH = 0
    dummyHiggs.mass = 0.
    dummyHiggs.Mass = 0.
    dummyHiggs.pt = -99.
    dummyHiggs.eta = -99.
    dummyHiggs.phi = -99.
    dummyHiggs.dRjets = -1
    dummyHiggs.FJ1idx = -1
    dummyHiggs.FJ2idx = -1
    dummyHiggs.J1idx = -1
    dummyHiggs.J2idx = -1
    dummyHiggs.T1idx = -1
    dummyHiggs.T2idx = -1
    dummyHiggs.L1idx = -1
    dummyHiggs.L2idx = -1

    # Prepare FatJets
    AK4jetsfromAK8 = [fj for fj in fatjets if fj.t21>0.55 or fj.n3b1<=0]
    AK8fatjets = [fj for fj in fatjets if fj not in AK4jetsfromAK8]

    probetau = sorted([fj for fj in AK8fatjets if fj.Xtauany > XtautauWP], key=lambda x: x.Xtauany, reverse = True)
    if len(probetau)>0:
        probetau = probetau[0]
    probejets = sorted([fj for fj in AK8fatjets if fj.Xbb > XbbWP and fj!=probetau], key=lambda x: x.Xbb, reverse = True)
    if len(probejets) > 4:
        probejets = probejets[:4]
    if probetau!=[]:
        if dotaus:
            if len(probejets) <= 3:
                probejets.append(probetau)
            else:
                probejets = probejets[:3]+[probetau]
        elif probetau.Xbb > XbbWP:
            if len(probejets) <= 3:
                probejets.append(probetau)
            elif probetau.Xbb > probejets[3].Xbb:
                probejets[3] = probetau
        if not dotaus: probetau = []

    if dotaus:
        # Prepare MET variables for FastMTT
        MET_x = METvars[0]*math.cos(METvars[1])
        MET_y = METvars[0]*math.sin(METvars[1])
        covMET = ROOT.TMatrixD(2,2)
        covMET[0][0] = METvars[2]
        covMET[1][0] = METvars[3]
        covMET[0][1] = METvars[3]
        covMET[1][1] = METvars[4]

    # Prepare Leptons
    leptons_4vec = []
    for lidx,l in enumerate(leptons):
        overlap = False
        for fj in probejets:
            if deltaR(l,fj) < 1.0: overlap = True
        if overlap == False:
            l_tmp = polarP4(l)
            l_tmp.lidx = lidx+1
            l_tmp.tidx = -1
            l_tmp.HiggsMatch = l.HiggsMatch
            l_tmp.HiggsMatchIndex = l.HiggsMatchIndex
            l_tmp.FatJetMatch = l.FatJetMatch
            l_tmp.charge = l.charge
            l_tmp.btagDeepFlavB = -1
            l_tmp.btagPNetB = -1
            l_tmp.kind = l.kind
            l_tmp.DM = -1
            l_tmp.DeepTauVsJet = -1
            if isMC:
                l_tmp.hadronFlavour = -1
            l_tmp.jetId = -1
            l_tmp.pdgId = l.Id
            l_tmp.rawFactor = -1
            l_tmp.mass = l.mass
            l_tmp.MatchedGenPt = l.MatchedGenPt
            if Run==2:
                l_tmp.puId = -1
                l_tmp.bRegCorr = -1
                l_tmp.bRegRes = -1
                l_tmp.cRegCorr = -1
                l_tmp.cRegRes = -1

            leptons_4vec.append(l_tmp)

    if len(leptons_4vec) > 4:
        leptons_4vec = leptons_4vec[:4]

    # Prepare Taus
    taus_4vec = []
    for tidx,t in enumerate(taus):
        if Run==2:
            if t.idDeepTau2017v2p1VSe < TauVsEl: continue
            if t.idDeepTau2017v2p1VSmu < TauVsMu: continue
            if t.idDeepTau2017v2p1VSjet < TauVsJet: continue
        else:
            if t.idDeepTau2018v2p5VSe < TauVsEl: continue
            if t.idDeepTau2018v2p5VSmu < TauVsMu: continue
            if t.idDeepTau2018v2p5VSjet < TauVsJet: continue
        overlap = False
        for fj in probejets:
            if deltaR(t,fj) < 1.0: overlap = True
        for l in leptons_4vec:
            if deltaR(t.eta,t.phi,l.Eta(),l.Phi()) < 0.5: overlap = True
        if overlap == False:
            t_tmp = polarP4(t)
            t_tmp.tidx = tidx+1
            t_tmp.lidx = -1
            t_tmp.HiggsMatch = t.HiggsMatch
            t_tmp.HiggsMatchIndex = t.HiggsMatchIndex
            t_tmp.FatJetMatch = t.FatJetMatch
            t_tmp.charge = t.charge
            t_tmp.btagDeepFlavB = -1
            t_tmp.btagPNetB = -1
            t_tmp.kind = t.kind
            t_tmp.DM = t.decayMode
            if Run==2:
                t_tmp.DeepTauVsJet = t.rawDeepTau2017v2p1VSjet
            else:
                t_tmp.DeepTauVsJet = t.rawDeepTau2018v2p5VSjet
            if isMC:
                t_tmp.hadronFlavour = -1
            t_tmp.jetId = -1
            t_tmp.pdgId = t.Id
            t_tmp.rawFactor = -1
            t_tmp.mass = t.mass
            t_tmp.MatchedGenPt = t.MatchedGenPt
            if Run==2:
                t_tmp.puId = -1
                t_tmp.bRegCorr = -1
                t_tmp.bRegRes = -1
                t_tmp.cRegCorr = -1
                t_tmp.cRegRes = -1

            taus_4vec.append(t_tmp)

    if len(taus_4vec) > 4:
        taus_4vec = taus_4vec[:4]

    # Make Lepton-Tau candidate pairs
    if dotaus:
      taulepton_4vec = taus_4vec+leptons_4vec
    else:
      taulepton_4vec = []
    taupairs = []
    for i1,t1 in enumerate(taulepton_4vec):
      for i2,t2 in enumerate(taulepton_4vec):
        if i2<=i1: continue
        if t1.charge * t2.charge >= 0: continue
        tau1 = ROOT.MeasuredTauLepton(t1.kind, t1.Pt(), t1.Eta(), t1.Phi(), t1.mass, t1.DM)
        tau2 = ROOT.MeasuredTauLepton(t2.kind, t2.Pt(), t2.Eta(), t2.Phi(), t2.mass, t2.DM)
        VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
        bothtaus = VectorOfTaus()
        bothtaus.push_back(tau1)
        bothtaus.push_back(tau2)
        FMTT = ROOT.FastMTT()
        FMTT.run(bothtaus, MET_x, MET_y, covMET)
        FMTToutput = FMTT.getBestP4()
        FastMTTmass = FMTToutput.M()
        taupairs.append((i1,i2,(t1+t2),t1.DeepTauVsJet*t2.DeepTauVsJet,"Tau",FastMTTmass,t1.pdgId*t2.pdgId)) # IdxTau/Lep1, IdxTau/Lep2, RecoHiggs, Comb.DeepTauScore, "Tau", RecoHiggsMass, TauFinalState
    #taupairs = sorted([p for p in taupairs], key=lambda x: x[3], reverse = True)
    taupairs_tautau = sorted([p for p in taupairs if p[6]==-15*15], key=lambda x: x[3], reverse = True) # All TauTau pairs, sorted by factor of both VSjet scores
    taupairs_leptau = sorted([p for p in taupairs if p[6] in [-13*15,-11*15]], key=lambda x: x[3], reverse = False) # All LepTau pairs, sorted by the Tau's VSjet score (values are all negative, because of the *(-1) of the lepton)
    taupairs_leplep = [p for p in taupairs if p[6] in [-11*13,-13*13,-11*11]] # All e-mu/mu-mu/e-e pairs. Not sorted, so should be sorted by pT already
    taupairs = taupairs_tautau + taupairs_leptau + taupairs_leplep # Merge like this

    # Prepare (AK4) Jets
    jets_4vec = []
    for jidx,j in enumerate(jets):
        if jetdicsr!="":
            if getattr(j, jetdicsr) < jetWP: continue
        overlap = False
        for fj in probejets:
            if deltaR(j,fj) < 1.0: overlap = True
        for t in taus_4vec:
            if deltaR(j.eta,j.phi,t.Eta(),t.Phi()) < 0.5: overlap = True
        # Jet-Lepton overlap already done before
        if overlap == False:
            j_tmp = polarP4(j)
            j_tmp.jidx = jidx+1
            j_tmp.fjidx = -1
            j_tmp.HiggsMatch = j.HiggsMatch
            j_tmp.HiggsMatchIndex = j.HiggsMatchIndex
            j_tmp.FatJetMatch = j.FatJetMatch
            j_tmp.btagDeepFlavB = j.btagDeepFlavB
            j_tmp.btagPNetB = j.btagPNetB
            j_tmp.DM = -1
            j_tmp.kind = -1
            j_tmp.DeepTauVsJet = -1
            if isMC:
                j_tmp.hadronFlavour = j.hadronFlavour
            j_tmp.jetId = j.jetId
            j_tmp.pdgId = -1
            j_tmp.rawFactor = j.rawFactor
            j_tmp.mass = j.mass
            j_tmp.MatchedGenPt = j.MatchedGenPt
            if Run==2:
                j_tmp.puId = j.puId
                j_tmp.bRegCorr = j.bRegCorr
                j_tmp.bRegRes = j.bRegRes
                j_tmp.cRegCorr = j.cRegCorr
                j_tmp.cRegRes = j.cRegRes

            jets_4vec.append(j_tmp)

    # Add FatJets which are identified to be AK4 jets to AK4 jets collection
    for ak4j in AK4jetsfromAK8:
        fjidx = fatjets.index(ak4j)
        if ak4j.Xbb < XbbWP: continue # Going to assume that only AK4 jets can be mis-ID'd as AK8, not Taus as AK8 (We couldn't treat those as actual Taus anyway if we're missing info such as decay mode)
        i=1
        for fj in fatjets:
            if ak4j==fj: break
            i+=1
        overlap = False
        for j in jets:
            if j.FatJetMatchIndex == i: overlap=True
        for t in taus_4vec:
            if deltaR(ak4j.eta,ak4j.phi,t.Eta(),t.Phi()) < 0.5: overlap = True
        for l in leptons_4vec:
            if deltaR(ak4j.eta,ak4j.phi,l.Eta(),l.Phi()) < 0.5: overlap = True
        if overlap == False:
            j_tmp = polarP4(ak4j)
            j_tmp.jidx = -1
            j_tmp.fjidx = fjidx+1
            j_tmp.HiggsMatch = ak4j.HiggsMatch
            j_tmp.HiggsMatchIndex = ak4j.HiggsMatchIndex
            j_tmp.FatJetMatch = False
            j_tmp.btagDeepFlavB = -1
            j_tmp.btagPNetB = ak4j.Xbb
            j_tmp.DM = -1
            j_tmp.kind = -1
            j_tmp.DeepTauVsJet = -1
            if isMC:
                j_tmp.hadronFlavour = ak4j.hadronFlavour
            j_tmp.jetId = ak4j.jetId
            j_tmp.pdgId = -1
            j_tmp.rawFactor = ak4j.rawFactor
            j_tmp.mass = ak4j.mass
            j_tmp.MatchedGenPt = ak4j.MatchedGenPt
            if Run==2:
                j_tmp.puId = -1
                j_tmp.bRegCorr = -1
                j_tmp.bRegRes = -1
                j_tmp.cRegCorr = -1
                j_tmp.cRegRes = -1
            jets_4vec.append(j_tmp)

    jets_4vec = sorted([j for j in jets_4vec], key=lambda x: x.btagPNetB, reverse = True)

    if len(jets_4vec) > 10:
        jets_4vec = jets_4vec[:10]

    # Make resolved jet candidate pairs
    jetpairs = []
    for i1,j1 in enumerate(jets_4vec):
      for i2,j2 in enumerate(jets_4vec):
        if i2<=i1: continue
        jetpairs.append((i1,i2,(j1+j2),j1.btagPNetB*j2.btagPNetB,"Jet",(j1+j2).M())) # IdxJet1, IdxJet2, RecoHiggs, Comb.BtagScore, "Jet", RecoHiggsMass
    jetpairs = sorted([p for p in jetpairs], key=lambda x: x[3], reverse = True)

    # Combine all boosted FatJet and resolved pair candidates
    allobjects = {}
    for i,fj in enumerate(probejets):
      if fj!=probetau: allobjects["FJ"+str(i+1)] = fj
    if probetau!=[]: allobjects["FatTau"] = probetau
    for i,j in enumerate(jetpairs):
      allobjects["JP_"+str(j[0]+1)+"_"+str(j[1]+1)] = j
    for i,t in enumerate(taupairs):
      if t[6]==-15*15: FS = "_TauTau"
      elif t[6]==-13*15: FS = "_MuTau"
      elif t[6]==-11*15: FS = "_ElTau"
      elif t[6]==-11*13: FS = "_ElMu"
      elif t[6]==-13*13: FS = "_MuMu"
      elif t[6]==-11*11: FS = "_ElEl"
      allobjects["TauP_"+str(t[0]+1)+"_"+str(t[1]+1)+FS] = t
    objlist_jet = [k for k in allobjects if "J" in k]
    objlist_tau = [k for k in allobjects if "Tau" in k]

    if dotaus:
      # Try to make 3H: 2b+1t
      permutations = manualPermutation(objlist_jet, objlist_jet, objlist_tau)
      if permutations==[]:
        # Either 2b only or 1b+1t
        permnotau = manualPermutation(objlist_jet, objlist_jet)
        permoneb = manualPermutation(objlist_jet, objlist_tau)
        assert permnotau==[] or permoneb==[]
        permutations = permnotau+permoneb
      if permutations==[]:
        # Either 1b only or 1t only
        #assert (objlist_jet==[] or objlist_tau==[]) and len(objlist_jet+objlist_tau)<=3, "Shouldn't this be just one pair? ["+", ".join(objlist_jet)+"] and ["+", ".join(objlist_tau)+"]" # We CAN have multiple Tau pairs! Previous steps fail because there's not a single jet pair
        assert (objlist_jet==[] or objlist_tau==[]) and len(objlist_jet)<=3
        if len(objlist_jet+objlist_tau)>=1:
          permutations = [[obj] for obj in objlist_jet+objlist_tau]
    else:
      # Try 3b
      permutations = manualPermutation(objlist_jet, objlist_jet, objlist_jet)
      if permutations==[]:
        # Try 2b
        permutations = manualPermutation(objlist_jet, objlist_jet)
      if permutations==[]:
        # Either 1b or 0b
        # "Worst case" can still have 3 jets, so 3 jet pairs. Since list is b-score-sorted, will just choose the one with highest score
        assert len(objlist_jet)<=3
        if len(objlist_jet)>=1:
          permutations = [[obj] for obj in objlist_jet]

    h = []
    for i in range(3):
      h.append(dummyHiggs)
    j = []
    for i in range(6):
      j.append(dummyJet)
    TauIsBoosted = 0
    TauIsResolved = 0
    TauFinalState = 0
    if permutations==[]: return 0,0,h[0],h[1],h[2],j[0],j[1],j[2],j[3],j[4],j[5],TauIsBoosted,TauIsResolved,TauFinalState
    nHiggs = len(permutations[0])
    assert all([nHiggs==len(perm) for perm in permutations])
    assert all([len([name for name in perm if "Tau" in name])<=1 for perm in permutations])

    if False: # Select the pair that results in masses being the most similar to each other
      min_chi2 = 1000000000000000
      #print("========")
      #print(dotaus)
      #if dotaus:
      #  print("N_tau:",len(taus_4vec),"; N_lep:",len(leptons_4vec))
      #print(permutations)
      for permutation in permutations:
        masses = []
        for name in permutation:
          thismass = allobjects[name].mass if name.startswith("F") else allobjects[name][5]
          masses.append(thismass)
        fitted_mass = sum(masses)/nHiggs
        chi2 = 0.0
        for m in masses:
          chi2 += (m - fitted_mass)**2
        if chi2 < min_chi2:
          m_fit = fitted_mass
          min_chi2 = chi2
          finalPermutation = permutation
        if nHiggs==1: break # Use the jet pair with highest score, which will be always the first
    else: # Select pair such that resulting masses are close to Higgs boson mass
      min_chi2 = 1000000000000000
      for permutation in permutations:
        masses = []
        expmasses = []
        for name in permutation:
          # The reconstructed Higgs mass depends on its pT
          if name.startswith("F"):
            masses.append(allobjects[name].mass)
            pt = allobjects[name].pt
            if "Tau" in name:
              # Generally lower than Hbb FatJets because neutrino contribution is completely missing here
              expmasses.append(-61.585812326376264 + 0.05557800375177941*pt + 160.86030949003612 * math.erf(0.00369513146940551*pt))
            else:
              expmasses.append(-351.51178268548455 - 0.0047614727864319405*pt + 483.2014163177754 * math.erf(0.005008209116766995*pt))
          else:
            masses.append(allobjects[name][5])
            pt = allobjects[name][2].Pt()
            expmasses.append(94.30387125781559 + 0.15218879725691964*pt - 0.00030092926621392484*pt*pt + 2.6331953438047053e-07*pt*pt*pt)
        chi2 = 0.0
        for m in range(len(masses)):
          chi2 += (masses[m] - expmasses[m])**2
        if chi2 < min_chi2:
          m_fit = sum(masses)/nHiggs
          min_chi2 = chi2
          finalPermutation = permutation
      

    nBoostedH = len([name for name in finalPermutation if name.startswith("F")])
    jetCandCount = 0
    for i in range(nHiggs):
      if finalPermutation[i].startswith("F"):
        h[i] = allobjects[finalPermutation[i]]
        h[i].FJ1idx = fatjets.index(allobjects[finalPermutation[i]])+1
        h[i].FJ2idx = -1
        h[i].J1idx = -1
        h[i].J2idx = -1
        h[i].T1idx = -1
        h[i].T2idx = -1
        h[i].L1idx = -1
        h[i].L2idx = -1
        h[i].matchH = h[i].HiggsMatchIndex if h[i].HiggsMatchIndex>=0 else 0
        if Run==2:
          h[i].Mass = h[i].particleNet_mass
        else:
          h[i].Mass = h[i].mass*h[i].particleNet_massCorr
        h[i].dRjets = -1
        if "Tau" in finalPermutation[i]:
          TauIsBoosted = i+1
          if h[i].hasMuon and h[i].hasElectron:
            if h[i].Xtaumu > h[i].Xtaue:
              TauFinalState = -13*15
            else:
              TauFinalState = -11*15
          elif h[i].hasMuon:
            TauFinalState = -13*15
          elif h[i].hasElectron:
            TauFinalState = -11*15
          else:
            TauFinalState = -15*15
      else:
        h[i] = allobjects[finalPermutation[i]][2]
        h[i].FJ1idx = -1
        h[i].FJ2idx = -1
        h[i].J1idx = -1
        h[i].J2idx = -1
        h[i].T1idx = -1
        h[i].T2idx = -1
        h[i].L1idx = -1
        h[i].L2idx = -1
        h[i].Mass = allobjects[finalPermutation[i]][5] # h[i].M()
        h[i].pt = h[i].Pt()
        h[i].eta = h[i].Eta()
        h[i].phi = h[i].Phi()
        h[i].matchH = 0
        if allobjects[finalPermutation[i]][4]=="Jet":
          j[jetCandCount] = jets_4vec[allobjects[finalPermutation[i]][0]]
          j[jetCandCount+1] = jets_4vec[allobjects[finalPermutation[i]][1]]
          h[i].dRjets = deltaR(j[jetCandCount].Eta(),j[jetCandCount].Phi(),j[jetCandCount+1].Eta(),j[jetCandCount+1].Phi())
          if j[jetCandCount].HiggsMatch == True and j[jetCandCount+1].HiggsMatch == True and j[jetCandCount].HiggsMatchIndex == j[jetCandCount+1].HiggsMatchIndex:
            h[i].matchH = j[jetCandCount].HiggsMatchIndex
          if j[jetCandCount].jidx > 0 and j[jetCandCount+1].jidx > 0:
            h[i].J1idx = min(j[jetCandCount].jidx, j[jetCandCount+1].jidx)
            h[i].J2idx = max(j[jetCandCount].jidx, j[jetCandCount+1].jidx)
          elif j[jetCandCount].jidx == -1 and j[jetCandCount+1].jidx == -1:
            h[i].FJ1idx = min(j[jetCandCount].fjidx, j[jetCandCount+1].fjidx)
            h[i].FJ2idx = max(j[jetCandCount].fjidx, j[jetCandCount+1].fjidx)
          else:
            h[i].J1idx = max(j[jetCandCount].jidx, j[jetCandCount+1].jidx)
            h[i].FJ1idx = max(j[jetCandCount].fjidx, j[jetCandCount+1].fjidx)
        elif allobjects[finalPermutation[i]][4]=="Tau":
          j[jetCandCount] = taulepton_4vec[allobjects[finalPermutation[i]][0]]
          j[jetCandCount+1] = taulepton_4vec[allobjects[finalPermutation[i]][1]]
          h[i].dRjets = deltaR(j[jetCandCount].Eta(),j[jetCandCount].Phi(),j[jetCandCount+1].Eta(),j[jetCandCount+1].Phi())
          if j[jetCandCount].HiggsMatch == True and j[jetCandCount+1].HiggsMatch == True and j[jetCandCount].HiggsMatchIndex == j[jetCandCount+1].HiggsMatchIndex:
            h[i].matchH = j[jetCandCount].HiggsMatchIndex
          if j[jetCandCount].tidx > 0 and j[jetCandCount+1].tidx > 0:
            h[i].T1idx = min(j[jetCandCount].tidx, j[jetCandCount+1].tidx)
            h[i].T2idx = max(j[jetCandCount].tidx, j[jetCandCount+1].tidx)
          elif j[jetCandCount].tidx == -1 and j[jetCandCount+1].tidx == -1:
            h[i].L1idx = min(j[jetCandCount].lidx, j[jetCandCount+1].lidx)
            h[i].L2idx = max(j[jetCandCount].lidx, j[jetCandCount+1].lidx)
          else:
            h[i].T1idx = max(j[jetCandCount].tidx, j[jetCandCount+1].tidx)
            h[i].L1idx = max(j[jetCandCount].lidx, j[jetCandCount+1].lidx)
        jetCandCount += 2
        if "Tau" in finalPermutation[i]:
          TauIsResolved = i+1
          TauFinalState = allobjects[finalPermutation[i]][6]
    if nHiggs==3:
      #if nBoostedH==3: recoidx = 1
      #elif nBoostedH==2: recoidx = 2
      #elif nBoostedH==1: recoidx = 3
      #elif nBoostedH==0: recoidx = 4
      recoidx = 4-nBoostedH
    elif nHiggs==2:
      #if nBoostedH==2: recoidx = 5
      #elif nBoostedH==1: recoidx = 6
      #elif nBoostedH==0: recoidx = 7
      recoidx = 7-nBoostedH
    elif nHiggs==1:
      #if nBoostedH==1: recoidx = 8
      #elif nBoostedH==0: recoidx = 9
      recoidx = 9-nBoostedH
    # return reco_idx, fitted_mass, Higgs 1/2/3, Jets 1/2/3/4/5/6, TauBoosted H Idx, TauResolved H Idx, FinalState of Htautau
    return recoidx,m_fit,h[0],h[1],h[2],j[0],j[1],j[2],j[3],j[4],j[5],TauIsBoosted,TauIsResolved,TauFinalState

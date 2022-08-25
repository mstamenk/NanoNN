import ROOT




def fitMass(m1,s1,m2,s2, m3,s3):
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)
    varDummy = ROOT.RooRealVar("varDummy","",1.0)
    dataDummy = ROOT.RooDataSet("dataDummy", "", ROOT.RooArgSet(varDummy))
    pdfDummy = ROOT.RooGaussian("dataDummy","" ,varDummy, ROOT.RooFit.RooConst(1.0),ROOT.RooFit.RooConst(1.0))

    deps = ROOT.RooArgSet()

    mH1 = ROOT.RooRealVar("mH1","", m1)
    mH2 = ROOT.RooRealVar("mH2","", m2)
    mH3 = ROOT.RooRealVar("mH3","", m3)

    deps.add(mH1)
    deps.add(mH2)
    deps.add(mH3)

    mFit = ROOT.RooRealVar("mFit","", 125, 0, 2000)

    deps.add(mFit)

    mH1.setConstant(ROOT.kTRUE)
    mH2.setConstant(ROOT.kTRUE)
    mH3.setConstant(ROOT.kTRUE)

    #likelihood = ROOT.RooFormulaVar("LLH", "( (mH1 - mFit)**2 + (mH2 - mFit)**2 + (mH3 - mFit)**2 )", deps)

    constr_mH1 = ROOT.RooGaussian("constr_mH1", "constr_mH1", mFit, ROOT.RooFit.RooConst(m1), ROOT.RooFit.RooConst(s1))
    constr_mH2 = ROOT.RooGaussian("constr_mH2", "constr_mH2", mFit, ROOT.RooFit.RooConst(m2), ROOT.RooFit.RooConst(s2))
    constr_mH3 = ROOT.RooGaussian("constr_mH3", "constr_mH3", mFit, ROOT.RooFit.RooConst(m3), ROOT.RooFit.RooConst(s3))

    constraints = ROOT.RooArgSet()
    constraints.add(constr_mH1)
    constraints.add(constr_mH2)
    constraints.add(constr_mH3)


    nll = pdfDummy.createNLL(dataDummy,ROOT.RooFit.ExternalConstraints(constraints))
    m = ROOT.RooMinuit(nll)

    m.setVerbose(ROOT.kTRUE)
    m.setPrintLevel(-1)

    m.migrad()
    #m.hesse()

    return nll.getVal(),mFit.getVal()




if __name__ == "__main__":

    m1 = 115.
    s1 = 15.
    m2 = 115.
    s2 = 15.
    m3 = 115.0
    s3 = 15.0
    
    # dummy variables to make the NLL to build, do nothing in the code
    nll, mass = fitMass(m1,s1,m2,s2,m3,s3)
    print(nll,mass)












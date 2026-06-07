"""Verify the workflow's bomb: is Law-2's 2a 'head-headroom' (a head FIT to session_E beats
the frozen session_T head by ~+2.5pp => drift localizes to the head) REAL, or a fit-on-test
LDA overfitting artifact (64-d LDA on ~288 trials)? Use clean cached features
(source_train_features_s0.npz). Compare:
  source       = LDA(train_feat) -> eval_feat                 [frozen T head]
  fit-on-test  = LDA(eval_feat)  -> eval_feat (same data)     [overfit, the inflated number]
  CV head      = 5-fold CV LDA on eval_feat                   [honest best head on E]
headroom_honest = CV - source. If ~0/negative => Law-2 on 2a is an overfitting mirage.
Also NCC (nearest-centroid) oracle as a low-variance cross-check.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
d=np.load("intentflow/offline/results/source_train_features_s0.npz",allow_pickle=True)
def ncc(Xtr,ytr,Xte):
    cls=np.unique(ytr);M=np.stack([Xtr[ytr==c].mean(0) for c in cls])
    return cls[np.argmin(((Xte[:,None,:]-M[None])**2).sum(-1),1)]
print(f"{'S':>3} {'source':>7} {'fit-on-test':>11} {'CV-head':>8} {'honest Δ':>9} {'NCC-CV Δ':>9}")
src=[];fot=[];cv=[];nccd=[]
for s in range(1,10):
    Ftr=d[f"train_feat_{s}"];ytr=d[f"train_label_{s}"];Fte=d[f"eval_feat_{s}"];yte=d[f"eval_label_{s}"]
    a_src=(LDA(solver="lsqr",shrinkage="auto").fit(Ftr,ytr).predict(Fte)==yte).mean()*100
    a_fot=(LDA(solver="lsqr",shrinkage="auto").fit(Fte,yte).predict(Fte)==yte).mean()*100   # fit-on-test
    # 5-fold CV head on eval
    skf=StratifiedKFold(5,shuffle=True,random_state=0);pr=np.zeros_like(yte)
    for tr,va in skf.split(Fte,yte):
        pr[va]=LDA(solver="lsqr",shrinkage="auto").fit(Fte[tr],yte[tr]).predict(Fte[va])
    a_cv=(pr==yte).mean()*100
    # NCC CV (low-variance oracle)
    prn=np.zeros_like(yte)
    for tr,va in skf.split(Fte,yte): prn[va]=ncc(Fte[tr],yte[tr],Fte[va])
    a_ncccv=(prn==yte).mean()*100
    a_nccsrc=(ncc(Ftr,ytr,Fte)==yte).mean()*100
    src.append(a_src);fot.append(a_fot);cv.append(a_cv);nccd.append(a_ncccv-a_nccsrc)
    print(f"S{s:>2} {a_src:>7.1f} {a_fot:>11.1f} {a_cv:>8.1f} {a_cv-a_src:>+9.1f} {a_ncccv-a_nccsrc:>+9.1f}")
print(f"{'MEAN':>3} {np.mean(src):>7.1f} {np.mean(fot):>11.1f} {np.mean(cv):>8.1f} {np.mean(cv)-np.mean(src):>+9.1f} {np.mean(nccd):>+9.1f}")
print(f"\nfit-on-test headroom (the OLD claim): {np.mean(fot)-np.mean(src):+.1f}pp")
print(f"HONEST CV head-headroom:              {np.mean(cv)-np.mean(src):+.1f}pp   <== if ~0/neg, Law-2 on 2a is an artifact")
print(f"NCC CV head-headroom (low-variance):  {np.mean(nccd):+.1f}pp")

"""[9] Drift-Direction Counterfactual Augmentation — decisive test (2a, clean cached feats).
Estimate a LABEL-FREE drift operator g mapping source-train feature distribution -> eval
distribution (CORAL transport: whiten by train cov, recolor by eval cov, shift centroid),
apply g to LABELED source-train features (labels preserved = real), recalibrate ONLY the
head (LDA) on g(train), predict eval. No eval labels used to build g.
Baseline discipline: source-head LDA(train)->eval must ~82.7 (2a source). Report delta vs
source and compare to DA-DC 87.2. Claim of [9]: delta>0 label-free.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import LedoitWolf
d=np.load("intentflow/offline/results/source_train_features_s0.npz",allow_pickle=True)
def sqrtm_inv(C,inv):
    w,V=np.linalg.eigh(C);w=np.clip(w,1e-8,None)
    return (V*((w**-0.5) if inv else (w**0.5)))@V.T
print(f"{'subj':>4} {'src_head':>9} {'mean_shift':>11} {'CORAL':>8} {'Δmean':>7} {'ΔCORAL':>8}")
rows=[]
for s in range(1,10):
    Ftr=d[f"train_feat_{s}"].astype(np.float64);ytr=d[f"train_label_{s}"]
    Fte=d[f"eval_feat_{s}"].astype(np.float64);yte=d[f"eval_label_{s}"]
    # source head baseline
    acc_src=(LDA(solver="lsqr",shrinkage="auto").fit(Ftr,ytr).predict(Fte)==yte).mean()*100
    mu_tr,mu_te=Ftr.mean(0),Fte.mean(0)
    # variant 1: mean-shift only
    Ftr_m=Ftr-mu_tr+mu_te
    acc_m=(LDA(solver="lsqr",shrinkage="auto").fit(Ftr_m,ytr).predict(Fte)==yte).mean()*100
    # variant 2: full CORAL transport (mean+cov)
    Ctr=LedoitWolf().fit(Ftr).covariance_;Cte=LedoitWolf().fit(Fte).covariance_
    A=sqrtm_inv(Cte,False)@sqrtm_inv(Ctr,True)
    Ftr_c=(Ftr-mu_tr)@A.T+mu_te
    acc_c=(LDA(solver="lsqr",shrinkage="auto").fit(Ftr_c,ytr).predict(Fte)==yte).mean()*100
    rows.append((acc_src,acc_m,acc_c))
    print(f"S{s:>3} {acc_src:>9.1f} {acc_m:>11.1f} {acc_c:>8.1f} {acc_m-acc_src:>+7.1f} {acc_c-acc_src:>+8.1f}")
m=lambda j:np.mean([r[j] for r in rows])
print(f"\nMEAN src_head={m(0):.1f} (SANITY ~82.7) | mean_shift={m(1):.1f} ({m(1)-m(0):+.1f}) | CORAL={m(2):.1f} ({m(2)-m(0):+.1f})")
print(f"reference: 2a source=82.7, DA-DC=87.2")
nm=sum(1 for r in rows if r[1]>r[0]);nc=sum(1 for r in rows if r[2]>r[0])
print(f"helps over source: mean_shift {nm}/9, CORAL {nc}/9")
print("\n[9] HOLDS if a label-free transported-head beats source on most subjects (delta>0). KILL if ~0/negative")
print("(=F2 re-enters: marginal drift operator can't fix class-conditional head drift).")

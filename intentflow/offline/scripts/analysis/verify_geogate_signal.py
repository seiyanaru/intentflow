"""VERIFY GeoGate's load-bearing claim (==A==) with BASELINE DISCIPLINE, on CLEAN
no-EA source features (source_train_features_s0.npz, 2a).
Claim: on the deployed head's CONFIDENT subset, source-class Mahalanobis distance still
detects errors (AUROC > 0.5) while margin/confidence is DEAD (AUROC ~ 0.5); and
Mahalanobis abstention reduces error.
Baseline-discipline sanity check FIRST: the deployed-head (LDA-on-train) eval accuracy
must land near the known 2a source = 82.7%. If not, the proxy is invalid -> abort trust.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
F="intentflow/offline/results/source_train_features_s0.npz"
d=np.load(F,allow_pickle=True)
def maha_setup(Xtr,ytr):
    mus={c:Xtr[ytr==c].mean(0) for c in np.unique(ytr)}
    res=np.concatenate([Xtr[ytr==c]-mus[c] for c in np.unique(ytr)],0)  # within-class residuals
    cov=LedoitWolf().fit(res); P=np.linalg.pinv(cov.covariance_)
    classes=sorted(mus)
    M=np.stack([mus[c] for c in classes])
    return M,P,classes
def dmin(X,M,P):
    # min over classes of Mahalanobis^2 distance
    out=[]
    for c in range(M.shape[0]):
        diff=X-M[c]; out.append(np.einsum('ij,jk,ik->i',diff,P,diff))
    return np.min(np.stack(out,1),1)

print(f"{'subj':>4} {'LDAacc':>7} {'Maha_AUROC_all':>14} {'Maha_AUROC_conf':>15} {'margin_AUROC_conf':>17} {'absto80_err':>12} {'full_err':>9} {'n_wrong_conf':>12}")
rows=[]
for s in range(1,10):
    Xtr=d[f"train_feat_{s}"];ytr=d[f"train_label_{s}"];Xte=d[f"eval_feat_{s}"];yte=d[f"eval_label_{s}"]
    # deployed head proxy = shrinkage LDA on session_T features
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xtr,ytr)
    proba=lda.predict_proba(Xte);pred=lda.classes_[proba.argmax(1)]
    acc=(pred==yte).mean()*100
    wrong=(pred!=yte).astype(int)
    ms=np.sort(proba,1);margin=ms[:,-1]-ms[:,-2]
    # source manifold Mahalanobis
    M,P,_=maha_setup(Xtr,ytr); dm=dmin(Xte,M,P)
    # AUROC: detect WRONG. Maha: higher d -> wrong. margin: lower margin -> wrong (use -margin).
    au_all=roc_auc_score(wrong,dm) if wrong.sum() not in (0,len(wrong)) else np.nan
    conf=margin>=np.median(margin)  # confident subset = top-50% margin
    wc=wrong[conf]
    au_maha_c=roc_auc_score(wc,dm[conf]) if wc.sum() not in (0,len(wc)) else np.nan
    au_marg_c=roc_auc_score(wc,-margin[conf]) if wc.sum() not in (0,len(wc)) else np.nan
    # abstain 20% highest-Maha (coverage 80%): error on kept vs full
    keep=dm<=np.quantile(dm,0.80)
    err_kept=(wrong[keep]).mean()*100; err_full=wrong.mean()*100
    rows.append((s,acc,au_all,au_maha_c,au_marg_c,err_kept,err_full,int(wc.sum())))
    print(f"S{s:>3} {acc:>7.1f} {au_all:>14.3f} {au_maha_c:>15.3f} {au_marg_c:>17.3f} {err_kept:>12.1f} {err_full:>9.1f} {int(wc.sum()):>12}")
m=lambda j:np.nanmean([r[j] for r in rows])
print(f"\nMEAN LDAacc={m(1):.1f} (SANITY: must ~82.7 = 2a source; else proxy invalid)")
print(f"MEAN Maha_AUROC all={m(2):.3f} | conf-subset Maha={m(3):.3f} vs margin={m(4):.3f}")
print(f"MEAN abstain@80% err={m(5):.1f} vs full err={m(6):.1f}  (drop = {m(6)-m(5):+.1f}pp)")
nM=sum(1 for r in rows if r[3]>0.5); print(f"\nconf-subset Maha_AUROC>0.5 on {nM}/9 subjects")
nA=sum(1 for r in rows if r[5]<r[6]); print(f"abstain reduces error on {nA}/9 subjects")
print("\nVERDICT: PASS if LDAacc~82.7 AND conf-Maha>>conf-margin(~0.5) on most subjects AND abstain reduces error.")

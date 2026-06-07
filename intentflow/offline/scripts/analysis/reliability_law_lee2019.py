"""Scale-up of the reliability-monitor law on Lee2019 (54 subj x 2 sessions, cross-session).
Per subject: train sess0 -> test sess1 with an EA-Riemannian tangent + shrinkage-LDA decoder.
Measure cross-session accuracy and the LABEL-FREE clusterability of test-session tangent
features. Test the law: clusterability predicts cross-session reliability (spearman), at n=54.
Downloads to workspace-local2 (root partition is full). Incremental save.
"""
import os
os.environ["MNE_DATA"]="/home/islabshi/workspace-local2/mne_data"
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, mne
mne.set_config("MNE_DATA", os.environ["MNE_DATA"])
from scipy.linalg import eigh, logm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
prm=LeftRightImagery(resample=250, fmin=8, fmax=30)
ds=Lee2019_MI()
def covs(X):  # X (n,C,T) already 8-30Hz -> per-trial covariance
    return np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]
def invsqrt(M):w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def tangent(C,P):
    c=C.shape[0];iu=np.triu_indices(c);s=np.sqrt(2)*np.ones((c,c));np.fill_diagonal(s,1.)
    A=P@C@P; w,V=eigh(A);w=np.clip(w,1e-10,None);L=(V*np.log(w))@V.T
    return (L*s)[iu]
def feats(X):  # EA (unsupervised, this session) + tangent
    Cs=covs(X.astype(np.float64));P=invsqrt(Cs.mean(0))
    return np.array([tangent(C,P) for C in Cs])
OUT="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260606_reliability_law_lee2019.json"
res=[]
for s in range(1,55):
    try:
        X,y,meta=prm.get_data(dataset=ds,subjects=[s])
        ses=meta['session'].values; y=(y=='right_hand').astype(int)
        tr=ses=='0'; te=ses=='1'
        Ftr=feats(X[tr]);Fte=feats(X[te])
        lda=LDA(solver="lsqr",shrinkage="auto").fit(Ftr,y[tr])
        acc=(lda.predict(Fte)==y[te]).mean()*100
        Z=(Fte-Fte.mean(0))/(Fte.std(0)+1e-8)
        clus=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))
        drift=np.linalg.norm(Ftr.mean(0)-Fte.mean(0))
        res.append(dict(subj=s,acc=float(acc),clus=float(clus),drift=float(drift)))
        print(f"S{s:>2}: acc={acc:5.1f} clus={clus:.3f} drift={drift:.1f}",flush=True)
        json.dump(res,open(OUT,'w'))
    except Exception as e:
        print(f"S{s}: FAIL {str(e)[:80]}",flush=True)
acc=[r['acc'] for r in res];clus=[r['clus'] for r in res];drift=[r['drift'] for r in res]
print(f"\n=== Lee2019 cross-session, n={len(res)} ===")
print(f"mean cross-session acc={np.mean(acc):.1f}")
print(f"corr(clusterability, acc) = {spearmanr(clus,acc).correlation:+.3f}   [reliability law at scale]")
print(f"corr(drift, acc)          = {spearmanr(drift,acc).correlation:+.3f}")
print(f"saved {OUT}")
print("\nLAW HARDENED if corr(clusterability, acc) ~ +0.8 at n=54 (matches 2a +0.78 / 2b +0.82).")

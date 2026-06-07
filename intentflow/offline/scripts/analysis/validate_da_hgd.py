"""HGD safety/robustness check for DA-L1 (source + 0.3*D), cross-subject LOSO.
HGD has only source TCFormer preds (no EA/shrink) and a very strong source (~94),
so this tests whether a weak diverse D harms a strong base. Load HGD once,
compute per-trial covariances, LOSO Riemannian (EA per subject + tangent + LDA).
Run with intentflow conda env.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from utils.load_hgd import load_hgd

SRC = glob.glob("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/phaseC_hgd_firstpass_*/source_only")[0]
prep = dict(sfreq=250, low_cut=4, high_cut=None, start=0.0, stop=4.0, z_scale=True,
            batch_size=48, num_workers=0)
SUBS = list(range(1, 15))
def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
b,a=butter(4,[8/125.,30/125.],btype="band")
def covs(X):  # X (n,C,T) -> per-trial cov, bandpassed
    Xf=filtfilt(b,a,X,axis=-1)
    return np.einsum('nct,ndt->ncd',Xf,Xf)/Xf.shape[-1]
def invsqrtm(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*(w**-0.5))@V.T
def logm_spd(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*np.log(w))@V.T
def ea(C): R=C.mean(0); P=invsqrtm(R); return np.einsum('ij,njk,kl->nil',P,C,P)
def tangent(C):
    c=C.shape[1]; iu=np.triu_indices(c); s=np.sqrt(2)*np.ones((c,c)); np.fill_diagonal(s,1.)
    return np.array([(logm_spd(C[i])*s)[iu] for i in range(C.shape[0])])

print("loading HGD (once)...", flush=True)
ds = load_hgd(subject_ids=SUBS, preprocessing_dict=prep)
by_run = ds.split("run")
train_runs = by_run["0train"].split("subject")   # for training D
test_runs = by_run["1test"].split("subject")      # eval (matches TCFormer source)
# precompute per-subject covariances+labels for train(0train) and test(1test)
trC={}; trY={}; teC={}; teY={}
for s in SUBS:
    ks=str(s)
    if ks in train_runs.keys():
        d=train_runs[ks]; X=np.stack([d[i][0] for i in range(len(d))]).astype(np.float64); y=np.array([d[i][1] for i in range(len(d))])
        trC[s]=ea(covs(X)); trY[s]=y
    if ks in test_runs.keys():
        d=test_runs[ks]; X=np.stack([d[i][0] for i in range(len(d))]).astype(np.float64); y=np.array([d[i][1] for i in range(len(d))])
        teC[s]=ea(covs(X)); teY[s]=y
    print(f"  cov done subj {s}", flush=True)

W=0.3; rows=[]
for s in SUBS:
    lf=f"{SRC}/logits_s{s}_tcformer_otta.npy"
    if not os.path.exists(lf) or s not in teC: continue
    src=sm(np.load(lf))
    others=[o for o in SUBS if o!=s and o in trC]
    Xtr=np.concatenate([tangent(trC[o]) for o in others]); ytr=np.concatenate([trY[o] for o in others])
    Xte=tangent(teC[s]); yte=teY[s]
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Xtr,ytr)
    pr=lda.predict_proba(Xte); ncl=src.shape[1]; Dd=np.zeros((len(yte),ncl)); Dd[:,lda.classes_]=pr
    n=min(len(yte),len(src)); yc=yte[:n]; sc=src[:n]; Dc=Dd[:n]
    sa=(sc.argmax(1)==yc).mean()*100; da=((sc+W*Dc).argmax(1)==yc).mean()*100; dalone=(Dc.argmax(1)==yc).mean()*100
    rows.append((s,sa,da,dalone))
    print(f"S{s}: src {sa:.2f} src+0.3D {da:.2f} (Δ{da-sa:+.2f}) D-alone {dalone:.2f} n={n}", flush=True)
M=lambda k: float(np.mean([r[k] for r in rows]))
hsc=sum(1 for r in rows if r[2]<r[1]-1e-9)
print(f"\n=== HGD (cross-subject LOSO, frozen w=0.3) ===")
print(f"source {M(1):.2f} | source+0.3D {M(2):.2f} ({M(2)-M(1):+.2f}) | D-alone {M(3):.2f} | HSC vs src {hsc}/{len(rows)}")

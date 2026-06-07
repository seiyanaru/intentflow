"""Validate DA-L1 (diverse-agreement ensemble) on bcic2b with FROZEN w=0.3.
Build diverse D (EA-Riemannian tangent + LDA, 3ch) on 2b, assemble TCFormer
source/full/shrink probs, and check gain + per-subject safety (esp S5 where EA
was catastrophic). Run with intentflow conda env.
"""
import os, sys, glob, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # intentflow/offline
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2b import BCICIV2b

prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=3.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True, data_path=None)
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results"
SRC_DIR = glob.glob(f"{RES}/TCFormer_bcic2b_seed-0_aug-True_GPU0_*")[0]
def pairdir(prefix, sid):
    pair = {1:"s12",2:"s12",3:"s34",4:"s34",5:"s56",6:"s56",7:"s789",8:"s789",9:"s789"}[sid]
    return glob.glob(f"{RES}/{prefix}_bcic2b_{pair}_seed0_*")[0]
def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def tcf(sid):
    src=sm(np.load(f"{SRC_DIR}/logits_s{sid}_TCFormer.npy"))
    full=sm(np.load(f"{pairdir('ea_aware_tcformer',sid)}/logits_s{sid}_TCFormer.npy"))
    shr=sm(np.load(f"{pairdir('ea_sh01_tcformer',sid)}/logits_s{sid}_TCFormer.npy"))
    lab=np.load(f"{SRC_DIR}/features_s{sid}_TCFormer.npz",allow_pickle=True)["labels"]
    return src,full,shr,lab

b,a=butter(4,[8/125.,30/125.],btype="band")
def bp(X): return filtfilt(b,a,X,axis=-1).copy()
def cov(X): return np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]
def invsqrtm(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*(w**-0.5))@V.T
def logm_spd(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*np.log(w))@V.T
def ea(C): R=C.mean(0); P=invsqrtm(R); return np.einsum('ij,njk,kl->nil',P,C,P)
def tangent(C):
    c=C.shape[1]; iu=np.triu_indices(c); s=np.sqrt(2)*np.ones((c,c)); np.fill_diagonal(s,1.)
    return np.array([(logm_spd(C[i])*s)[iu] for i in range(C.shape[0])])
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    y=np.array([int(ds[i][1]) for i in range(len(ds))]); return X,y

W=0.3; SAFE=np.array([0.65,0.25,0.10])
rows=[]
for sid in range(1,10):
    src,full,shr,lab=tcf(sid)
    dm=BCICIV2b(prep,sid); dm.setup()
    Xtr,ytr=getxy(dm.train_dataset); Xte,yte=getxy(dm.test_dataset)
    Dtr=tangent(ea(cov(bp(Xtr)))); Dte=tangent(ea(cov(bp(Xte))))
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Dtr,ytr)
    pr=lda.predict_proba(Dte); Dd=np.zeros((len(yte),2)); Dd[:,lda.classes_]=pr
    if len(yte)!=len(lab) or (yte!=lab).mean()>0.01:
        print(f"S{sid}: ALIGN FAIL nE={len(yte)} nTCF={len(lab)} lab-match={(yte==lab).mean() if len(yte)==len(lab) else 'NA'}");
        # try to proceed if lengths match
    y=lab; n=min(len(y),len(Dd)); y=y[:n];src=src[:n];full=full[:n];shr=shr[:n];Dd=Dd[:n]
    sa=(src.argmax(1)==y).mean()*100
    blend=SAFE[0]*src+SAFE[1]*full+SAFE[2]*shr
    ba=(blend.argmax(1)==y).mean()*100
    da_src=((src+W*Dd).argmax(1)==y).mean()*100
    da_bl=((blend+W*Dd).argmax(1)==y).mean()*100
    dacc=(Dd.argmax(1)==y).mean()*100
    rows.append((sid,sa,ba,da_src,da_bl,dacc))
    print(f"S{sid}: src {sa:.2f} safeblend {ba:.2f} src+0.3D {da_src:.2f} blend+0.3D {da_bl:.2f} | D-alone {dacc:.2f} | n={n}",flush=True)
M=lambda k:float(np.mean([r[k] for r in rows]))
print(f"\n=== bcic2b (frozen w=0.3) ===")
print(f"source     {M(1):.2f}")
print(f"safe-blend {M(2):.2f} ({M(2)-M(1):+.2f})")
print(f"source+0.3D {M(3):.2f} ({M(3)-M(1):+.2f} vs src)  HSC vs src {sum(1 for r in rows if r[3]<r[1]-1e-9)}/9")
print(f"blend +0.3D {M(4):.2f} ({M(4)-M(1):+.2f} vs src)  HSC vs src {sum(1 for r in rows if r[4]<r[1]-1e-9)}/9")
print(f"D-alone(Riemann) {M(5):.2f}")
s5=[r for r in rows if r[0]==5][0]
print(f"\nS5 (EA-catastrophe subject): src {s5[1]:.2f} | src+0.3D {s5[3]:.2f} | blend+0.3D {s5[4]:.2f} | D-alone {s5[5]:.2f}")

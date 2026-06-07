"""D3 = spectral band-power + LDA (NON-covariance family => orthogonal to D1
(Riemannian) and D2 (CSP)). Per-channel log band-power in 6 bands -> LDA.
Build on 2a, save preds. Run with intentflow conda env.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2a import BCICIV2a

DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=DATA, eval_label_path=DATA+"labels")
BANDS = [(4,8),(8,12),(12,16),(16,20),(20,26),(26,32)]
filts = [butter(4, [lo/125., hi/125.], btype="band") for lo,hi in BANDS]
def feat(X):  # X (n,C,T) -> (n, C*nbands) log band-power
    outs=[]
    for (b,a) in filts:
        Xf = filtfilt(b,a,X,axis=-1)
        outs.append(np.log(np.var(Xf,axis=-1)+1e-8))   # (n,C)
    return np.concatenate(outs,1)
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    return X, np.array([int(ds[i][1]) for i in range(len(ds))])
P=[]
for sid in range(1,10):
    dm=BCICIV2a(prep,sid); dm.setup()
    Xtr,ytr=getxy(dm.train_dataset); Xte,yte=getxy(dm.test_dataset)
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(feat(Xtr),ytr)
    pr=lda.predict_proba(feat(Xte)); pp=np.zeros((len(yte),4)); pp[:,lda.classes_]=pr
    P.append(pp); print(f"S{sid}: D3(spectral) acc={(pp.argmax(1)==yte).mean()*100:.2f}",flush=True)
np.savez("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_d3_spectral_preds.npz",
         probs=np.array(P), subjects=np.arange(1,10))
print("saved D3")

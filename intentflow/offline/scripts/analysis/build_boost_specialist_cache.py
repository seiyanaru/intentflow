"""Cache EA-aligned Riemannian tangent features for session_T (train) and session_E
(eval) per subject on bcic2a, so boosting-specialist experiments can run fast in numpy
without reloading raw EEG each time.

Same EA-Riemann pipeline as build_diverse_riemann.py (band 8-30Hz, EA whitening, SPD log
tangent). Train tangent uses train-EA reference; eval tangent uses eval-EA reference
(transductive EA on each session independently -- identical to the saved D1 protocol).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # -> intentflow/offline
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from datamodules.bcic4_2a import BCICIV2a

DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
LAB = DATA + "labels"
OUT = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_boost_tangent_cache.npz"
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=DATA, eval_label_path=LAB)

b, a = butter(4, [8/125., 30/125.], btype="band")
def bp(X): return filtfilt(b, a, X, axis=-1).copy()
def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M):
    w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * (w ** -0.5)) @ V.T
def logm_spd(M):
    w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * np.log(w)) @ V.T
def ea_align(C):
    R = C.mean(0); P = invsqrtm(R); return np.einsum('ij,njk,kl->nil', P, C, P)
def tangent(C):
    c = C.shape[1]; iu = np.triu_indices(c)
    sc = np.sqrt(2) * np.ones((c, c)); np.fill_diagonal(sc, 1.0)
    return np.array([(logm_spd(C[i]) * sc)[iu] for i in range(C.shape[0])])

def get_xy(ds):
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    y = np.array([int(ds[i][1]) for i in range(len(ds))])
    return X, y

Ttr_all, ytr_all, Tte_all, yte_all = [], [], [], []
ntr = []
for sid in range(1, 10):
    dm = BCICIV2a(prep, sid); dm.setup()
    Xtr, ytr = get_xy(dm.train_dataset)
    Xte, yte = get_xy(dm.test_dataset)
    Ttr = tangent(ea_align(cov(bp(Xtr))))
    Tte = tangent(ea_align(cov(bp(Xte))))
    Ttr_all.append(Ttr); ytr_all.append(ytr); Tte_all.append(Tte); yte_all.append(yte)
    ntr.append(len(ytr))
    print(f"S{sid}: train={len(ytr)} eval={len(yte)} tangent_dim={Ttr.shape[1]}", flush=True)

# train sizes differ slightly -> save as object arrays
np.savez(OUT,
         Ttr=np.array(Ttr_all, dtype=object), ytr=np.array(ytr_all, dtype=object),
         Tte=np.array(Tte_all, dtype=object), yte=np.array(yte_all, dtype=object),
         subjects=np.arange(1, 10))
print("saved ->", OUT)

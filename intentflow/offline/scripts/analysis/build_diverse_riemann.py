"""Build a DIVERSE second model D (EA-aligned Riemannian tangent + shrinkage-LDA)
on bcic2a, to test the diversity hypothesis: are D's errors decorrelated from the
TCFormer (T) backbone's, and is T-vs-D AGREEMENT a label-grade trust signal that
could break the confident-error label-lock?

D is covariance/manifold-based = maximally different inductive bias from TCFormer's
temporal-conv-transformer. Train on session_T, predict session_E, in the SAME
trial order as the saved TCFormer logits (verified by label match downstream).
Run with the intentflow conda env (needs mne/braindecode).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))  # -> intentflow/offline
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2a import BCICIV2a

DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
LAB = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels"
OUT = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz"
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

P_all, Y_all, accs = [], [], []
for sid in range(1, 10):
    dm = BCICIV2a(prep, sid); dm.setup()
    Xtr, ytr = get_xy(dm.train_dataset)
    Xte, yte = get_xy(dm.test_dataset)
    Ttr = tangent(ea_align(cov(bp(Xtr))))
    Tte = tangent(ea_align(cov(bp(Xte))))
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Ttr, ytr)
    proba = lda.predict_proba(Tte)
    P = np.zeros((len(yte), 4))
    P[:, lda.classes_] = proba                  # align columns to class idx 0..3
    acc = (P.argmax(1) == yte).mean() * 100
    P_all.append(P); Y_all.append(yte); accs.append(acc)
    print(f"S{sid}: D(Riemann) acc={acc:5.2f}  n_test={len(yte)}  train={len(ytr)}", flush=True)

np.savez(OUT, probs=np.array(P_all), labels=np.array(Y_all), subjects=np.arange(1, 10))
print(f"\nD mean acc = {np.mean(accs):.2f}  (TCFormer source ref ~82.7)")
print(f"saved -> {OUT}")

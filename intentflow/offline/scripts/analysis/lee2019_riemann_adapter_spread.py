"""Fair test (GPT hole #1): does a REAL different-family adapter have benefit/HARM SPREAD on the
well-powered Lee2019 (n=54)? The cheap self-refit head was net-zero; the Riemann-tangent-LDA is a
genuinely different inductive bias (covariance manifold, no deep net, no retraining) = a true DA-DC
component. Per held-out subject, within-subject session0(labeled)->session1, EA-aligned tangent LDA.
Source = the deployed E4 decoder LDA(F0,y0)->F1 (net penultimate feats, from the cached dump).
Outputs per-session: acc_source, acc_riemann, acc_blend(0.5/0.5), Δ_riemann, Δ_blend, and the SPREAD
(helped/harmed counts, worst). Saves per-session Δ + Riemann probs for the fair gate re-test.
CPU; reloads Lee2019 raw via MOABB (cached). No GPU, no training.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
DUMP = np.load(f"{RES}/260608_lee2019_pertrial_tcformer.npz")

prm = LeftRightImagery(resample=250, fmin=1, fmax=45); ds = Lee2019_MI()
b, a = butter(5, [8/125, 30/125], btype="band")
def bp(X): return filtfilt(b, a, X, axis=-1).copy()
def cov(X): C = np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]; return C
def invsqrtm(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * (w**-0.5)) @ V.T
def logm_spd(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * np.log(w)) @ V.T
def ea_align(C): R = C.mean(0); P = invsqrtm(R); return np.einsum('ij,njk,kl->nil', P, C, P)
def tangent(C):
    d = C.shape[-1]; iu = np.triu_indices(d)
    sc = np.sqrt(2) * np.ones((d, d)); sc[np.diag_indices(d)] = 1; sc = sc[iu]
    return np.array([(logm_spd(C[i]) * sc.reshape(-1) if False else (logm_spd(C[i]))[iu] * sc) for i in range(len(C))])

def src_probs(s):  # deployed E4 source decoder: LDA on net penultimate feats F0 -> F1
    F0 = DUMP[f"F0_{s}"].astype(np.float64); y0 = DUMP[f"y0_{s}"]; F1 = DUMP[f"F1_{s}"].astype(np.float64)
    lda = LDA(solver="lsqr", shrinkage="auto").fit(F0, y0)
    return lda.predict_proba(F1), DUMP[f"y1_{s}"]

subs = sorted({int(k.split("_")[1]) for k in DUMP.files if k.startswith("F1_")})
_cache = {}
def load_raw(sub):
    if sub in _cache: return _cache[sub]
    X, y, meta = prm.get_data(dataset=ds, subjects=[sub]); y = (y == 'right_hand').astype(int)
    ses = meta['session'].values
    out = {sv: (X[ses == sv].astype(np.float64), y[ses == sv]) for sv in ['0', '1']}
    _cache[sub] = out; return out

rows = []; npz = {}
print(f"{'S':>3} {'src':>6} {'Riem':>6} {'blend':>6} {'dR':>6} {'dB':>6}")
for s in subs:
    try:
        Ps, y1 = src_probs(s)
        d = load_raw(s); X0, y0r = d['0']; X1, y1r = d['1']
        if len(y1r) != len(y1):  # order/length mismatch guard
            print(f"S{s}: LEN MISMATCH dump{len(y1)} raw{len(y1r)} -- skip"); continue
        T0 = tangent(ea_align(cov(bp(X0)))); T1 = tangent(ea_align(cov(bp(X1))))
        lda = LDA(solver="lsqr", shrinkage="auto").fit(T0, y0r)
        Pr = lda.predict_proba(T1)
        cls = lda.classes_; Pr_full = np.zeros((len(T1), 2)); Pr_full[:, cls] = Pr
        acc_s = (Ps.argmax(1) == y1).mean() * 100
        acc_r = (Pr_full.argmax(1) == y1r).mean() * 100
        Pb = 0.5 * Ps + 0.5 * Pr_full
        acc_b = (Pb.argmax(1) == y1).mean() * 100
        rows.append(dict(s=s, acc_s=acc_s, acc_r=acc_r, acc_b=acc_b, dR=acc_r - acc_s, dB=acc_b - acc_s))
        npz[f"Pr_{s}"] = Pr_full.astype(np.float32); npz[f"Ps_{s}"] = Ps.astype(np.float32); npz[f"y_{s}"] = y1
        print(f"S{s:>2} {acc_s:>6.1f} {acc_r:>6.1f} {acc_b:>6.1f} {acc_r-acc_s:>+6.1f} {acc_b-acc_s:>+6.1f}", flush=True)
    except Exception as e:
        print(f"S{s}: FAIL {str(e)[:60]}")

dR = np.array([r["dR"] for r in rows]); dB = np.array([r["dB"] for r in rows])
print("\n===== SPREAD on Lee2019 n=%d (is there benefit/harm to gate?) =====" % len(rows))
for nm, dd in [("Riemann-only", dR), ("blend(0.5net+0.5Riem)", dB)]:
    print(f"  {nm:>22}: meanΔ={dd.mean():+.2f} std={dd.std():.2f} helped(>1)={int((dd>1).sum())} "
          f"harmed(<-1)={int((dd<-1).sum())} worst={dd.min():+.1f} best={dd.max():+.1f}")
json.dump(rows, open(f"{RES}/260609_lee2019_riemann_spread.json", "w"), indent=2)
np.savez_compressed(f"{RES}/260609_lee2019_adapter_probs.npz", **npz)
print(f"\nsaved {RES}/260609_lee2019_riemann_spread.json + adapter_probs.npz")
print("READ: if Δ has real spread (some helped, some harmed), there IS something to gate -> fair gate")
print("re-test (ATTA-LCB probe-veto + overrule-audit + LEEP/LogME) against THIS real adapter Δ is next.")

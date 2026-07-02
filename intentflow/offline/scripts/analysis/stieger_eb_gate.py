"""Stieger2021 (62 subj x up to 11 longitudinal sessions): well-powered MIXED-adapter test of the
EB-pooled selective-adaptation gate. CHECKPOINTED + RESUMABLE (saves cases.npz per subject; skips
already-done subjects on restart) so a download/kill mid-run loses nothing. Runs the gate on whatever
cases are accumulated (partial OK).

Per subject: session-1 = labeled calib -> EA-whiten + Riemann tangent-LDA = source decoder.
Per later session j: source (whiten by SESSION-1 ref R1) vs adapt (EA recenter to own ref Rj).
per-(subj,sess) benefit/harm = adapt_acc - source_acc. Then EB-pooled gate (leak-free first-k probe).
CPU. Re-run anytime to resume; run stieger_gate_eval.py for gate-only on cases.npz.
"""
import os, sys, warnings, json, glob, re, gc
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from moabb.datasets import Stieger2021
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
CASES = f"{RES}/260609_stieger_cases.npz"
EEG60 = ['AF3','AF4','C1','C2','C3','C4','C5','C6','CP1','CP2','CP3','CP4','CP5','CP6','CPz','Cz','F1','F2','F3','F4','F5','F6','F7','F8','FC1','FC2','FC3','FC4','FC5','FC6','FCz','FT7','FT8','Fp1','Fp2','Fpz','Fz','O1','O2','Oz','P1','P2','P3','P4','P5','P6','P7','P8','PO3','PO4','PO5','PO6','PO7','PO8','POz','Pz','T7','T8','TP7','TP8']

def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * (w**-0.5)) @ V.T
def logm_spd(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V * np.log(w)) @ V.T
def tangent_set(C, P):
    d = C.shape[-1]; iu = np.triu_indices(d); sc = np.sqrt(2)*np.ones((d, d)); sc[np.diag_indices(d)] = 1; sc = sc[iu]
    return np.array([(logm_spd(P @ C[i] @ P))[iu] * sc for i in range(len(C))])

prm = LeftRightImagery(channels=EEG60, resample=250, fmin=8, fmax=30)
ds = Stieger2021()
cases = dict(np.load(CASES)) if os.path.exists(CASES) else {}
done = sorted({int(k.split("_")[1]) for k in cases if k.startswith("cs_")})
print(f"resume: {len(done)} subjects already done: {done}", flush=True)

for sub in ds.subject_list:
    if sub in done: continue
    try:
        X, y, meta = prm.get_data(dataset=ds, subjects=[sub])
        y = (y == 'right_hand').astype(int); sess = meta['session'].values
        order = sorted(set(sess), key=lambda v: int(v)); s1 = order[0]; m1 = sess == s1
        if len(np.unique(y[m1])) < 2: print(f"S{sub}: sess1 single-class skip", flush=True); continue
        C1 = cov(X[m1].astype(np.float64)); P1 = invsqrtm(C1.mean(0))
        lda = LDA(solver="lsqr", shrinkage="auto").fit(tangent_set(C1, P1), y[m1])
        ncase = 0
        for sj in order[1:]:
            mj = sess == sj; yj = y[mj]
            if len(yj) < 8 or len(np.unique(yj)) < 2: continue
            Cj = cov(X[mj].astype(np.float64)); Pj = invsqrtm(Cj.mean(0))
            c_s = (lda.predict(tangent_set(Cj, P1)) == yj).astype(np.float32)
            c_a = (lda.predict(tangent_set(Cj, Pj)) == yj).astype(np.float32)
            cases[f"cs_{sub}_{sj}"] = c_s; cases[f"ca_{sub}_{sj}"] = c_a; ncase += 1
        np.savez_compressed(CASES, **cases)  # CHECKPOINT after each subject
        print(f"S{sub}: {ncase} cases done, total cases={len([k for k in cases if k.startswith('cs_')])}", flush=True)
        del X, y, meta, C1, lda; gc.collect()  # free MNE/moabb memory per subject
    except Exception as e:
        print(f"S{sub}: FAIL {str(e)[:100]}", flush=True); gc.collect()

# ---------- gate eval on accumulated cases ----------
keys = sorted({k[3:] for k in cases if k.startswith("cs_")}, key=lambda s: (int(s.split("_")[0]), int(s.split("_")[1])))
S = [(cases[f"cs_{k}"], cases[f"ca_{k}"]) for k in keys]
D = np.array([(ca.mean()-cs.mean())*100 for cs, ca in S])
print(f"\n===== SPREAD (n={len(S)} subj-session cases) =====")
print(f"meanΔ={D.mean():+.2f} std={D.std():.2f} helped(>1)={int((D>1).sum())} harmed(<-1)={int((D<-1).sum())} "
      f"worst={D.min():+.1f} best={D.max():+.1f}  (MIXED if both helped&harmed are sizable)")
def eb(dhat, se):
    mu = np.full_like(dhat, dhat.mean()); tau2 = max(dhat.var()-(se**2).mean(), 1e-6)
    prec = 1/se**2 + 1/tau2; return (dhat/se**2 + mu/tau2)/prec - 1.645*np.sqrt(1/prec) > 0
print(f"\n===== EB-pooled gate (leak-free) =====")
print(f"{'k':>3} {'always':>7} {'oracle':>7} {'LCB effΔ(害)':>13} {'EB effΔ(害,適応率)':>20}")
gate = []
for k in [8, 16, 32]:
    dhat=[];se=[];cst=[];cat=[]
    for cs, ca in S:
        if len(cs) <= k: continue
        diff=(ca-cs)[:k]; dhat.append(diff.mean()); se.append(max(diff.std(ddof=1)/np.sqrt(k) if k>1 else .5,1e-3))
        cst.append(cs[k:].mean()); cat.append(ca[k:].mean())
    dhat,se,cst,cat=map(np.array,(dhat,se,cst,cat)); base=cst.mean()
    def ev(ad): eff=np.where(ad,cat,cst); return (eff.mean()-base)*100, int(((cat-cst<-0.01)&ad).sum()), float(ad.mean())
    l=ev(dhat-1.645*se>0); e=ev(eb(dhat,se))
    alw=(cat.mean()-base)*100; ora=(np.maximum(cst,cat).mean()-base)*100
    print(f"{k:>3} {alw:>+7.2f} {ora:>+7.2f}  {l[0]:>+6.2f}({l[1]})       {e[0]:>+6.2f}({e[1]},{e[2]:.2f})")
    gate.append(dict(k=k, n=len(dhat), always=alw, oracle=ora, lcb_eff=l[0], lcb_harm=l[1], eb_eff=e[0], eb_harm=e[1], eb_adopt=e[2]))
json.dump({"n_cases":len(S),"spread":dict(mean=float(D.mean()),std=float(D.std()),helped=int((D>1).sum()),harmed=int((D<-1).sum()),worst=float(D.min()),best=float(D.max())),"gate":gate},
          open(f"{RES}/260609_stieger_gate.json","w"), indent=2)
print(f"\nsaved {RES}/260609_stieger_gate.json (cases checkpointed in {os.path.basename(CASES)})")

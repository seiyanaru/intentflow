"""STEP 2 (DC-Commit L3): self-train the diverse model D on TARGET-session
trials where T (TCFormer source) and D AGREE (94%-clean pseudo-labels), then test
whether the adapted ensemble captures the residual headroom — especially on the
DISAGREEMENT trials (the hard trials, not used for pseudo-training).
Run with intentflow conda env (rebuilds D tangent features from raw bcic2a).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from datamodules.bcic4_2a import BCICIV2a

EXP = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz"
exp = np.load(EXP, allow_pickle=True); ex = [str(x) for x in exp["experts"].tolist()]
EI = {n: ex.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
LAB = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/labels"
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=DATA, eval_label_path=LAB)
b, a = butter(4, [8/125., 30/125.], btype="band")
def bp(X): return filtfilt(b, a, X, axis=-1).copy()
def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V*(w**-0.5))@V.T
def logm_spd(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V*np.log(w))@V.T
def ea(C): R = C.mean(0); P = invsqrtm(R); return np.einsum('ij,njk,kl->nil', P, C, P)
def tangent(C):
    c = C.shape[1]; iu = np.triu_indices(c); s = np.sqrt(2)*np.ones((c, c)); np.fill_diagonal(s, 1.)
    return np.array([(logm_spd(C[i])*s)[iu] for i in range(C.shape[0])])
def getxy(ds):
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    y = np.array([int(ds[i][1]) for i in range(len(ds))]); return X, y
def fitD(Xtan, y):
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xtan, y)
    def pred(T):
        pr = lda.predict_proba(T); P = np.zeros((len(T), 4)); P[:, lda.classes_] = pr; return P
    return pred
W = 0.3
res = {k: [] for k in ["src","blend","DA","DA_adapt","ceil","dis_DA","dis_DAadapt","dis_n","agr_prec"]}
for sid in range(1, 10):
    i = sid-1; y = exp["labels"][i]
    T = exp["probs"][i, EI["source"]]; F = exp["probs"][i, EI["full_ea"]]; S = exp["probs"][i, EI["shrink_0.1"]]
    blend = 0.3*T+0.4*F+0.3*S
    dm = BCICIV2a(prep, sid); dm.setup()
    Xtr, ytr = getxy(dm.train_dataset); Xte, yte = getxy(dm.test_dataset)
    assert (yte == y).all(), f"S{sid} align fail"
    Ttr = tangent(ea(cov(bp(Xtr)))); Tte = tangent(ea(cov(bp(Xte))))
    D_src = fitD(Ttr, ytr); Dp = D_src(Tte)
    # agreement pseudo-labels on target
    tp = T.argmax(1); dp = Dp.argmax(1); agr = tp == dp
    pl = tp[agr]
    res["agr_prec"].append((tp[agr] == y[agr]).mean()*100)
    # self-train D on target-agreement pseudo-labels (+ source for stability)
    Xcomb = np.concatenate([Ttr, Tte[agr]]); ycomb = np.concatenate([ytr, pl])
    D_adapt = fitD(Xcomb, ycomb); Dp2 = D_adapt(Tte)
    ens = (blend + W*Dp).argmax(1); ens2 = (blend + W*Dp2).argmax(1)
    res["src"].append((T.argmax(1) == y).mean()*100)
    res["blend"].append((blend.argmax(1) == y).mean()*100)
    res["DA"].append((ens == y).mean()*100)
    res["DA_adapt"].append((ens2 == y).mean()*100)
    res["ceil"].append((blend.argmax(1) == y)|((blend.argmax(1)!=y)&False))  # placeholder
    # held-out DISAGREEMENT trials (NOT used for pseudo-training): the hard trials
    dis = ~agr
    res["dis_n"].append(dis.sum())
    res["dis_DA"].append((ens[dis] == y[dis]).mean()*100 if dis.sum() else np.nan)
    res["dis_DAadapt"].append((ens2[dis] == y[dis]).mean()*100 if dis.sum() else np.nan)
    print(f"S{sid}: blend {res['blend'][-1]:.2f} DA {res['DA'][-1]:.2f} DA+adapt {res['DA_adapt'][-1]:.2f} | "
          f"disagree(n={dis.sum()}): DA {res['dis_DA'][-1]:.1f} -> DA+adapt {res['dis_DAadapt'][-1]:.1f} | agr_prec {res['agr_prec'][-1]:.1f}", flush=True)
m = lambda k: float(np.nanmean(res[k]))
print(f"\n=== STEP 2: agreement-pseudo-label self-training (2a seed0, frozen w=0.3) ===")
print(f"source {m('src'):.2f} | blend {m('blend'):.2f} | DA-L1 {m('DA'):.2f} (+{m('DA')-m('blend'):.2f}) | DA+L3adapt {m('DA_adapt'):.2f} (+{m('DA_adapt')-m('blend'):.2f} vs blend, +{m('DA_adapt')-m('DA'):.2f} vs DA-L1)")
print(f"agreement pseudo-label precision: {m('agr_prec'):.1f}%")
print(f"on HELD-OUT DISAGREEMENT trials: DA {m('dis_DA'):.2f} -> DA+L3adapt {m('dis_DAadapt'):.2f} ({m('dis_DAadapt')-m('dis_DA'):+.2f})  <- does clean-pseudo adaptation generalize to hard trials?")
hsc = sum(1 for i in range(9) if res['DA_adapt'][i] < res['src'][i]-1e-9)
print(f"HSC(DA+adapt vs source): {hsc}/9")

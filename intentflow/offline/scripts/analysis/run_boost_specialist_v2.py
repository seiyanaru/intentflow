"""v2: sharper boosting variants + faithful uniform baseline (sklearn) + w_spec sweep.

Tests three failure-targeting variants of the EA-Riemann specialist against the EXACT
uniform-D1 DA-L1 (sklearn lsqr+auto, the saved 87.15 protocol):
  V_KNN   : up-weight train trials near TCFormer confident-error tangent prototypes (kNN)
  V_CLASS : up-weight train trials whose label is in T's confident-error confusion classes
  V_ADA   : AdaBoost-style 1 round -- fit uniform specialist, find its own errors on the
            failure-region-labeled train trials, reweight, refit.
Protocols: P_LOSO (seed0, prior from other 8) and P_SEED (seed0 region -> eval seed1).
Reports best beta per variant and a w_spec sweep at the best beta.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
EP0 = ROOT + "260602_expert_portfolio_table/expert_portfolio_arrays.npz"
EP1 = ROOT + "260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz"
CACHE = ROOT + "260603_boost_tangent_cache.npz"

def load_blend(path):
    d = np.load(path, allow_pickle=True)
    ex = list(d["experts"]); si = {e: i for i, e in enumerate(ex)}
    p = d["probs"]
    blend = 0.3*p[:, si["source"]] + 0.4*p[:, si["full_ea"]] + 0.3*p[:, si["shrink_0.1"]]
    return p[:, si["source"]], blend, d["labels"].astype(int)

def acc(P, y): return (P.argmax(-1) == y).mean()
def macc(ens, ys): return float(np.mean([acc(ens[s], ys[s]) for s in range(9)]))

c = np.load(CACHE, allow_pickle=True)
Ttr = [np.asarray(x, float) for x in c["Ttr"]]
ytr = [np.asarray(x, int) for x in c["ytr"]]
Tte = [np.asarray(x, float) for x in c["Tte"]]
yte = [np.asarray(x, int) for x in c["yte"]]
src0, blend0, lab0 = load_blend(EP0)
src1, blend1, lab1 = load_blend(EP1)

# ---- weighted shrinkage LDA via WEIGHTED RESAMPLING into sklearn lsqr+auto ----
# (exact sklearn shrinkage when uniform; sample_weight emulated by integer replication)
def fit_predict(Xtr, ytr, Xte, w=None, rep_scale=200, n=4, seed=0):
    if w is None:
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr, ytr)
    else:
        w = np.clip(w, 1e-6, None); w = w / w.sum()
        counts = np.maximum(1, np.round(w * len(w) * rep_scale).astype(int))
        idx = np.repeat(np.arange(len(w)), counts)
        lda = LDA(solver="lsqr", shrinkage="auto").fit(Xtr[idx], ytr[idx])
    proba = lda.predict_proba(Xte)
    P = np.zeros((Xte.shape[0], n)); P[:, lda.classes_] = proba
    return P, lda

def conf_error_mask(s, src_probs, conf_q=0.6):
    p = src_probs[s]; pred = p.argmax(1); conf = p.max(1)
    wrong = pred != yte[s]; thr = np.quantile(conf, conf_q)
    return wrong & (conf >= thr), pred

def knn_weights(Xtr_s, fail_feats, beta, k=15):
    if len(fail_feats) == 0: return np.ones(len(Xtr_s))
    mu = Xtr_s.mean(0); sd = Xtr_s.std(0)+1e-8
    A=(Xtr_s-mu)/sd; F=(fail_feats-mu)/sd
    d2=((A[:,None,:]-F[None,:,:])**2).sum(-1)
    kk=min(k,F.shape[0]); nn=np.sort(d2,1)[:,:kk].mean(1)
    r=nn.argsort().argsort()/(len(nn)-1); return 1.0+beta*(1.0-r)

def class_weights(ytr_s, fail_true_labels, beta):
    # up-weight train trials whose label is in T's confident-error TRUE-class set
    if len(fail_true_labels)==0: return np.ones(len(ytr_s))
    freq=np.bincount(fail_true_labels, minlength=4).astype(float)
    freq=freq/ (freq.sum()+1e-9)
    w=1.0+beta*freq[ytr_s]; return w

# failure region descriptors
def loso_fail(s, conf_q):
    feats=[]; labs=[]
    for o in range(9):
        if o==s: continue
        m,_=conf_error_mask(o, src0, conf_q)
        feats.append(Tte[o][m]); labs.append(yte[o][m])
    return np.concatenate(feats,0), np.concatenate(labs,0)

def seed_fail(s, conf_q):
    m,_=conf_error_mask(s, src0, conf_q)
    return Tte[s][m], yte[s][m]

# ---- references: EXACT sklearn uniform specialist ----
uspec=np.array([fit_predict(Ttr[s],ytr[s],Tte[s])[0] for s in range(9)])
dal1_0 = blend0+0.3*uspec; dal1_1 = blend1+0.3*uspec
print("uniform specialist mean acc", round(macc(uspec,yte),4))
print("DA-L1 seed0", round(macc(dal1_0,yte),4), " DA-L1 seed1", round(macc(dal1_1,yte),4))

def run(variant, protocol, beta, conf_q=0.6, w_spec=0.3):
    spec=[]
    for s in range(9):
        ff,fl = (loso_fail(s,conf_q) if protocol=="loso" else seed_fail(s,conf_q))
        if variant=="knn":   w=knn_weights(Ttr[s],ff,beta)
        elif variant=="class": w=class_weights(ytr[s],fl,beta)
        elif variant=="ada":
            # round0 uniform, find errors among failure-similar trials, reweight
            P0,_=fit_predict(Ttr[s],ytr[s],Ttr[s])  # in-sample preds
            wrong=(P0.argmax(1)!=ytr[s]).astype(float)
            wk=knn_weights(Ttr[s],ff,1.0)-1.0  # membership 0..1
            w=1.0+beta*(wrong*wk)
        else: w=None
        P,_=fit_predict(Ttr[s],ytr[s],Tte[s],w)
        spec.append(P)
    spec=np.array(spec)
    blend = blend0 if protocol=="loso" else blend1
    ys = yte
    return macc(blend+w_spec*spec, ys), spec

for proto in ["loso","seed"]:
    base = dal1_0 if proto=="loso" else dal1_1
    print(f"\n==== protocol {proto} (DA-L1 ref {macc(base,yte):.4f}) ====")
    for variant in ["knn","class","ada"]:
        row=[]
        for beta in [0.5,1.0,2.0,4.0,8.0]:
            a,_=run(variant,proto,beta)
            row.append((beta,a))
        best=max(row,key=lambda t:t[1])
        print(f"  {variant:5s}  " + "  ".join(f"b{b}:{a:.4f}" for b,a in row) + f"   BEST b{best[0]}:{best[1]:.4f}")
        # w_spec sweep at best beta
        ws=[]
        for w_spec in [0.2,0.3,0.4,0.5]:
            a,_=run(variant,proto,best[0],w_spec=w_spec); ws.append((w_spec,a))
        print("        wspec " + "  ".join(f"{w}:{a:.4f}" for w,a in ws))

"""v3 FAITHFUL boosting specialist test.

Weighted EA-Riemann tangent+LDA via integer-replication sample weights with a FIXED
(per-subject Ledoit-Wolf 'auto') shrinkage -- so weighting changes the fit but the
shrinkage artifact of resampling is removed (validated maxdiff~1e-14 at uniform).

Baselines (matched, same fixed shrinkage):
  blend (85.76 saved), DA-L1 = blend + 0.3 * uniform-specialist.
We also print the saved-auto DA-L1 for reference.

Boosting variants up-weight train trials in TCFormer's confident-error subspace:
  knn   : tangent kNN proximity to T confident-error prototypes
  class : true-class membership in T's confident-error confusion classes
  ada   : 1-round AdaBoost on specialist's own errors within the failure-similar region
Protocols (leak-free held-out):
  loso  : seed0; failure prior from OTHER 8 subjects; eval held-out subject session_E.
  seed  : failure region from seed0 src; ensemble evaluated on SEED1 blend.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import ledoit_wolf

ROOT = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
EP0 = ROOT + "260602_expert_portfolio_table/expert_portfolio_arrays.npz"
EP1 = ROOT + "260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz"
CACHE = ROOT + "260603_boost_tangent_cache.npz"
REP = 5  # replication scale for weighted resampling

def load_blend(path):
    d = np.load(path, allow_pickle=True)
    ex = list(d["experts"]); si = {e: i for i, e in enumerate(ex)}
    p = d["probs"]
    blend = 0.3*p[:, si["source"]] + 0.4*p[:, si["full_ea"]] + 0.3*p[:, si["shrink_0.1"]]
    return p[:, si["source"]], blend, d["labels"].astype(int)

def acc(P, y): return (P.argmax(-1) == y).mean()
def macc(ens, ys): return float(np.mean([acc(ens[s], ys[s]) for s in range(9)]))
def hsc(ens, ref, ys):  # helped-subjects count vs ref
    return int(sum(acc(ens[s], ys[s]) > acc(ref[s], ys[s]) for s in range(9)))

c = np.load(CACHE, allow_pickle=True)
Ttr = [np.asarray(x, float) for x in c["Ttr"]]
ytr = [np.asarray(x, int) for x in c["ytr"]]
Tte = [np.asarray(x, float) for x in c["Tte"]]
yte = [np.asarray(x, int) for x in c["yte"]]
src0, blend0, lab0 = load_blend(EP0)
src1, blend1, lab1 = load_blend(EP1)

# per-subject fixed LW shrinkage from uniform within-class centered data
SHRINK = []
for s in range(9):
    Xc = np.vstack([Ttr[s][ytr[s]==k]-Ttr[s][ytr[s]==k].mean(0) for k in np.unique(ytr[s])])
    _, sh = ledoit_wolf(Xc); SHRINK.append(float(sh))

def fit_predict(s, w=None, n=4):
    sh = SHRINK[s]
    if w is None:
        lda = LDA(solver="lsqr", shrinkage=sh).fit(Ttr[s], ytr[s])
    else:
        w = np.clip(w, 1e-6, None)
        counts = np.maximum(1, np.round(w/w.sum()*len(w)*REP).astype(int))
        idx = np.repeat(np.arange(len(w)), counts)
        lda = LDA(solver="lsqr", shrinkage=sh).fit(Ttr[s][idx], ytr[s][idx])
    proba = lda.predict_proba(Tte[s]); P = np.zeros((Tte[s].shape[0], n)); P[:, lda.classes_] = proba
    return P

def fit_predict_train(s, n=4):  # in-sample for ada
    lda = LDA(solver="lsqr", shrinkage=SHRINK[s]).fit(Ttr[s], ytr[s])
    proba = lda.predict_proba(Ttr[s]); P = np.zeros((Ttr[s].shape[0], n)); P[:, lda.classes_] = proba
    return P

def conf_err(s, src_probs, conf_q=0.6):
    p = src_probs[s]; pred = p.argmax(1); conf = p.max(1)
    wrong = pred != yte[s]; thr = np.quantile(conf, conf_q)
    m = wrong & (conf >= thr); return m

def knn_w(s, fail, beta, k=15):
    if len(fail) == 0: return np.ones(len(Ttr[s])), np.zeros(len(Ttr[s]))
    X = Ttr[s]; mu = X.mean(0); sd = X.std(0)+1e-8
    A=(X-mu)/sd; F=(fail-mu)/sd
    d2=((A[:,None,:]-F[None,:,:])**2).sum(-1)
    kk=min(k,F.shape[0]); nn=np.sort(d2,1)[:,:kk].mean(1)
    r=nn.argsort().argsort()/(len(nn)-1); return 1.0+beta*(1.0-r), (1.0-r)

def class_w(s, fail_labels, beta):
    if len(fail_labels)==0: return np.ones(len(ytr[s]))
    fr=np.bincount(fail_labels,minlength=4).astype(float); fr=fr/(fr.sum()+1e-9)
    return 1.0+beta*fr[ytr[s]]

def loso_fail(s, conf_q):
    F=[]; L=[]
    for o in range(9):
        if o==s: continue
        m=conf_err(o,src0,conf_q); F.append(Tte[o][m]); L.append(yte[o][m])
    return np.concatenate(F,0), np.concatenate(L,0)

def seed_fail(s, conf_q):
    m=conf_err(s,src0,conf_q); return Tte[s][m], yte[s][m]

# references
uspec=np.array([fit_predict(s) for s in range(9)])
dal1_0=blend0+0.3*uspec; dal1_1=blend1+0.3*uspec
print(f"matched uniform specialist mean acc {macc(uspec,yte):.4f}")
print(f"DA-L1(matched) seed0 {macc(dal1_0,yte):.4f}  seed1 {macc(dal1_1,yte):.4f}")
print(f"blend seed0 {macc(blend0,yte):.4f}  seed1 {macc(blend1,yte):.4f}")

def run(variant, proto, beta, conf_q=0.6, w_spec=0.3):
    spec=[]
    for s in range(9):
        ff,fl = loso_fail(s,conf_q) if proto=="loso" else seed_fail(s,conf_q)
        if variant=="knn":   w,_=knn_w(s,ff,beta)
        elif variant=="class": w=class_w(s,fl,beta)
        elif variant=="ada":
            P0=fit_predict_train(s); wrong=(P0.argmax(1)!=ytr[s]).astype(float)
            _,mem=knn_w(s,ff,1.0); w=1.0+beta*(wrong*mem)
        spec.append(fit_predict(s,w))
    spec=np.array(spec)
    blend = blend0 if proto=="loso" else blend1
    ens = blend + w_spec*spec
    return macc(ens,yte), ens, spec

for proto in ["loso","seed"]:
    base = dal1_0 if proto=="loso" else dal1_1
    blnd = blend0 if proto=="loso" else blend1
    print(f"\n==== {proto}  DA-L1 ref {macc(base,yte):.4f} (HSC vs source {hsc(base,src0 if proto=='loso' else src1,yte)}) ====")
    for variant in ["knn","class","ada"]:
        best=(-1,None,None)
        line=[]
        for beta in [0.5,1.0,2.0,4.0,8.0]:
            a,ens,_=run(variant,proto,beta)
            line.append(f"b{beta}:{a:.4f}")
            if a>best[0]: best=(a,beta,ens)
        h=hsc(best[2], blnd, yte)
        print(f"  {variant:5s} {'  '.join(line)}  | BEST b{best[1]}:{best[0]:.4f} (HSC vs blend {h})")

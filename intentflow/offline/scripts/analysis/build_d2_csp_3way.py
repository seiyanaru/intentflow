"""3rd diverse model D2 = EA + one-vs-rest CSP + LDA (discriminative spatial,
mechanistically different from D1's Riemannian tangent geometry). Build on 2a,
then measure 3-way diversity & whether T/D1/D2 schemes capture more of the
residual headroom than the 2-way DA-L1. Run with intentflow conda env.
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
D1F = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz"
exp = np.load(EXP, allow_pickle=True); ex = [str(x) for x in exp["experts"].tolist()]
EI = {n: ex.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
D1 = np.load(D1F, allow_pickle=True)["probs"]
DATA = "/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
LAB = DATA + "labels"
prep = dict(sfreq=250, low_cut=None, high_cut=None, start=0.0, stop=4.0,
            batch_size=48, test_batch_size=48, num_workers=0, z_scale=True,
            data_path=DATA, eval_label_path=LAB)
b, a = butter(4, [8/125., 30/125.], btype="band")
def bp(X): return filtfilt(b, a, X, axis=-1).copy()
def covn(X):  # trial-normalized covariances
    C = np.einsum('nct,ndt->ncd', X, X)
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None]
    return C / np.clip(tr, 1e-12, None)
def invsqrtm(M): w, V = eigh(M); w = np.clip(w, 1e-10, None); return (V*(w**-0.5))@V.T
def ea_filters(Xtr):  # EA reference from train mean cov
    R = covn(Xtr).mean(0); return invsqrtm(R)
def csp_ovr(Xtr, ytr, m=3):
    classes = np.unique(ytr)
    covs = {c: covn(Xtr[ytr == c]).mean(0) for c in classes}
    filt = []
    for c in classes:
        Cc = covs[c]; Cr = sum(covs[k] for k in classes if k != c)/(len(classes)-1)
        w, V = eigh(Cc, Cc+Cr); idx = np.argsort(w)[::-1]
        filt.append(V[:, idx[:m]])
    return np.concatenate(filt, 1)  # (C, 4m)
def csp_feat(X, W): return np.log(np.var(np.einsum('cf,nct->nft', W, X), axis=-1)+1e-8)
def getxy(ds):
    X = np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    return X, np.array([int(ds[i][1]) for i in range(len(ds))])

D2_all = []
for sid in range(1, 10):
    dm = BCICIV2a(prep, sid); dm.setup()
    Xtr, ytr = getxy(dm.train_dataset); Xte, yte = getxy(dm.test_dataset)
    Xtr, Xte = bp(Xtr), bp(Xte)
    Ptr = ea_filters(Xtr); Pte = ea_filters(Xte)  # EA whiten per domain
    Xtr_a = np.einsum('ij,njt->nit', Ptr, Xtr); Xte_a = np.einsum('ij,njt->nit', Pte, Xte)
    W = csp_ovr(Xtr_a, ytr)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(csp_feat(Xtr_a, W), ytr)
    pr = lda.predict_proba(csp_feat(Xte_a, W)); P = np.zeros((len(yte), 4)); P[:, lda.classes_] = pr
    D2_all.append(P)
    print(f"S{sid}: D2(CSP) acc={(P.argmax(1)==yte).mean()*100:.2f}", flush=True)
D2 = np.array(D2_all)
np.savez("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260603_d2_csp_preds.npz", probs=D2, subjects=np.arange(1,10))

m = lambda L: float(np.mean(L))
acc=lambda p,y:(p.argmax(1)==y).mean()*100
# diversity + 3-way + realized
d2acc=[];resc_T=[];ov_d1d2=[];orac2=[];orac3=[];ag3_cov=[];ag3_prec=[]
da_l1=[];e3=[];blend_=[];src_=[];maj=[]
for i in range(9):
    y=exp["labels"][i];T=exp["probs"][i,EI["source"]];F=exp["probs"][i,EI["full_ea"]];S=exp["probs"][i,EI["shrink_0.1"]]
    d1=D1[i];d2=D2[i];blend=0.3*T+0.4*F+0.3*S
    src_.append(acc(T,y));blend_.append(acc(blend,y));d2acc.append(acc(d2,y))
    tp=T.argmax(1);d1p=d1.argmax(1);d2p=d2.argmax(1);Tc=tp==y
    err=~Tc;tmarg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2];conf_err=err&(tmarg>np.median(tmarg))
    resc_T.append((d2p[conf_err]==y[conf_err]).mean()*100 if conf_err.sum() else 0)
    e1=~ (d1p==y); e2=~(d2p==y); ov_d1d2.append((e1&e2).sum()/max(1,(e1|e2).sum())*100)
    orac2.append((Tc|(d1p==y)).mean()*100); orac3.append((Tc|(d1p==y)|(d2p==y)).mean()*100)
    ag3=(tp==d1p)&(tp==d2p); ag3_cov.append(ag3.mean()*100); ag3_prec.append((tp[ag3]==y[ag3]).mean()*100 if ag3.sum() else 0)
    da_l1.append(acc(blend+0.3*d1,y))
    e3.append(acc(blend+0.3*d1+0.3*d2,y))
    # majority vote among T,D1,D2 (fallback blend on no-majority)
    out=tp.copy()
    for t in range(len(y)):
        votes=[tp[t],d1p[t],d2p[t]];import collections
        cnt=collections.Counter(votes);top,n=cnt.most_common(1)[0]
        out[t]= top if n>=2 else blend[t].argmax()
    maj.append((out==y).mean()*100)
print(f"\n=== 3rd model D2 (CSP) — diversity & 3-way (2a seed0) ===")
print(f"D2 acc alone: {m(d2acc):.2f} (D1 was 70.0, T-source 82.7)")
print(f"D2 rescue on T confident-errors: {m(resc_T):.1f}% (D1 was 46%)")
print(f"D1∩D2 error overlap (Jaccard): {m(ov_d1d2):.1f}% (low=diverse from each other)")
print(f"per-trial oracle: (blend,D1)={m(orac2):.2f} -> (blend,D1,D2)={m(orac3):.2f} (+{m(orac3)-m(orac2):.2f})")
print(f"3-way agreement: coverage {m(ag3_cov):.1f}% precision {m(ag3_prec):.2f}%")
print(f"\n=== realized (frozen w=0.3) ===")
print(f"blend {m(blend_):.2f} | DA-L1 (2-way) {m(da_l1):.2f} | blend+0.3D1+0.3D2 {m(e3):.2f} ({m(e3)-m(da_l1):+.2f} vs DA-L1) | majority-vote {m(maj):.2f}")
hsc=sum(1 for i in range(9) if e3[i]<src_[i]-1e-9)
print(f"HSC(3-way ens vs source): {hsc}/9 | vs blend {sum(1 for i in range(9) if e3[i]<blend_[i]-1e-9)}/9")
PY=0

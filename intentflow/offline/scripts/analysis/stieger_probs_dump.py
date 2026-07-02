"""E-A: does a LABEL-FREE signal flag the harmful Stieger sessions (the make-or-break for a free veto)?

Self-contained re-decode of Stieger2021 subjects 1-60 (cached .mat only -> no download race with a
running stieger_eb_gate.py; writes a SEPARATE file). For each (subj, session j>=2):
  - frozen session-1 Riemann-tangent-LDA source; adapter = EA-recenter to session-j own reference.
  - LABEL-FREE signals computed from the FIRST k=32 trials' source/adapt PROBABILITIES only (no y),
    i.e. exactly what you'd see BEFORE deciding to adopt:
      H            = mean( max(p_src) * 1[argmax p_src != argmax p_adapt] )   (confident-overrule mass)
      overrule     = mean( 1[argmax differ] )
      conf_drop    = mean max(p_src) - mean max(p_adapt)                       (adaptation lowered confidence)
      entropy_a    = mean entropy(p_adapt)
      dispersity   = nuclear_norm(P_adapt)/sqrt(k*C)                           (Deng ICML23, lower=collapsed)
      riemann_dist = AIRM distance( cov_mean(session-j, first-k) , cov_mean(session-1) )  (drift, model-free)
  - GROUND TRUTH (leak-free): true Δacc on held-out trials [k:] -> harmful = (Δ < -1pp).
Then AUROC / Spearman of each signal vs harmful, + leave-one-subject-out logistic combo AUROC.
Lee2019 reference: confident-overrule H had rho=-0.72 with Δ. Cross-dataset generalization is the test.
"""
import os, warnings, json, numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA", "/home/islabshi/workspace-local2/mne_data")
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import glob
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
CACHE = "/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"  # from stieger_dump_epochs.py
K = 32  # probe window for label-free signals (decision-time, no labels); eval on [K:]

def cov(X): return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
def invsqrtm(M): w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def logm_spd(M): w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T
def tangent_set(C,P):
    d=C.shape[-1];iu=np.triu_indices(d);sc=np.sqrt(2)*np.ones((d,d));sc[np.diag_indices(d)]=1;sc=sc[iu]
    return np.array([(logm_spd(P@C[i]@P))[iu]*sc for i in range(len(C))])
def airm(A,B):  # AIRM distance between SPD A,B
    Ai=invsqrtm(A);w=eigh(Ai@B@Ai,eigvals_only=True);w=np.clip(w,1e-10,None);return float(np.sqrt((np.log(w)**2).sum()))
def ent(P): return float(-(P*np.log(np.clip(P,1e-12,1))).sum(1).mean())
def disp(P): return float(np.linalg.norm(P,'nuc')/np.sqrt(P.shape[0]*P.shape[1]))

files=sorted(glob.glob(f"{CACHE}/S*_epochs.npz"), key=lambda f:int(os.path.basename(f).split("_")[0][1:]))
rows=[]  # each: dict(sub,sj,true_d, H,overrule,conf_drop,entropy,dispersity,riemann)
for f in files:
    sub=int(os.path.basename(f).split("_")[0][1:])
    try:
        z=np.load(f);X=z["X"].astype(np.float64);y=z["y"].astype(int);sess=z["sess"]
        order=sorted(set(sess.tolist()));s1=order[0];m1=sess==s1
        if len(np.unique(y[m1]))<2: print(f"S{sub}: s1 single-class skip",flush=True);continue
        C1=cov(X[m1].astype(np.float64));R1=C1.mean(0);P1=invsqrtm(R1)
        lda=LDA(solver="lsqr",shrinkage="auto").fit(tangent_set(C1,P1),y[m1])
        nc=0
        for sj in order[1:]:
            mj=sess==sj;yj=y[mj]
            if len(yj)<K+10 or len(np.unique(yj))<2: continue
            Cj=cov(X[mj].astype(np.float64));Rj=Cj.mean(0);Pj=invsqrtm(Rj)
            Ts=tangent_set(Cj,P1);Ta=tangent_set(Cj,Pj)
            ps=lda.predict_proba(Ts);pa=lda.predict_proba(Ta)
            cs=(ps.argmax(1)==yj).astype(float);ca=(pa.argmax(1)==yj).astype(float)
            true_d=(ca[K:].mean()-cs[K:].mean())*100
            # label-free signals on first-K only
            psk=ps[:K];pak=pa[:K]
            ov=(psk.argmax(1)!=pak.argmax(1))
            H=float((psk.max(1)*ov).mean()); overrule=float(ov.mean())
            conf_drop=float(psk.max(1).mean()-pak.max(1).mean())
            Cjk=cov(X[mj][:K].astype(np.float64));Rjk=Cjk.mean(0)  # first-K cov drift
            rows.append(dict(sub=int(sub),sj=int(sj),true_d=float(true_d),
                H=H,overrule=overrule,conf_drop=conf_drop,entropy=ent(pak),
                dispersity=disp(pak),riemann=airm(R1,Rjk)));nc+=1
        print(f"S{sub}: {nc} sessions",flush=True)
        del X,y,z
    except Exception as e:
        print(f"S{sub}: FAIL {str(e)[:90]}",flush=True)

np.save(f"{RES}/260610_stieger_signals.npy", rows, allow_pickle=True)
n=len(rows);print(f"\n{n} sessions dumped")
harm=np.array([r['true_d']<-1 for r in rows]).astype(int)
print(f"harmful={harm.sum()}/{n}")
SIG=["H","overrule","conf_drop","entropy","dispersity","riemann"]
print(f"\n=== label-free signal -> harmful-session discrimination (AUROC, |Spearman vs Δ|) ===")
res={}
for s in SIG:
    v=np.array([r[s] for r in rows]);td=np.array([r['true_d'] for r in rows])
    # orient so higher=more harmful: AUROC with sign that maximizes
    auc=roc_auc_score(harm,v);auc=max(auc,1-auc)
    rho=spearmanr(v,td).correlation
    print(f"  {s:>11}: AUROC={auc:.3f}  Spearman(Δ)={rho:+.3f}")
    res[s]=dict(auroc=float(auc),spearman=float(rho))
# leave-one-subject-out logistic combo
subs=np.array([r['sub'] for r in rows]);Xm=np.array([[r[s] for s in SIG] for r in rows],float)
Xm=(Xm-Xm.mean(0))/(Xm.std(0)+1e-9);pred=np.zeros(n)
for u in sorted(set(subs)):
    tr=subs!=u;te=subs==u
    if len(set(harm[tr]))<2: pred[te]=harm[tr].mean();continue
    lr=LogisticRegression(max_iter=1000).fit(Xm[tr],harm[tr]);pred[te]=lr.predict_proba(Xm[te])[:,1]
combo=roc_auc_score(harm,pred)
print(f"  {'LOSO-combo':>11}: AUROC={combo:.3f}")
res["loso_combo"]=dict(auroc=float(combo))
res["n"]=n;res["harmful"]=int(harm.sum())
json.dump(res, open(f"{RES}/260610_stieger_signals_auroc.json","w"), indent=2)
print(f"\nsaved {RES}/260610_stieger_signals.npy + _auroc.json")
print("WIN if any label-free signal (or combo) AUROC >= 0.65 -> a free per-session harm veto is feasible.")

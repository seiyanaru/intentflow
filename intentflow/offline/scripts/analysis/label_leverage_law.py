"""Decisive test of the LABEL-LEVERAGE LAW: a label spent at the cross-family
DISAGREEMENT frontier is worth more than one spent by uncertainty sampling.
Base predictor = frozen deep T (source). Experts T(deep), R(Riemann), S(spectral).
Part A (selection quality, NO training): for top-K trials chosen by each criterion,
  density of {T-wrong, recoverable (T-wrong & a family right), confident-wrong}.
Part B (realized accuracy): use the K labels to train a per-subject 3-way router
  {trust T/R/S} (regularized logistic); labeled trials use true label; compare
  test accuracy of DISAGREE vs UNCERTAINTY vs RANDOM selection at matched K.
Cached preds only; no GPU needed. seed0 + seed1 (held-out frozen design check).
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
R=np.load(RES+"260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
C=np.load(RES+"260603_d2_csp_preds.npz",allow_pickle=True)["probs"]
S=np.load(RES+"260603_d3_spectral_preds.npz",allow_pickle=True)["probs"]
def load(seed):
    p=RES+f"260602_expert_portfolio_table{'' if seed==0 else '_seed1'}/expert_portfolio_arrays.npz"
    e=np.load(p,allow_pickle=True);si=[str(x) for x in e["experts"].tolist()].index("source")
    return e["probs"][:,si],e["labels"]
m=lambda L:float(np.nanmean(L))
def symkl(p,q):return ((p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(q,1e-12,1)))).sum(1)+(q*(np.log(np.clip(q,1e-12,1))-np.log(np.clip(p,1e-12,1)))).sum(1))
def ent(p):return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def feats(T,Ri,Si):
    mT=np.sort(T,1);mR=np.sort(Ri,1);mS=np.sort(Si,1)
    return np.stack([T.max(1),mT[:,-1]-mT[:,-2],ent(T),Ri.max(1),mR[:,-1]-mR[:,-2],Si.max(1),
                     symkl(T,Ri),symkl(T,Si),(T.argmax(1)!=Ri.argmax(1)).astype(float),(T.argmax(1)!=Si.argmax(1)).astype(float)],1)

for seed in [0,1]:
    Tall,Lab=load(seed)
    print(f"\n===== seed{seed} =====")
    base=m([(Tall[i].argmax(1)==Lab[i]).mean()*100 for i in range(9)])
    # oracle router ceiling (all labels, best of T/R/S per trial)
    orc=[]
    for i in range(9):
        cor=(Tall[i].argmax(1)==Lab[i])|(R[i].argmax(1)==Lab[i])|(S[i].argmax(1)==Lab[i]);orc.append(cor.mean()*100)
    print(f"base T {base:.2f} | oracle-router(T/R/S) {m(orc):.2f} | locked headroom {m(orc)-base:+.2f}pp")
    # ---- Part A: selection quality ----
    print("Part A (selection density in top-K, mean over subj):  [recoverable=T-wrong & (R or S right)]")
    print(f"{'K%':>4} {'crit':>10} {'T-wrong%':>9} {'recover%':>9} {'confwrong%':>11}")
    for frac in [0.05,0.10]:
        for crit in ["UNC","DIS","RAND"]:
            tw=[];rec=[];cw=[]
            for i in range(9):
                y=Lab[i];T=Tall[i];tp=T.argmax(1);n=len(y);K=max(1,int(n*frac))
                marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2]
                if crit=="UNC": sig=-marg
                elif crit=="DIS": sig=symkl(T,R[i])+symkl(T,S[i])
                else: sig=np.linspace(0,1,n)[np.random.RandomState(i).permutation(n)]
                sel=np.argsort(-sig)[:K]
                Tw=tp[sel]!=y[sel]; Rr=R[i].argmax(1)[sel]==y[sel]; Sr=S[i].argmax(1)[sel]==y[sel]
                hi=marg>=np.median(marg)
                tw.append(Tw.mean()*100); rec.append((Tw&(Rr|Sr)).mean()*100); cw.append((hi[sel]&Tw).mean()*100)
            print(f"{int(frac*100):>3}% {crit:>10} {m(tw):>9.1f} {m(rec):>9.1f} {m(cw):>11.1f}")
    # ---- Part B: GUARDED routing (only override in disagreement region; easy trials stay T) ----
    print("Part B GUARDED (override only where T disagrees w/ R or S; else trust T):")
    print(f"{'K%':>4} {'crit':>5} {'acc':>7} {'labels_in_Dregion':>18}")
    for frac in [0.05,0.10,0.20]:
        accs={"UNC":[],"DIS":[],"RAND":[]}; inreg={"UNC":[],"DIS":[],"RAND":[]}
        for i in range(9):
            y=Lab[i];T=Tall[i];tp=T.argmax(1);n=len(y);K=max(8,int(n*frac))
            X=feats(T,R[i],S[i]); X=(X-X.mean(0))/(X.std(0)+1e-8)
            preds=np.stack([tp,R[i].argmax(1),S[i].argmax(1)],1)
            tgt=np.zeros(n,int)
            for j in range(n):
                if preds[j,0]==y[j]: tgt[j]=0
                elif preds[j,1]==y[j]: tgt[j]=1
                elif preds[j,2]==y[j]: tgt[j]=2
            Dreg=(tp!=preds[:,1])|(tp!=preds[:,2])   # disagreement region
            marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2]
            for crit in accs:
                if crit=="UNC": sig=-marg
                elif crit=="DIS": sig=symkl(T,R[i])+symkl(T,S[i])
                else: sig=np.linspace(0,1,n)[np.random.RandomState(i).permutation(n)]
                lab_idx=np.argsort(-sig)[:K]; lab_mask=np.zeros(n,bool);lab_mask[lab_idx]=True
                out=tp.copy(); out[lab_idx]=y[lab_idx]
                tr=lab_idx[Dreg[lab_idx]]                      # labels that fell in disagreement region
                inreg[crit].append(len(tr))
                apply_idx=np.where(Dreg&~lab_mask)[0]          # unlabeled disagreement trials to override
                if len(np.unique(tgt[tr]))>=2 and len(tr)>=4:
                    clf=LogisticRegression(C=0.5,max_iter=2000).fit(X[tr],tgt[tr])
                    out[apply_idx]=preds[apply_idx,clf.predict(X[apply_idx])]
                accs[crit].append((out==y).mean()*100)
        for crit in ["UNC","DIS","RAND"]:
            print(f"{int(frac*100):>3}% {crit:>5} {m(accs[crit]):>7.2f} {m(inreg[crit]):>18.1f}")
print("\nLAW HOLDS if DIS > UNC on Part-A recoverable/confwrong density AND Part-B accuracy at matched K.")

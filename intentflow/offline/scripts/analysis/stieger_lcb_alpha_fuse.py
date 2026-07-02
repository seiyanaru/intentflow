"""per-session LCB x alpha-floor FUSION: keep harmed low (LCB's strength) WITHOUT throwing away gain.

per-session LCB rejects (a=0=source) when it cannot confirm benefit -> harmed=9 but meanD only +2.5.
Idea: when LCB cannot confirm, do NOT reject; instead apply a SOFT alpha-floor (a=a_floor, mostly source
but slightly adapted) so we recover some gain while still bounding harm. Sweep a_floor.

Two-pass to avoid re-decode cost being wasted:
 PASS1: per (subj,sess) dump probe LCB stats (dhat,se on first M) AND held-out acc over a fine a-grid.
        cache to 260622_lcb_alpha_curves.npz (reused on reruns).
 PASS2: pure-numpy policy eval from the cache:
        a = 1.0 if (dhat-1.645*se>0) else a_floor ;  Delta = acc[a]-acc[0] on held-out.
        sweep a_floor in {0(=plain LCB),0.2,0.3,0.5}. + always-EA / source / oracle for reference.
        subject-cluster bootstrap CI on (harmed,meanD) vs plain-LCB and vs always-EA.
Leak-free: LCB uses first M only; eval on [M:]. From cached epochs. CPU.
"""
import os, glob, warnings, json, numpy as np
warnings.filterwarnings("ignore")
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
np.random.seed(0)
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
CACHE="/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
CURVES=f"{RES}/260622_lcb_alpha_curves.npz"
M=16
AGRID=np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0])

def cov(X): return np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]
def invsqrtm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def sqrtm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*(w**0.5))@V.T
def logm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T
def tangent(C,P):
    d=C.shape[-1];iu=np.triu_indices(d);sc=np.sqrt(2)*np.ones((d,d));sc[np.diag_indices(d)]=1;sc=sc[iu]
    return np.array([(logm(P@C[i]@P))[iu]*sc for i in range(len(C))])
def basepoint(R1h,Mm,a):
    mu,U=eigh(Mm);mu=np.clip(mu,1e-10,None);return invsqrtm(R1h@((U*np.exp(a*np.log(mu)))@U.T)@R1h)

if os.path.exists(CURVES):
    z=np.load(CURVES); SUB=z["sub"];DH=z["dhat"];SE=z["se"];ACC=z["acc"]
    print(f"cache hit: {len(SUB)} sessions",flush=True)
else:
    rows_sub=[];rows_dh=[];rows_se=[];rows_acc=[]
    files=sorted(glob.glob(f"{CACHE}/S*_epochs.npz"),key=lambda f:int(os.path.basename(f).split("_")[0][1:]))
    for f in files:
        sub=int(os.path.basename(f).split("_")[0][1:])
        try:
            z=np.load(f);X=z["X"].astype(np.float64);y=z["y"].astype(int);sess=z["sess"]
            order=sorted(set(sess.tolist()));s1=order[0];m1=sess==s1
            if len(np.unique(y[m1]))<2: continue
            R1=cov(X[m1]).mean(0);P1=invsqrtm(R1);R1h=sqrtm(R1)
            lda=LDA(solver="lsqr",shrinkage="auto").fit(tangent(cov(X[m1]),P1),y[m1])
            for sj in order[1:]:
                mj=sess==sj;yj=y[mj];Xj=X[mj]
                if len(yj)<M+12 or len(np.unique(yj))<2: continue
                Cj=cov(Xj);Rj=Cj.mean(0);Mm=P1@Rj@P1
                cs=(lda.predict(tangent(Cj[:M],P1))==yj[:M]).astype(float)
                ca=(lda.predict(tangent(Cj[:M],invsqrtm(Rj)))==yj[:M]).astype(float)
                dhat=(ca.mean()-cs.mean())*100; se=max((ca-cs).std(ddof=1)/np.sqrt(M)*100,1e-3)
                acc=[ (lda.predict(tangent(Cj[M:],P1 if a==0 else basepoint(R1h,Mm,a)))==yj[M:]).mean() for a in AGRID ]
                rows_sub.append(sub);rows_dh.append(dhat);rows_se.append(se);rows_acc.append(acc)
            print(f"S{sub} done",flush=True); del X,y,z
        except Exception as e: print(f"S{sub} FAIL {str(e)[:60]}",flush=True)
    SUB=np.array(rows_sub);DH=np.array(rows_dh);SE=np.array(rows_se);ACC=np.array(rows_acc)
    np.savez_compressed(CURVES,sub=SUB,dhat=DH,se=SE,acc=ACC,agrid=AGRID)
    print(f"saved curves {len(SUB)} sessions",flush=True)

ai=lambda a: int(np.argmin(np.abs(AGRID-a)))
src=ACC[:,ai(0.0)]; ea=ACC[:,ai(1.0)]; true=(ea-src)*100
ora=np.maximum(0,true).mean(); base=src.mean()
confirm = DH-1.645*SE>0   # per-session LCB confirms benefit

def evalpol(adopt_acc):
    eff=adopt_acc; d=(eff-src)*100
    return dict(meanD=(eff.mean()-base)*100, pct=( (eff.mean()-base)*100)/ora*100,
                harmed=int((d<-1).sum()), worst=float(d.min()), eff=eff)
res={}
res["source"]=evalpol(src); res["always-EA"]=evalpol(ea)
res["oracle"]=evalpol(np.maximum(src,ea))
for af in [0.0,0.2,0.3,0.5]:
    a_used=np.where(confirm,1.0,af)
    acc_used=np.array([ACC[i,ai(a_used[i])] for i in range(len(SUB))])
    res[f"LCB+floor{af}"]=evalpol(acc_used)
print(f"\n{'policy':>16} {'meanD':>7} {'%ora':>6} {'harmed':>7} {'worst':>7}  (n={len(SUB)}, oracle=+{ora:.2f}, labels={M}/sess)")
for k,v in res.items():
    print(f"{k:>16} {v['meanD']:>+7.2f} {v['pct']:>5.0f}% {v['harmed']:>7d} {v['worst']:>+7.1f}")

# bootstrap: best floor vs plain LCB (floor0) and vs always-EA
def boot(aacc,bacc,key,B=3000):
    subs=np.array(sorted(set(SUB.tolist())));d=[]
    for _ in range(B):
        pick=np.random.choice(subs,len(subs),replace=True);m=np.concatenate([np.where(SUB==s)[0] for s in pick])
        da=(aacc[m]-src[m])*100; db=(bacc[m]-src[m])*100
        if key=='harmed': d.append(int((db<-1).sum())-int((da<-1).sum()))
        else: d.append(db.mean()-da.mean())
    return np.percentile(d,2.5),np.median(d),np.percentile(d,97.5)
print("\n=== bootstrap (B=floor版 − A=基準, harmed: 負=改善 / meanD: 正=改善) ===")
plain=np.array([ACC[i,ai(1.0 if confirm[i] else 0.0)] for i in range(len(SUB))])
for af in [0.2,0.3,0.5]:
    cand=np.array([ACC[i,ai(1.0 if confirm[i] else af)] for i in range(len(SUB))])
    lo,md,hi=boot(plain,cand,'harmed'); lo2,md2,hi2=boot(plain,cand,'meanD')
    print(f"  floor{af} vs plainLCB: Δharmed md={md:+.0f}[{lo:+.0f},{hi:+.0f}] | ΔmeanD md={md2:+.2f}[{lo2:+.2f},{hi2:+.2f}]")
json.dump({k:{kk:vv for kk,vv in v.items() if kk!='eff'} for k,v in res.items()},open(f"{RES}/260622_lcb_alpha_fuse.json","w"),indent=2)
print(f"\nsaved 260622_lcb_alpha_fuse.json")
print("WIN if a floor>0 keeps harmed near plainLCB(9) while meanD jumps well above +2.5 (CI favorable).")

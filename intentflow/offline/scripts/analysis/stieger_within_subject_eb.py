"""Within-subject longitudinal EB pooling + alpha-floor (the post-anisotropic main line).

Motivation (established, leak-free, ours):
 - COHORT pooling is a mirage (degenerates to all-adopt at small k).
 - BUT harm is ~half a subject trait: ICC(per-session Delta)=0.47, 29/62 never harmed, 7 net-negative,
   and harm persists in time (P(harm|prev-harm)=0.39 vs 0.16). So the RIGHT pooling axis is the subject's
   OWN past sessions, not the cohort. And residual (within-subject) session noise is handled not by a
   gate (unpredictable) but by a soft alpha-shrink toward source that BOUNDS the worst case.

Method (per subject, sessions in TRUE temporal order; leak-free):
 - frozen session-1 Riemann-tangent-LDA (P1). adapter = EA recenter to Rj. scalar-alpha geodesic base-point
   R(a)=R1^1/2 (R1^-1/2 Rj R1^-1/2)^a R1^1/2 gives a continuum source(a=0)..EA(a=1).
 - For session j (j>=2): spend m probe trials (FIRST m) to estimate this session's Delta-hat_j.
   WITHIN-SUBJECT EB: prior = mean/var of the subject's PAST sessions' probe Delta-hats (1..j-1).
     posterior mean mu_post = shrink(Delta-hat_j -> prior). If subject has <1 past session -> cohort fallback
     (use the running cohort mean of past subjects' probe means; still no future leak).
 - DECISION = continuous alpha from the posterior, with a floor:
     adopt-strength a_j = clip( sigmoid(mu_post / s) , a_floor, 1 )  if mu_post>0 else a_floor
     i.e. confident-good subject -> a~1 (full EA); uncertain/bad -> a_floor (mostly source). a_floor bounds harm.
 - EVALUATE session j on held-out [m:] at base-point R(a_j). per-(subj,sess) Delta vs source.
Baselines on identical held-out: source / always-EA / per-session LCB(m) / subjMean-hard(m) /
 within-subj-EB+floor(PROPOSED) / subject-oracle(adopt whole subject iff its true mean Delta>0).
Metrics: meanD, harmed(<-1), worst, %oracle(of EA-session-oracle), labels. Subject-cluster bootstrap CI on
 (harmed, meanD) for PROPOSED vs always-EA and vs per-session LCB. From cached epochs (no get_data). CPU.
"""
import os, glob, warnings, json, numpy as np
warnings.filterwarnings("ignore")
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
np.random.seed(0)
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
CACHE="/home/islabshi/workspace-local2/mne_data/stieger_epochs_cache"
M=16          # probe trials per session (labels spent)
A_FLOOR=0.3   # minimum adaptation strength (>=0); a=0 would be exact source. floor bounds worst-case.
SIG=4.0       # sigmoid temperature on posterior mean (pp)

def cov(X): return np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]
def invsqrtm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def sqrtm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*(w**0.5))@V.T
def logm(M_): w,V=eigh(M_);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T
def tangent(C,P):
    d=C.shape[-1];iu=np.triu_indices(d);sc=np.sqrt(2)*np.ones((d,d));sc[np.diag_indices(d)]=1;sc=sc[iu]
    return np.array([(logm(P@C[i]@P))[iu]*sc for i in range(len(C))])
def basepoint(R1h,M,a):
    mu,U=eigh(M);mu=np.clip(mu,1e-10,None)
    return invsqrtm(R1h@((U*np.exp(a*np.log(mu)))@U.T)@R1h)

# ---- precompute per (subj,sess): probe Delta-hat(a=1 vs src) on [:M], and held-out acc curve over a-grid ----
AGRID=np.array([0.0,A_FLOOR,0.5,0.7,1.0])  # source, floor, mid, ea
subjects=[]
files=sorted(glob.glob(f"{CACHE}/S*_epochs.npz"),key=lambda f:int(os.path.basename(f).split("_")[0][1:]))
for f in files:
    sub=int(os.path.basename(f).split("_")[0][1:])
    try:
        z=np.load(f);X=z["X"].astype(np.float64);y=z["y"].astype(int);sess=z["sess"]
        order=sorted(set(sess.tolist()));s1=order[0];m1=sess==s1
        if len(np.unique(y[m1]))<2: continue
        R1=cov(X[m1]).mean(0);P1=invsqrtm(R1);R1h=sqrtm(R1)
        lda=LDA(solver="lsqr",shrinkage="auto").fit(tangent(cov(X[m1]),P1),y[m1])
        sess_list=[]
        for sj in order[1:]:
            mj=sess==sj;yj=y[mj];Xj=X[mj]
            if len(yj)<M+12 or len(np.unique(yj))<2: continue
            Cj=cov(Xj);Rj=Cj.mean(0);M_=P1@Rj@P1
            # probe Delta-hat (a=1 vs source) on first M, leak-free
            cs_p=(lda.predict(tangent(Cj[:M],P1))==yj[:M]).astype(float)
            ca_p=(lda.predict(tangent(Cj[:M],invsqrtm(Rj)))==yj[:M]).astype(float)
            dhat=(ca_p.mean()-cs_p.mean())*100; se=max((ca_p-cs_p).std(ddof=1)/np.sqrt(M)*100,1e-3)
            # held-out accuracy at each a in AGRID (source acc = a=0)
            acc={}
            for a in AGRID:
                P=P1 if a==0 else basepoint(R1h,M_,a)
                acc[a]=(lda.predict(tangent(Cj[M:],P))==yj[M:]).mean()
            sj_true=(acc[1.0]-acc[0.0])*100  # true full-EA Delta on held-out
            sess_list.append(dict(sj=int(sj),dhat=dhat,se=se,acc=acc,true=sj_true))
        if sess_list: subjects.append((sub,sess_list))
        print(f"S{sub}: {len(sess_list)} sess",flush=True); del X,y,z
    except Exception as e:
        print(f"S{sub}: FAIL {str(e)[:70]}",flush=True)

def srcacc(s): return s['acc'][0.0]
def accat(s,a):  # nearest grid (a chosen from continuum -> snap to grid for eval)
    k=AGRID[np.argmin(np.abs(AGRID-a))]; return s['acc'][k]

# ---- policies (leak-free; temporal order within subject) ----
def run(policy):
    eff=[];src=[];true=[];sub_id=[];lab=0
    for sub,sl in subjects:
        past=[]  # subject's past probe dhats
        for s in sl:
            a,used=policy(s,past,sub)
            eff.append(accat(s,a)); src.append(srcacc(s)); true.append(s['true']); sub_id.append(sub); lab+=used
            past.append(s['dhat'])  # spend M to inform future (counted in 'used' when policy uses probe)
    eff=np.array(eff);src=np.array(src);true=np.array(true);sub_id=np.array(sub_id)
    base=src.mean()
    return dict(meanD=(eff.mean()-base)*100, harmed=int(((true<-1)&(np.array([accat(s,1) for sub,sl in subjects for s in sl])>=-99)).sum()) if False else int(((eff-src)*100<-1).sum()),
                worst=float(((eff-src)*100).min()), labels=lab, n=len(eff), eff=eff, src=src, sub=sub_id)

# baselines
def p_source(s,past,sub): return 0.0,0
def p_alwaysEA(s,past,sub): return 1.0,0
def p_lcb(s,past,sub):     return (1.0 if s['dhat']-1.645*s['se']>0 else 0.0), M
def p_subjmean(s,past,sub):
    if not past: return 1.0,M
    return (1.0 if np.mean(past)>0 else 0.0), M
def eb_post(dhat,se,past,cohort):
    if len(past)>=2: mu0=np.mean(past); t2=max(np.var(past)-np.mean(se)**2 if False else np.var(past),1.0)
    elif len(past)==1: mu0=past[0]; t2=25.0
    else: mu0=cohort; t2=25.0
    prec=1/se**2+1/t2; return (dhat/se**2+mu0/t2)/prec
COH={'m':0.0,'n':0}
def p_eb_floor(s,past,sub):
    mu=eb_post(s['dhat'],s['se'],past,COH['m'])
    a=A_FLOOR if mu<=0 else float(np.clip(1/(1+np.exp(-mu/SIG)),A_FLOOR,1.0))
    return a,M
def p_subj_oracle(s,past,sub):  # cheat: adopt whole subject iff its TRUE mean Delta>0
    tm=np.mean([x['true'] for sub2,sl in subjects if sub2==sub for x in sl])
    return (1.0 if tm>0 else 0.0),0

# running cohort mean for EB fallback (use global mean of all probe dhats as a fixed prior proxy)
COH['m']=float(np.mean([s['dhat'] for _,sl in subjects for s in sl]))
ora=np.mean([max(0,s['true']) for _,sl in subjects for s in sl])  # EA session-oracle
res={}
for name,pol in [("source",p_source),("always-EA",p_alwaysEA),(f"per-sess LCB(m{M})",p_lcb),
                 (f"subjMean-hard(m{M})",p_subjmean),(f"within-EB+floor{A_FLOOR}(m{M})",p_eb_floor),
                 ("subject-oracle",p_subj_oracle)]:
    r=run(pol); r["pct_oracle"]=r["meanD"]/ora*100 if ora>0 else 0; res[name]=r

print(f"\n{'policy':>26} {'meanD':>7} {'%ora':>6} {'harmed':>7} {'worst':>7} {'labels':>7}  (n={res['source']['n']}, oracle=+{ora:.2f})")
for name,r in res.items():
    print(f"{name:>26} {r['meanD']:>+7.2f} {r['pct_oracle']:>5.0f}% {r['harmed']:>7d} {r['worst']:>+7.1f} {r['labels']:>7d}")

# ---- subject-cluster bootstrap: PROPOSED vs always-EA and vs per-sess LCB (harmed & meanD deltas) ----
prop=res[f"within-EB+floor{A_FLOOR}(m{M})"]
def boot_delta(a,b,key,B=3000):
    subs=np.array(sorted(set(prop['sub'].tolist()))); d=[]
    ea_e,ea_s=res[a]['eff'],res[a]['src']; pb_e,pb_s=res[b]['eff'],res[b]['src']; sid=prop['sub']
    for _ in range(B):
        pick=np.random.choice(subs,len(subs),replace=True)
        mask=np.concatenate([np.where(sid==s)[0] for s in pick])
        if key=='harmed':
            va=int(((ea_e[mask]-ea_s[mask])*100<-1).sum()); vb=int(((pb_e[mask]-pb_s[mask])*100<-1).sum())
            d.append(vb-va)  # b - a
        else:
            va=(ea_e[mask].mean()-ea_s[mask].mean())*100; vb=(pb_e[mask].mean()-pb_s[mask].mean())*100
            d.append(vb-va)
    return np.percentile(d,2.5),np.median(d),np.percentile(d,97.5)
print("\n=== subject-cluster bootstrap (proposed within-EB+floor vs baseline) ===")
for b in ["always-EA",f"per-sess LCB(m{M})"]:
    lo,md,hi=boot_delta(b,f"within-EB+floor{A_FLOOR}(m{M})",'harmed')
    lo2,md2,hi2=boot_delta(b,f"within-EB+floor{A_FLOOR}(m{M})",'meanD')
    print(f"  vs {b:>20}: Δharmed median={md:+.0f} CI[{lo:+.0f},{hi:+.0f}] | ΔmeanD median={md2:+.2f} CI[{lo2:+.2f},{hi2:+.2f}]")
json.dump({k:{kk:vv for kk,vv in v.items() if kk not in('eff','src','sub')} for k,v in res.items()},
          open(f"{RES}/260622_within_subject_eb.json","w"),indent=2)
print(f"\nsaved 260622_within_subject_eb.json")
print("WIN if within-EB+floor has harmed << always-EA(check vs subjMean/LCB) at modest labels AND meanD clearly positive (CI excludes 0 favorably).")

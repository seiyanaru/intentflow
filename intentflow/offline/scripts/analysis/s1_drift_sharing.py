"""S1 make-or-break (GPT 'Drift-Set' core): is the SESSION-DRIFT CORRECTION shared
across subjects, in a cross-subject-comparable space (Riemann tangent, channel space)?

The only part of drift that needs OTHER subjects' LABELS (=new info beyond the single
unlabeled test session) is the CLASS-DIFFERENTIAL drift: how the right-hand class centroid
moves RELATIVE to the left-hand class centroid from session_T -> session_E. The marginal
(class-agnostic) drift is self-estimable unsupervised (=EA), so it is NOT new info.

Decisive measurement (label-free transport is only possible if this is high):
  rho_i = tangent_mean(eval_i, classR) - tangent_mean(eval_i, classL)
        - [ tangent_mean(train_i, classR) - tangent_mean(train_i, classL) ]
  = the subject-i class-differential drift vector.
Cross-subject cosine(rho_i, rho_j): high (shared mode) => S1 ALIVE; ~0 (idiosyncratic) => S1 DEAD.
Also: marginal drift cosine (context) + a LOO transport check (does others'-mean rho correct target?).
No EA (EA removes exactly the marginal session shift). Common tangent ref = grand-mean cov.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from datamodules.bcic4_2a import BCICIV2a
from datamodules.bcic4_2b import BCICIV2b

b8,a8 = butter(4,[8/125.,30/125.],btype="band")
def bp(X): return filtfilt(b8,a8,X,axis=-1)
def covs(X):  # (n,C,T)->(n,C,C) normalized per-trial covariance
    Xf=bp(X.astype(np.float64)); return np.einsum('nct,ndt->ncd',Xf,Xf)/Xf.shape[-1]
def invsqrt(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*(w**-0.5))@V.T
def tangent(C,P):
    c=C.shape[0]; iu=np.triu_indices(c); s=np.sqrt(2)*np.ones((c,c)); np.fill_diagonal(s,1.)
    A=P@C@P; w,V=eigh(A); w=np.clip(w,1e-10,None); L=(V*np.log(w))@V.T
    return (L*s)[iu]
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    return X, np.array([int(ds[i][1]) for i in range(len(ds))])
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def run(name, dm_cls, prep, clsR, clsL, sids=range(1,10)):
    print(f"\n===== {name}: S1 cross-subject drift-sharing (tangent space, no EA) =====")
    # pass1: load covs, build common ref from all train covs
    data={}; allC=[]
    for s in sids:
        try:
            dm=dm_cls(prep, s); dm.setup() if name=="2b" else dm.setup("fit")
            Xtr,ytr=getxy(dm.train_dataset); Xte,yte=getxy(dm.test_dataset)
            if Xtr.shape[-1]>1000: Xtr=Xtr[...,:1000]; Xte=Xte[...,:1000]
            Ctr=covs(Xtr); Cte=covs(Xte); data[s]=(Ctr,ytr,Cte,yte); allC.append(Ctr.mean(0))
        except Exception as e:
            print(f"  S{s} load FAIL: {str(e)[:60]}")
    P=invsqrt(np.mean(allC,0))  # common tangent reference
    marg={}; rho={}
    for s,(Ctr,ytr,Cte,yte) in data.items():
        Ttr=np.array([tangent(C,P) for C in Ctr]); Tte=np.array([tangent(C,P) for C in Cte])
        marg[s]=Tte.mean(0)-Ttr.mean(0)
        def cd(T,y): return T[y==clsR].mean(0)-T[y==clsL].mean(0)  # class-differential axis
        rho[s]=cd(Tte,yte)-cd(Ttr,ytr)  # how the R-vs-L axis drifts T->E (needs labels)
        data[s]=(Ttr,ytr,Tte,yte)
    S=list(data.keys())
    def pair_cos(D):
        v=[cos(D[i],D[j]) for a,i in enumerate(S) for j in S[a+1:]]; return np.mean(v),np.std(v)
    mm,ms=pair_cos(marg); rm,rs=pair_cos(rho)
    print(f"  marginal-drift pairwise cosine     = {mm:+.3f} ± {ms:.3f}  (self-estimable=EA, context only)")
    print(f"  CLASS-DIFFERENTIAL drift cosine     = {rm:+.3f} ± {rs:.3f}  <== S1 NEW-INFO make-or-break")
    # LOO transport: subtract others'-mean class-differential correction; does it help R/L separation?
    print("  --- LOO transport gate (others'-mean rho applied to target) ---")
    base=[]; selfc=[]; others=[]
    for j in S:
        Ttr,ytr,Tte,yte=data[j]
        mR=Ttr[ytr==clsR].mean(0); mL=Ttr[ytr==clsL].mean(0); w=mR-mL; thr=0.5*(mR+mL)@w  # train LDA-lite axis
        m=np.isin(yte,[clsR,clsL]); Te=Tte[m]; ye=yte[m]
        def acc(shift):
            sc=(Te-shift)@w; pred=np.where(sc>thr,clsR,clsL);
            return max((pred==ye).mean(),(pred[::-1]==ye[::-1]).mean() if False else (pred==ye).mean())*100
        base.append(acc(0.0))
        selfc.append(acc(marg[j]))                                   # self unsupervised marginal (EA-like)
        oth=np.mean([rho[k] for k in S if k!=j],0)                   # others' labeled class-diff drift
        others.append(acc(0.5*oth))                                   # transported correction (0.5 step)
    print(f"  R/L acc  source={np.mean(base):.1f}  +self-marginal={np.mean(selfc):.1f}  +others-rho={np.mean(others):.1f}")
    print(f"  VERDICT: class-diff cosine>~0.4 AND +others-rho>source => S1 ALIVE (drift shared, others' labels help).")
    print(f"           cosine~0 OR others-rho<=source => S1 DEAD (drift idiosyncratic).")
    return rm, np.mean(others)-np.mean(base)

DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep2a=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,
            num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
run("2a", BCICIV2a, prep2a, clsR=1, clsL=0)   # class0=left hand, class1=right hand
prep2b=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,
            num_workers=0,z_scale=True,data_path=None,eval_label_path=None)
run("2b", BCICIV2b, prep2b, clsR=1, clsL=0)

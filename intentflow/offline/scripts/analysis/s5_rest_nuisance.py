"""S5 make-or-break (GPT): use the PRE-CUE / REST window as a label-free NEGATIVE CONTROL
for session nuisance, and whiten the MI window by the REST covariance (not the MI covariance
as EA does). Distinct from the dead ERD anchor: we do NOT recover class from rest; we only
estimate the CLASS-INVARIANT session nuisance from rest and remove it.

Decisive question: is REST-whitening (a) different from EA (MI-whitening) and (b) better,
especially on the stable subjects EA HURTS? If REST ~= EA (whitening matrices ~identical,
acc ~same) => S5 reduces to EA => DEAD. If REST helps where EA hurts => S5 ALIVE (new info).

Cross-session: train tangent-LDA on session_T MI, test on session_E MI. R/L hand only.
EEG only (drop EOG). Bandpass 8-30. Common tangent ref = grand-mean MI cov.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA","/home/islabshi/workspace-local2/mne_data")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, mne
mne.set_log_level("ERROR")
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from moabb.datasets import BNCI2014_001

b8,a8=butter(4,[8/125.,30/125.],btype="band")
def bp(X): return filtfilt(b8,a8,X.astype(np.float64),axis=-1)
def cov1(X): return np.einsum('ct,dt->cd',X,X)/X.shape[-1]
def covs(E): Ef=bp(E); return np.array([cov1(x) for x in Ef])
def invsqrt(M): w,V=eigh(M); w=np.clip(w,1e-10,None); return (V*(w**-0.5))@V.T
def tangent(C,P):
    c=C.shape[0]; iu=np.triu_indices(c); s=np.sqrt(2)*np.ones((c,c)); np.fill_diagonal(s,1.)
    A=P@C@P; w,V=eigh(A); w=np.clip(w,1e-10,None); L=(V*np.log(w))@V.T
    return (L*s)[iu]

def epochs_two_windows(raw, ev, eid):
    # MI = onset+[0.5,3.5]; REST = onset+[-2.0,-0.5] (pre-cue fixation). 250Hz.
    mi  = mne.Epochs(raw,ev,eid,tmin=0.5,tmax=3.5,baseline=None,preload=True,picks="eeg")
    rest= mne.Epochs(raw,ev,eid,tmin=-2.0,tmax=-0.5,baseline=None,preload=True,picks="eeg")
    return mi, rest

def whiten_apply(Cs, R):  # whiten each cov by ref R: R^-1/2 C R^-1/2
    P=invsqrt(R); return np.einsum('ij,njk,kl->nil',P,Cs,P)

ds=BNCI2014_001()
print("===== 2a S5: REST-whitening vs EA vs source (cross-session R/L) =====")
print(f"{'S':>3} {'restcls':>7} {'source':>7} {'EA':>6} {'REST':>6} {'EA-cos-REST':>11}")
acc_s=[];acc_e=[];acc_r=[];harmed_ea=[];harmed_rest=[]
for s in range(1,10):
    try:
        d=ds.get_data(subjects=[s])[s]
        sess=sorted(d.keys())  # ['0train','1test'] expected
        def load(sk):
            mis=[];rss=[];ys=[]
            for run,raw in d[sk].items():
                raw.pick_types(eeg=True,eog=False,stim=False)
                ev,eid=mne.events_from_annotations(raw)
                # keep only left/right hand
                lr={k:v for k,v in eid.items() if 'left' in k.lower() or 'right' in k.lower()}
                if not lr: continue
                mi,rest=epochs_two_windows(raw,ev,lr)
                lab=np.array([1 if 'right' in [k for k,v in lr.items() if v==c][0].lower() else 0 for c in mi.events[:,2]])
                mis.append(mi.get_data()); rss.append(rest.get_data()); ys.append(lab)
            return np.concatenate(mis),np.concatenate(rss),np.concatenate(ys)
        MiT,RsT,yT=load(sess[0]); MiE,RsE,yE=load(sess[1])
        CmiT=covs(MiT);CmiE=covs(MiE);CrsT=covs(RsT);CrsE=covs(RsE)
        # sanity: does rest carry class info? (should be ~chance)
        P0=invsqrt(np.concatenate([CrsT,CrsE]).mean(0))
        TrsT=np.array([tangent(C,P0) for C in CrsT])
        wr=TrsT[yT==1].mean(0)-TrsT[yT==0].mean(0); thr=0.5*(TrsT[yT==1].mean(0)+TrsT[yT==0].mean(0))@wr
        restcls=max(((TrsT@wr>thr).astype(int)==yT).mean(),((TrsT@wr<thr).astype(int)==yT).mean())*100
        def xsess(CT,CE):  # train tangent-LDA on session_T, test session_E (R/L)
            P=invsqrt(np.concatenate([CT,CE]).mean(0))
            TT=np.array([tangent(C,P) for C in CT]); TE=np.array([tangent(C,P) for C in CE])
            w=TT[yT==1].mean(0)-TT[yT==0].mean(0); th=0.5*(TT[yT==1].mean(0)+TT[yT==0].mean(0))@w
            pr=(TE@w>th).astype(int); return max((pr==yE).mean(),((1-pr)==yE).mean())*100
        a_src=xsess(CmiT,CmiE)
        a_ea =xsess(whiten_apply(CmiT,CmiT.mean(0)), whiten_apply(CmiE,CmiE.mean(0)))   # EA: MI-cov whiten
        a_rs =xsess(whiten_apply(CmiT,CrsT.mean(0)), whiten_apply(CmiE,CrsE.mean(0)))   # S5: REST-cov whiten
        # are EA and REST whitening matrices ~identical? (Frobenius cosine on session_E)
        Pea=invsqrt(CmiE.mean(0)); Prs=invsqrt(CrsE.mean(0))
        eacos=float((Pea*Prs).sum()/(np.linalg.norm(Pea)*np.linalg.norm(Prs)+1e-12))
        acc_s.append(a_src);acc_e.append(a_ea);acc_r.append(a_rs)
        harmed_ea.append(a_ea-a_src);harmed_rest.append(a_rs-a_src)
        print(f"S{s:>2} {restcls:>7.1f} {a_src:>7.1f} {a_ea:>6.1f} {a_rs:>6.1f} {eacos:>11.3f}")
    except Exception as e:
        print(f"S{s:>2} FAIL {str(e)[:55]}")
import numpy as np
print(f"{'MEAN':>3} {'':>7} {np.mean(acc_s):>7.1f} {np.mean(acc_e):>6.1f} {np.mean(acc_r):>6.1f}")
print(f"\nEA  Δ vs source: mean {np.mean(harmed_ea):+.1f}  (subjects harmed: {sum(1 for x in harmed_ea if x<-1)})")
print(f"REST Δ vs source: mean {np.mean(harmed_rest):+.1f} (subjects harmed: {sum(1 for x in harmed_rest if x<-1)})")
print("VERDICT: restcls~50 (rest is class-free, good control). S5 ALIVE if REST>source AND REST avoids EA's harmed")
print("         subjects AND EA-cos-REST<~0.95 (genuinely different). S5 DEAD if REST~=EA (cos~1, same acc).")

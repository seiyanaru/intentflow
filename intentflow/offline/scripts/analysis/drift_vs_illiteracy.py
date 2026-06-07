"""Diagnostic: is each subject's low accuracy DRIFT (recoverable) or LOW-SNR/illiteracy?
Within-session_T 5-fold CV accuracy (same-day separability) vs cross-session T->E accuracy.
Family-agnostic signal proxy = EA-Riemann tangent + shrinkage LDA (no GPU).
  within-CV high, cross low  -> DRIFT (signal exists same-day, lost across days)
  within-CV low              -> ILLITERACY (weak signal even same-day; label-free can't rescue)
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from datamodules.bcic4_2a import BCICIV2a
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
b,a=butter(4,[8/125.,30/125.],btype="band")
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    return X,np.array([int(ds[i][1]) for i in range(len(ds))])
def invsqrtm(M):w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*(w**-0.5))@V.T
def logm(M):w,V=eigh(M);w=np.clip(w,1e-10,None);return (V*np.log(w))@V.T
def covs(X):
    Xf=filtfilt(b,a,X,axis=-1);return np.einsum('nct,ndt->ncd',Xf,Xf)/Xf.shape[-1]
def tangent(Cs,P):
    c=Cs.shape[1];iu=np.triu_indices(c);s=np.sqrt(2)*np.ones((c,c));np.fill_diagonal(s,1.)
    return np.array([(logm(P@C@P)*s)[iu] for C in Cs])
def riem_fit_pred(Xtr,ytr,Xte):
    Ctr=covs(Xtr);P=invsqrtm(Ctr.mean(0));Vtr=tangent(Ctr,P)
    lda=LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto").fit(Vtr,ytr)
    Cte=covs(Xte);Pte=invsqrtm(Cte.mean(0))   # unsupervised EA ref on the eval set itself
    return lda.predict(tangent(Cte,Pte))
# reference: deep source E-accuracy (cached) for context
exp=np.load("/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",allow_pickle=True)
si=[str(x) for x in exp["experts"].tolist()].index("source")
Tacc=[(exp["probs"][i,si].argmax(1)==exp["labels"][i]).mean()*100 for i in range(9)]
print(f"{'subj':>4} {'deepE':>6} {'Riem_withinCV':>13} {'Riem_crossE':>12} {'drop(within-cross)':>18} {'verdict':>12}")
rows=[]
for sid in range(1,10):
    dm=BCICIV2a(prep,sid);dm.setup()
    Xtr,ytr=getxy(dm.train_dataset);Xte,yte=getxy(dm.test_dataset)
    # within-session_T 5-fold CV (EA ref per train fold)
    skf=StratifiedKFold(5,shuffle=True,random_state=0);accs=[]
    for tr,va in skf.split(Xtr,ytr):
        pred=riem_fit_pred(Xtr[tr],ytr[tr],Xtr[va]);accs.append((pred==ytr[va]).mean()*100)
    within=np.mean(accs)
    cross=(riem_fit_pred(Xtr,ytr,Xte)==yte).mean()*100
    rows.append((sid,Tacc[sid-1],within,cross))
for sid,de,within,cross in sorted(rows,key=lambda r:r[1]):
    drop=within-cross
    verdict="ILLITERACY" if within<65 else ("DRIFT" if drop>10 else "mixed/ok")
    print(f"S{sid:>3} {de:>6.1f} {within:>13.1f} {cross:>12.1f} {drop:>18.1f} {verdict:>12}")
print("\nverdict rule: within<65% -> ILLITERACY(救えない); within高&drop>10 -> DRIFT(救える); else mixed")

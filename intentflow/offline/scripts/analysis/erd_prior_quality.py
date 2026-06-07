"""Decisive label-free test of the physiology-anchored direction (BPE/ERD-head-repair):
how good is the ERD lateralization index (LI) ALONE at predicting left/right-hand MI,
per subject, on 2a (L/R-hand subset) and 2b (binary)? This is the CEILING on what a
label-free ERD pseudo-target can contribute to head recalibration. No labels used to
build LI (pure physiology); labels only score it.
LI = log mu/beta power(right sensorimotor cluster) - log power(left cluster).
Run with intentflow conda env.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ".")
import numpy as np
from scipy.signal import butter, filtfilt
b,a=butter(4,[8/125.,30/125.],btype="band")
def power(X,idx):  # X (n,C,T) -> mean log band-power over channel idx set
    Xf=filtfilt(b,a,X[:,idx,:],axis=-1); return np.log(np.var(Xf,axis=-1)+1e-8).mean(1)
def li_acc(X,y,right,left,cls=None):
    LI=power(X,right)-power(X,left)            # >0 ~ right-hand ERD over left cortex
    if cls is not None:
        m=np.isin(y,cls); X2=LI[m]; y2=y[m]
        # map to the two classes by best sign
        lo,hi=sorted(cls)
        p1=np.where(X2>np.median(X2),hi,lo); p2=np.where(X2>np.median(X2),lo,hi)
        return max((p1==y2).mean(),(p2==y2).mean())*100, m.mean()*100
    else:
        p1=(LI>np.median(LI)).astype(int);
        return max((p1==y).mean(),((1-p1)==y).mean())*100, 100.0
def getxy(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float64)
    return X,np.array([int(ds[i][1]) for i in range(len(ds))])

# ---- 2a (22ch, 4-class; LI only separates class0=left vs class1=right hand) ----
from datamodules.bcic4_2a import BCICIV2a
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
prep=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
# montage: C5,C3,C1=6,7,8 (left) ; C2,C4,C6=10,11,12 (right)
print("=== 2a (22ch): ERD-LI accuracy on left/right-HAND trials (class0 vs class1) ===")
print(f"{'subj':>4} {'LI_acc(LRhand)':>14} {'LRhand_frac':>11}")
a2=[]
for sid in range(1,10):
    dm=BCICIV2a(prep,sid);dm.setup();X,y=getxy(dm.test_dataset)
    acc,fr=li_acc(X,y,[10,11,12],[6,7,8],cls=[0,1]);a2.append(acc)
    print(f"S{sid:>3} {acc:>14.1f} {fr:>11.1f}")
print(f"{'MEAN':>4} {np.mean(a2):>14.1f}   (これは L/R hand 2値での話; 4classでは feet/tongue は救えない)")

# ---- 2b (3ch C3,Cz,C4; binary L/R hand) ----
from datamodules.bcic4_2b import BCICIV2b
prep2=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=None,eval_label_path=None)
print("\n=== 2b (3ch): ERD-LI accuracy (binary), C4 vs C3 ===")
print(f"{'subj':>4} {'LI_acc':>8}")
a2b=[]
for sid in range(1,10):
    try:
        dm=BCICIV2b(prep2,sid);dm.setup();X,y=getxy(dm.test_dataset)
        acc,_=li_acc(X,y,[2],[0]);a2b.append(acc)  # C4=2, C3=0
        print(f"S{sid:>3} {acc:>8.1f}")
    except Exception as e:
        print(f"S{sid:>3} ERR {str(e)[:50]}")
if a2b: print(f"{'MEAN':>4} {np.mean(a2b):>8.1f}")
print("\nVERDICT: LI_acc >> chance(50%) and decent on weak subjects => physiology prior carries real")
print("label-free class info => head-repair direction is ALIVE. If ~chance => DEAD (esp. where needed).")

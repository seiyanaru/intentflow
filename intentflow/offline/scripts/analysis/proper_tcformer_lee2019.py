"""PROPER TCFormer training on Lee2019 (match the 2a +0.8 pipeline): interaug
segment-recombination + Euclidean Alignment (per session) + F1=32 + kernels[20,32,64]
+ warmup(20)+cosine, lr 9e-4, wd 1e-3, batch 48. Verify accuracy recovers (vs the
under-trained 56.5). Default: 5 subjects. Set SUBS env for more.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn
from scipy.linalg import eigh
from models.tcformer.tcformer import TCFormerModule
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"; torch.manual_seed(0); np.random.seed(0)
prm=LeftRightImagery(resample=250, fmin=1, fmax=45); ds=Lee2019_MI()
def ea(X):  # per-session Euclidean Alignment (whiten by mean covariance)
    Cs=np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]; R=Cs.mean(0)
    w,V=eigh(R);w=np.clip(w,1e-10,None);P=(V*(w**-0.5))@V.T
    return np.einsum('ij,njt->nit',P,X)
def zsc(X): return ((X-X.mean(-1,keepdims=True))/(X.std(-1,keepdims=True)+1e-7)).astype(np.float32)
def interaug(X,y,n_per_class,n_seg=8):
    C,T=X.shape[1],X.shape[2];seg=T//n_seg;aX=[];aY=[]
    for c in np.unique(y):
        Xc=X[y==c]
        for _ in range(n_per_class):
            new=np.empty((C,T),dtype=X.dtype)
            for s in range(n_seg):
                k=np.random.randint(len(Xc)); a=s*seg; b=T if s==n_seg-1 else (s+1)*seg
                new[:,a:b]=Xc[k][:,a:b]
            aX.append(new);aY.append(c)
    return np.array(aX),np.array(aY)
def train(Xtr,ytr,C,T,epochs=600):
    net=TCFormerModule(n_channels=C,n_classes=2,F1=32,temp_kernel_lengths=(20,32,64)).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=9e-4,weight_decay=1e-3)
    warm=20
    def lr_at(e): return (e+1)/warm if e<warm else 0.5*(1+np.cos(np.pi*(e-warm)/(epochs-warm)))
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev);n=len(yt)
    for e in range(epochs):
        for g in opt.param_groups: g['lr']=9e-4*lr_at(e)
        aX,aY=interaug(Xtr,ytr,n_per_class=n//2)  # ~n augmented samples
        Xa=torch.tensor(np.concatenate([Xtr,aX])).to(dev);ya=torch.tensor(np.concatenate([ytr,aY])).long().to(dev)
        p=torch.randperm(len(ya),device=dev)
        net.train()
        for i in range(0,len(ya),48):
            idx=p[i:i+48];loss=nn.functional.cross_entropy(net(Xa[idx]),ya[idx])
            opt.zero_grad();loss.backward();opt.step()
    net.eval();return net
@torch.no_grad()
def acc_of(net,X,y):
    pr=[]
    for i in range(0,len(X),64): pr.append(net(torch.tensor(X[i:i+64]).to(dev)).argmax(1).cpu().numpy())
    return (np.concatenate(pr)==y).mean()*100
SUBS=[int(x) for x in os.environ.get("SUBS","1,2,3,4,5").split(",")]
print(f"PROPER training (interaug+EA+F1=32+600ep), subjects={SUBS}")
accs=[]
for s in SUBS:
    X,y,meta=prm.get_data(dataset=ds,subjects=[s]);y=(y=='right_hand').astype(int);ses=meta['session'].values
    tr=ses=='0';te=ses=='1'
    Xtr=zsc(ea(X[tr].astype(np.float64)));Xte=zsc(ea(X[te].astype(np.float64)))
    net=train(Xtr,y[tr],X.shape[1],X.shape[2])
    a=acc_of(net,Xte,y[te]);accs.append(a)
    print(f"S{s}: cross-session acc = {a:.1f}",flush=True);del net;torch.cuda.empty_cache()
print(f"\nMEAN={np.mean(accs):.1f}  (under-trained run was 56.5; PROPER should be ~65-75 if config fixed)")

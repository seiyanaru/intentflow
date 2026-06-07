"""E1 PILOT (F7-safe decoder go/no-go): does a CROSS-SUBJECT pretrained TCFormer reach
~70% cross-session on HELD-OUT Lee2019 subjects, vs the known ~58% from-scratch (F7)?
If yes => the F7-safe substrate works and CLAIM-3 monitor can be judged on a strong decoder.

Protocol (pilot, leave-several-out approximation of LOSO to keep it cheap):
  pretrain pool = POOL subjects' session0 (EA per-subject, z-score), train TCFormer cross-subject.
  for each held-out TGT subject (NOT in pool):
    EA session0 (train) + session1 (test) separately, z-score.
    (a) FROZEN backbone -> penultimate feats -> LDA on session0 -> eval session1   [cheapest; monitor uses these feats]
    (b) HEAD-ONLY finetune (freeze backbone, train tcn_head.classifier) on session0 -> eval session1
  report per-target acc + mean; compare to the established from-scratch ~58.3% (F7).
Also: clusterability(silhouette) of session1 frozen feats vs acc (preliminary CLAIM-3 signal, small n).
Run: intentflow env, GPU. Env: POOL="1..16", TGT="17..22", EPOCHS=150.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA","/home/islabshi/workspace-local2/mne_data")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
from models.tcformer.tcformer import TCFormerModule
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda","need GPU"
torch.manual_seed(0); np.random.seed(0)
prm=LeftRightImagery(resample=250, fmin=1, fmax=45); ds=Lee2019_MI()

def ea(X):
    Cs=np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]; R=Cs.mean(0)
    w,V=eigh(R); w=np.clip(w,1e-10,None); P=(V*(w**-0.5))@V.T
    return np.einsum('ij,njt->nit',P,X)
def zsc(X): return ((X-X.mean(-1,keepdims=True))/(X.std(-1,keepdims=True)+1e-7)).astype(np.float32)
def load(sub, sess):  # sess '0'/'1' -> (X ea+zsc, y)
    X,y,meta=prm.get_data(dataset=ds,subjects=[sub]); y=(y=='right_hand').astype(int)
    m=meta['session'].values==sess; return zsc(ea(X[m].astype(np.float64))), y[m]
def interaug(X,y,n_per_class,n_seg=8):
    C,T=X.shape[1],X.shape[2]; seg=T//n_seg; aX=[]; aY=[]
    for c in np.unique(y):
        Xc=X[y==c]
        for _ in range(n_per_class):
            new=np.empty((C,T),dtype=X.dtype)
            for s in range(n_seg):
                k=np.random.randint(len(Xc)); a=s*seg; b=T if s==n_seg-1 else (s+1)*seg
                new[:,a:b]=Xc[k][:,a:b]
            aX.append(new); aY.append(c)
    return np.array(aX),np.array(aY)
def new_net(C):
    return TCFormerModule(n_channels=C,n_classes=2,F1=32,temp_kernel_lengths=(20,32,64)).to(dev)
def train(net,Xtr,ytr,epochs,lr=9e-4,head_only=False):
    if head_only:
        for p in net.parameters(): p.requires_grad=False
        for p in net.tcn_head.classifier.parameters(): p.requires_grad=True
        params=[p for p in net.parameters() if p.requires_grad]; aug=False
    else:
        params=net.parameters(); aug=True
    opt=torch.optim.Adam(params,lr=lr,weight_decay=1e-3); warm=min(20,epochs//5)
    lr_at=lambda e:(e+1)/warm if e<warm else 0.5*(1+np.cos(np.pi*(e-warm)/(epochs-warm)))
    n=len(ytr)
    for e in range(epochs):
        for g in opt.param_groups: g['lr']=lr*lr_at(e)
        if aug:
            aX,aY=interaug(Xtr,ytr,n_per_class=n//2)
            Xe=np.concatenate([Xtr,aX]); ye=np.concatenate([ytr,aY])
        else: Xe,ye=Xtr,ytr
        Xt=torch.tensor(Xe).to(dev); yt=torch.tensor(ye).long().to(dev)
        p=torch.randperm(len(yt),device=dev); net.train()
        for i in range(0,len(yt),64):
            idx=p[i:i+64]; loss=nn.functional.cross_entropy(net(Xt[idx]),yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net
cap={}
def feats_logits(net,X):
    h=net.tcn_head.classifier.register_forward_hook(lambda m,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach()))
    F=[];L=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            o=net(torch.tensor(X[i:i+64]).to(dev)); L.append(o.cpu().numpy())
            Ff=cap['f']; F.append((Ff[:,:,0] if Ff.ndim==3 else Ff).cpu().numpy())
    h.remove(); return np.concatenate(F),np.concatenate(L)

def parse(s,d): return [int(x) for x in os.environ.get(s,d).split(",")]
POOL=parse("POOL",",".join(str(i) for i in range(1,17)))
TGT =parse("TGT", ",".join(str(i) for i in range(17,23)))
EP=int(os.environ.get("EPOCHS","150"))
print(f"POOL(pretrain)={POOL}  TGT(held-out)={TGT}  pretrain_epochs={EP}")

# ---- cross-subject pretrain ----
print("loading pool...", flush=True)
Xp=[];yp=[]
for s in POOL:
    try: X,y=load(s,'0'); Xp.append(X); yp.append(y)
    except Exception as e: print(f"  pool S{s} FAIL {str(e)[:50]}")
Xp=np.concatenate(Xp); yp=np.concatenate(yp); C=Xp.shape[1]
print(f"pool trials={len(yp)} ch={C}; pretraining...", flush=True)
net=train(new_net(C),Xp,yp,EP)
torch.save(net.state_dict(), "/tmp/lee2019_xsubj_pretrain.pt")
sd=net.state_dict()

# ---- evaluate held-out targets ----
print(f"\n{'TGT':>4} {'frozenLDA':>9} {'headFT':>7} {'silhouette':>10}")
fa=[];ha=[];clus=[]
for s in TGT:
    try:
        X0,y0=load(s,'0'); X1,y1=load(s,'1')
    except Exception as e: print(f"S{s} load FAIL {str(e)[:50]}"); continue
    # (a) frozen backbone + LDA
    net.load_state_dict(sd); F0,_=feats_logits(net,X0); F1,_=feats_logits(net,X1)
    a_frz=(LDA(solver="lsqr",shrinkage="auto").fit(F0,y0).predict(F1)==y1).mean()*100
    Z=(F1-F1.mean(0))/(F1.std(0)+1e-8)
    sil=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))
    # (b) head-only finetune
    net.load_state_dict(sd); net=train(net,X0,y0,60,lr=5e-3,head_only=True)
    _,L1=feats_logits(net,X1); a_hft=(L1.argmax(1)==y1).mean()*100
    fa.append(a_frz); ha.append(a_hft); clus.append(sil)
    print(f"S{s:>3} {a_frz:>9.1f} {a_hft:>7.1f} {sil:>10.3f}", flush=True)
print(f"{'MEAN':>4} {np.mean(fa):>9.1f} {np.mean(ha):>7.1f}")
print(f"\nfrom-scratch baseline (established, F7): ~58.3%")
if len(clus)>=4:
    print(f"prelim clusterability->acc(headFT) Spearman = {spearmanr(clus,ha).correlation:+.3f} (n={len(clus)}, small)")
print("VERDICT: if frozenLDA/headFT MEAN >=~68-70% => F7 DODGED, E1 full run is GO.")
print("         if <60% => decoder still weak, fix protocol before scaling.")

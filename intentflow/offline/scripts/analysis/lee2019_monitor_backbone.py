"""E5 (backbone-agnosticism) + E4(a) (monitor beats baselines, subject-level), Lee2019 n=54.
Generalize the E1 strong cross-subject decoder to BACKBONE in {tcformer, eegnet}. 3-fold,
all 54 held-out once. Per held-out subject, frozen backbone -> penultimate feats:
  acc      = LDA(sess0 feats)->sess1 acc      (cross-session)
  sacc     = LDA self-fit sess0 acc           (variance/source control)
  sil      = clusterability (silhouette KMeans k=2, z-scored sess1 feats)   [the monitor]
  conf     = mean max-softmax on sess1        [baseline predictor 1]
  negent   = -mean predictive entropy sess1   [baseline predictor 2]
  negmaha  = -median min-source-class Mahalanobis of sess1 feats [baseline predictor 3]
Then per-backbone: Spearman(predictor, acc) for sil/conf/negent/negmaha => does clusterability
WIN at predicting per-subject reliability? (E4a). And does sil-Spearman hold on EEGNet? (E5).
Saves per-subject JSON + per-trial npz (for trial-level AURC later).
Run: intentflow env, GPU. Env BACKBONE=eegnet|tcformer, EPOCHS=150.
"""
import os, sys, warnings, json
warnings.filterwarnings("ignore")
os.environ.setdefault("MNE_DATA","/home/islabshi/workspace-local2/mne_data")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr, rankdata
from models.tcformer.tcformer import TCFormerModule
# eegnet.py references external 'channel_attention.utils.weight_initialization';
# the identical glorot init exists locally (tcformer uses it). Stub the missing module to it.
import types as _t
from utils.weight_initialization import glorot_weight_zero_bias as _glorot
_m=_t.ModuleType("channel_attention.utils.weight_initialization"); _m.glorot_weight_zero_bias=_glorot
sys.modules.setdefault("channel_attention",_t.ModuleType("channel_attention"))
sys.modules.setdefault("channel_attention.utils",_t.ModuleType("channel_attention.utils"))
sys.modules["channel_attention.utils.weight_initialization"]=_m
from models.tcformer.eegnet import EEGNetModule
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"
torch.manual_seed(0); np.random.seed(0)
BK=os.environ.get("BACKBONE","eegnet"); EP=int(os.environ.get("EPOCHS","150"))
prm=LeftRightImagery(resample=250, fmin=1, fmax=45); ds=Lee2019_MI()
def ea(X):
    Cs=np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]; R=Cs.mean(0)
    w,V=eigh(R); w=np.clip(w,1e-10,None); P=(V*(w**-0.5))@V.T
    return np.einsum('ij,njt->nit',P,X)
def zsc(X): return ((X-X.mean(-1,keepdims=True))/(X.std(-1,keepdims=True)+1e-7)).astype(np.float32)
_c={}
def load(sub):
    if sub in _c: return _c[sub]
    X,y,meta=prm.get_data(dataset=ds,subjects=[sub]); y=(y=='right_hand').astype(int); ses=meta['session'].values
    o={s:(zsc(ea(X[ses==s].astype(np.float64))), y[ses==s]) for s in ['0','1']}; _c[sub]=o; return o
def interaug(X,y,npc,n_seg=8):
    C,T=X.shape[1],X.shape[2]; seg=T//n_seg; aX=[];aY=[]
    for c in np.unique(y):
        Xc=X[y==c]
        for _ in range(npc):
            new=np.empty((C,T),dtype=X.dtype)
            for s in range(n_seg):
                k=np.random.randint(len(Xc)); a=s*seg; b=T if s==n_seg-1 else (s+1)*seg
                new[:,a:b]=Xc[k][:,a:b]
            aX.append(new);aY.append(c)
    return np.array(aX),np.array(aY)
def build(C,T):
    if BK=="tcformer": return TCFormerModule(n_channels=C,n_classes=2,F1=32,temp_kernel_lengths=(20,32,64)).to(dev)
    return EEGNetModule(n_channels=C,n_classes=2,input_window_samples=T).to(dev)
def head_module(net): return net.tcn_head.classifier if BK=="tcformer" else net.classifier
def pretrain(Xtr,ytr,C,T,epochs=EP,lr=9e-4):
    net=build(C,T); opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-3); warm=20
    lr_at=lambda e:(e+1)/warm if e<warm else 0.5*(1+np.cos(np.pi*(e-warm)/(epochs-warm)))
    n=len(ytr)
    for e in range(epochs):
        for g in opt.param_groups: g['lr']=lr*lr_at(e)
        aX,aY=interaug(Xtr,ytr,n//2); Xe=np.concatenate([Xtr,aX]); ye=np.concatenate([ytr,aY])
        Xt=torch.tensor(Xe).to(dev); yt=torch.tensor(ye).long().to(dev)
        p=torch.randperm(len(yt),device=dev); net.train()
        for i in range(0,len(yt),64):
            idx=p[i:i+64]; loss=nn.functional.cross_entropy(net(Xt[idx]),yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net
cap={}
def emb_prob(net,X):
    hm=head_module(net)
    h=hm.register_forward_hook(lambda m,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach()))
    F=[];P=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            o=net(torch.tensor(X[i:i+64]).to(dev)); P.append(torch.softmax(o,1).cpu().numpy())
            f=cap['f']
            if f.ndim==4: f=f.mean(dim=(2,3))     # eegnet (B,F2,1,out)->(B,F2)
            elif f.ndim==3: f=f[:,:,0]            # tcformer (B,d,1)->(B,d)
            F.append(f.cpu().numpy())
    h.remove(); return np.concatenate(F),np.concatenate(P)
def maha_scores(F0,y0,F1):
    cls=np.unique(y0); M=np.stack([F0[y0==c].mean(0) for c in cls])
    resid=np.concatenate([F0[y0==c]-F0[y0==c].mean(0) for c in cls],0)
    P=np.linalg.pinv(LedoitWolf().fit(resid).covariance_)
    return np.min(np.stack([np.einsum('ij,jk,ik->i',F1-M[c],P,F1-M[c]) for c in range(len(cls))],1),1)
def partial_spear(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    res=lambda a,b:(a-np.c_[np.ones_like(b),b]@np.linalg.lstsq(np.c_[np.ones_like(b),b],a,rcond=None)[0])
    return float(np.corrcoef(res(rx,rz),res(ry,rz))[0,1])

SUBS=list(range(1,55)); K=3; folds=[SUBS[i::K] for i in range(K)]
OUTJ=f"/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260607_lee2019_monitor_{BK}.json"
OUTN=f"/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260607_lee2019_pertrial_{BK}.npz"
print(f"BACKBONE={BK} EPOCHS={EP}")
res=[]; npz={}
for fi in range(K):
    held=folds[fi]; pool=[s for s in SUBS if s not in held]
    Xp=[];yp=[]
    for s in pool:
        try: d=load(s); Xp.append(d['0'][0]); yp.append(d['0'][1])
        except Exception as e: print(f"pool S{s} FAIL {str(e)[:40]}")
    Xp=np.concatenate(Xp); yp=np.concatenate(yp); C,T=Xp.shape[1],Xp.shape[2]
    print(f"\n=== fold {fi+1}/{K}: pretrain {len(yp)} trials ({len(pool)} subj) ===",flush=True)
    net=pretrain(Xp,yp,C,T)
    for s in held:
        try:
            d=load(s); F0,_=emb_prob(net,d['0'][0]); F1,P1=emb_prob(net,d['1'][0]); y0=d['0'][1]; y1=d['1'][1]
            lda=LDA(solver="lsqr",shrinkage="auto").fit(F0,y0)
            acc=(lda.predict(F1)==y1).mean()*100; sacc=(lda.predict(F0)==y0).mean()*100
            Z=(F1-F1.mean(0))/(F1.std(0)+1e-8)
            sil=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))
            conf=float(P1.max(1).mean()); ent=float((-(P1*np.log(P1+1e-9)).sum(1)).mean())
            negmaha=-float(np.median(maha_scores(F0,y0,F1)))
            res.append(dict(subj=s,acc=float(acc),sil=float(sil),sacc=float(sacc),conf=conf,negent=-ent,negmaha=negmaha))
            npz[f"F1_{s}"]=F1.astype(np.float32); npz[f"P1_{s}"]=P1.astype(np.float32); npz[f"y1_{s}"]=y1
            print(f"   S{s:>2}: acc={acc:5.1f} sil={sil:.3f} conf={conf:.3f}",flush=True)
        except Exception as e: print(f"   S{s} FAIL {str(e)[:40]}")
    json.dump(res,open(OUTJ,'w')); del net; torch.cuda.empty_cache()
np.savez_compressed(OUTN,**npz)
A=np.array([r['acc'] for r in res]); n=len(A)
def rep(key):
    v=np.array([r[key] for r in res]); rho=spearmanr(v,A).correlation
    rng=np.random.RandomState(0); b=[spearmanr(v[i],A[i]).correlation for i in (rng.randint(0,n,n) for _ in range(3000))]
    return rho,np.percentile(b,2.5),np.percentile(b,97.5)
print(f"\n===== Lee2019 monitor [{BK}] n={n}, mean acc={A.mean():.1f} (F7-dodged if>=68) =====")
print(f"{'predictor':>10} {'Spearman':>9} {'95%CI':>16}")
for k in ['sil','conf','negent','negmaha']:
    r,lo,hi=rep(k); print(f"{k:>10} {r:>+9.3f}   [{lo:+.2f},{hi:+.2f}]")
sacc=np.array([r['sacc'] for r in res]); sil=np.array([r['sil'] for r in res])
print(f"clusterability PARTIAL Spearman (control sacc) = {partial_spear(sil,A,sacc):+.3f}")
print(f"saved {OUTJ}\nsaved {OUTN}")
print("E5 OK if sil-Spearman>=+0.4 CI excl 0 on EEGNet (backbone-agnostic). E4a OK if sil beats conf/negent/negmaha.")

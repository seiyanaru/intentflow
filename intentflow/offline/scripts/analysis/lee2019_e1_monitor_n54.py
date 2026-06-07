"""E1 FULL (CLAIM-3 decisive): clusterability monitor generality at n=54 on a STRONG
F7-safe cross-subject decoder. 3-fold cross-subject: pretrain on 2/3 subjects' session0,
hold out 1/3; every subject is held-out exactly once => n=54. Per held-out subject:
  frozen backbone -> penultimate feats -> LDA(session0) -> eval session1   (acc, source-acc)
  silhouette(KMeans k=2) of z-scored session1 feats                        (monitor)
Then: Spearman(silhouette, acc) + PARTIAL-Spearman controlling source(session1->? no: source=session0 self) acc
(the mean-acc/variance-compression confound), with BCa-ish bootstrap CI + permutation p.
Pilot (n=6) gave acc 79.7% and Spearman +0.94; this confirms at n=54.
Run: intentflow env, GPU. ~2.5h (3 TCFormer pretrains).
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
from scipy.stats import spearmanr, rankdata
from models.tcformer.tcformer import TCFormerModule
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
import mne; mne.set_log_level("ERROR")
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"
torch.manual_seed(0); np.random.seed(0)
prm=LeftRightImagery(resample=250, fmin=1, fmax=45); ds=Lee2019_MI()
def ea(X):
    Cs=np.einsum('nct,ndt->ncd',X,X)/X.shape[-1]; R=Cs.mean(0)
    w,V=eigh(R); w=np.clip(w,1e-10,None); P=(V*(w**-0.5))@V.T
    return np.einsum('ij,njt->nit',P,X)
def zsc(X): return ((X-X.mean(-1,keepdims=True))/(X.std(-1,keepdims=True)+1e-7)).astype(np.float32)
_cache={}
def load(sub):
    if sub in _cache: return _cache[sub]
    X,y,meta=prm.get_data(dataset=ds,subjects=[sub]); y=(y=='right_hand').astype(int); ses=meta['session'].values
    out={}
    for s in ['0','1']:
        m=ses==s; out[s]=(zsc(ea(X[m].astype(np.float64))), y[m])
    _cache[sub]=out; return out
def interaug(X,y,n_per_class,n_seg=8):
    C,T=X.shape[1],X.shape[2]; seg=T//n_seg; aX=[];aY=[]
    for c in np.unique(y):
        Xc=X[y==c]
        for _ in range(n_per_class):
            new=np.empty((C,T),dtype=X.dtype)
            for s in range(n_seg):
                k=np.random.randint(len(Xc)); a=s*seg; b=T if s==n_seg-1 else (s+1)*seg
                new[:,a:b]=Xc[k][:,a:b]
            aX.append(new);aY.append(c)
    return np.array(aX),np.array(aY)
def pretrain(Xtr,ytr,C,epochs=150,lr=9e-4):
    net=TCFormerModule(n_channels=C,n_classes=2,F1=32,temp_kernel_lengths=(20,32,64)).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-3); warm=20
    lr_at=lambda e:(e+1)/warm if e<warm else 0.5*(1+np.cos(np.pi*(e-warm)/(epochs-warm)))
    n=len(ytr)
    for e in range(epochs):
        for g in opt.param_groups: g['lr']=lr*lr_at(e)
        aX,aY=interaug(Xtr,ytr,n_per_class=n//2)
        Xe=np.concatenate([Xtr,aX]); ye=np.concatenate([ytr,aY])
        Xt=torch.tensor(Xe).to(dev); yt=torch.tensor(ye).long().to(dev)
        p=torch.randperm(len(yt),device=dev); net.train()
        for i in range(0,len(yt),64):
            idx=p[i:i+64]; loss=nn.functional.cross_entropy(net(Xt[idx]),yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net
cap={}
def feats(net,X):
    h=net.tcn_head.classifier.register_forward_hook(lambda m,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach()))
    F=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            net(torch.tensor(X[i:i+64]).to(dev)); Ff=cap['f']; F.append((Ff[:,:,0] if Ff.ndim==3 else Ff).cpu().numpy())
    h.remove(); return np.concatenate(F)
def partial_spearman(x,y,z):  # corr(x,y | z) on ranks via residualization
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    def resid(a,b):
        b1=np.c_[np.ones_like(b),b]; beta=np.linalg.lstsq(b1,a,rcond=None)[0]; return a-b1@beta
    return float(np.corrcoef(resid(rx,rz),resid(ry,rz))[0,1])

SUBS=list(range(1,55)); K=3
folds=[SUBS[i::K] for i in range(K)]   # 3 interleaved folds covering all 54
OUT="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260606_lee2019_e1_monitor_n54.json"
res=[]
for fi in range(K):
    held=folds[fi]; pool=[s for s in SUBS if s not in held]
    print(f"\n=== fold {fi+1}/{K}: pretrain on {len(pool)} subj, hold out {len(held)} ===",flush=True)
    Xp=[];yp=[]
    for s in pool:
        try: d=load(s); Xp.append(d['0'][0]); yp.append(d['0'][1])
        except Exception as e: print(f"  pool S{s} FAIL {str(e)[:40]}")
    Xp=np.concatenate(Xp); yp=np.concatenate(yp); C=Xp.shape[1]
    net=pretrain(Xp,yp,C); print(f"  pretrained on {len(yp)} trials, eval held-out...",flush=True)
    for s in held:
        try:
            d=load(s); F0=feats(net,d['0'][0]); F1=feats(net,d['1'][0]); y0=d['0'][1]; y1=d['1'][1]
            lda=LDA(solver="lsqr",shrinkage="auto").fit(F0,y0)
            acc=(lda.predict(F1)==y1).mean()*100
            sacc=(lda.predict(F0)==y0).mean()*100   # session0 self-fit acc (source-acc proxy / variance control)
            Z=(F1-F1.mean(0))/(F1.std(0)+1e-8)
            sil=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))
            res.append(dict(subj=s,acc=float(acc),sil=float(sil),sacc=float(sacc)))
            print(f"   S{s:>2}: acc={acc:5.1f} sil={sil:.3f} sacc={sacc:.1f}",flush=True)
        except Exception as e: print(f"   S{s} eval FAIL {str(e)[:40]}")
    json.dump(res,open(OUT,'w')); del net; torch.cuda.empty_cache()

acc=np.array([r['acc'] for r in res]); sil=np.array([r['sil'] for r in res]); sacc=np.array([r['sacc'] for r in res])
rho=spearmanr(sil,acc).correlation
prho=partial_spearman(sil,acc,sacc)
# bootstrap CI + permutation p
rng=np.random.RandomState(0); n=len(acc)
boot=[spearmanr(sil[i],acc[i]).correlation for i in (rng.randint(0,n,n) for _ in range(5000))]
lo,hi=np.percentile(boot,[2.5,97.5])
perm=[spearmanr(sil,acc[rng.permutation(n)]).correlation for _ in range(5000)]
pval=(np.sum(np.abs(perm)>=abs(rho))+1)/(len(perm)+1)
print(f"\n===== Lee2019 E1 monitor, n={n} (strong cross-subject decoder) =====")
print(f"mean acc = {acc.mean():.1f} (range {acc.min():.0f}-{acc.max():.0f})   [F7 dodged if >=68]")
print(f"clusterability->acc Spearman = {rho:+.3f}  95%CI[{lo:+.2f},{hi:+.2f}]  perm-p={pval:.4f}")
print(f"PARTIAL Spearman (control session0-self acc) = {prho:+.3f}")
print(f"saved {OUT}")
print("VALIDATE CLAIM-3 if mean acc>=68 AND Spearman>=+0.4 (CI excludes 0) AND partial>=+0.3.")

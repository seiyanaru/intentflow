"""Faithful scale test: TCFormer (same backbone/feature space as 2a/2b) on Lee2019,
cross-session (sess0 train -> sess1 test), per subject. Reliability law: does clusterability
of sess1 DEEP (64-d penultimate) features predict cross-session accuracy across 54 subjects?
Broadband + per-trial z-score (match 2a). Downloads to local2 via ~/mne_data symlink.
"""
import os, warnings, json, traceback
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn
from models.tcformer.tcformer import TCFormerModule
from moabb.datasets import Lee2019_MI
from moabb.paradigms import LeftRightImagery
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"; torch.manual_seed(0); np.random.seed(0)
prm=LeftRightImagery(resample=250, fmin=1, fmax=45)   # broadband, like 2a
ds=Lee2019_MI()
def zsc(X): return ((X-X.mean(-1,keepdims=True))/(X.std(-1,keepdims=True)+1e-7)).astype(np.float32)
def train(Xtr,ytr,C,T,epochs=250):
    net=TCFormerModule(n_channels=C,n_classes=2).to(dev)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev);n=len(yt)
    net.train()
    for e in range(epochs):
        p=torch.randperm(n,device=dev)
        for i in range(0,n,32):
            idx=p[i:i+32];loss=nn.functional.cross_entropy(net(Xt[idx]),yt[idx])
            opt.zero_grad();loss.backward();opt.step()
        sch.step()
    net.eval();return net
def feat_logit(net,X):
    cap={};h=net.tcn_head.classifier.register_forward_hook(lambda m,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach()))
    F=[];L=[]
    with torch.no_grad():
        for i in range(0,len(X),64):
            o=net(torch.tensor(X[i:i+64]).to(dev));L.append(o.cpu().numpy());Ff=cap['f'];F.append((Ff[:,:,0] if Ff.ndim==3 else Ff).cpu().numpy())
    h.remove();return np.concatenate(F),np.concatenate(L)
OUT="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260606_reliability_law_tcformer_lee2019.json"
res=[]
for s in range(1,55):
    try:
        X,y,meta=prm.get_data(dataset=ds,subjects=[s]); y=(y=='right_hand').astype(int); ses=meta['session'].values
        tr=ses=='0';te=ses=='1';C,T=X.shape[1],X.shape[2]
        net=train(zsc(X[tr]),y[tr],C,T)
        Fte,Lte=feat_logit(net,zsc(X[te]));Ftr,_=feat_logit(net,zsc(X[tr]))
        acc=(Lte.argmax(1)==y[te]).mean()*100
        Z=(Fte-Fte.mean(0))/(Fte.std(0)+1e-8);clus=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))
        drift=float(np.linalg.norm(Ftr.mean(0)-Fte.mean(0)))
        res.append(dict(subj=s,acc=float(acc),clus=float(clus),drift=drift));json.dump(res,open(OUT,'w'))
        print(f"S{s:>2}: acc={acc:5.1f} clus={clus:.3f} drift={drift:.1f}",flush=True)
        del net;torch.cuda.empty_cache()
    except Exception as e:
        print(f"S{s}: FAIL {str(e)[:90]}",flush=True)
if res:
    acc=[r['acc'] for r in res];clus=[r['clus'] for r in res];drift=[r['drift'] for r in res]
    print(f"\n=== TCFormer Lee2019 cross-session n={len(res)} ===")
    print(f"mean acc={np.mean(acc):.1f} (range {min(acc):.0f}-{max(acc):.0f})")
    print(f"corr(clusterability, acc) = {spearmanr(clus,acc).correlation:+.3f}   [reliability law, DEEP features]")
    print(f"corr(drift, acc)          = {spearmanr(drift,acc).correlation:+.3f}")
print("LAW HARDENED if corr(clus,acc) ~ +0.8 (matches 2a/2b deep-feature finding).")

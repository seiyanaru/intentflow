"""Calibration-conditional law: does a deep MI backbone's OVERCONFIDENCE (ECE) govern
whether source-geometry beats confidence for confident-region error detection?
Per backbone x subject (2a, clean, own-softmax): acc, ECE, margin_conf AUROC, Maha_conf
AUROC, advantage = Maha_conf - margin_conf. Hypothesis: as ECE rises, margin_conf -> 0.5
and advantage rises. Backbones: TCFormer(ckpt) + braindecode {EEGNet, ShallowFBCSP, Deep4}.
"""
import os, sys, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from datamodules.bcic4_2a import BCICIV2a
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"; torch.manual_seed(0); np.random.seed(0)
DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
PREP=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
def arr(ds):
    X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32);return X,np.array([int(ds[i][1]) for i in range(len(ds))])
def sm(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def ece(prob,y,bins=15):
    conf=prob.max(1);pred=prob.argmax(1);acc=(pred==y).astype(float);e=0;N=len(y)
    for b in range(bins):
        lo,hi=b/bins,(b+1)/bins;m=(conf>lo)&(conf<=hi)
        if m.sum():e+=abs(acc[m].mean()-conf[m].mean())*m.sum()/N
    return e*100
def maha(Xtr,ytr,Xte):
    cls=np.unique(ytr);M=np.stack([Xtr[ytr==c].mean(0) for c in cls])
    res=np.concatenate([Xtr[ytr==c]-Xtr[ytr==c].mean(0) for c in cls],0)
    P=np.linalg.pinv(LedoitWolf().fit(res).covariance_)
    return np.min(np.stack([np.einsum('ij,jk,ik->i',Xte-M[c],P,Xte-M[c]) for c in range(len(cls))],1),1)
def metrics(prob_tr,ytr,Ftr,prob_te,yte,Fte):
    pred=prob_te.argmax(1);wrong=(pred!=yte).astype(int)
    ms=np.sort(prob_te,1);margin=ms[:,-1]-ms[:,-2]
    dm=maha(Ftr,ytr,Fte);conf=margin>=np.median(margin);wc=wrong[conf]
    g=lambda yt,sc: roc_auc_score(yt,sc) if yt.sum() not in (0,len(yt)) else np.nan
    return dict(acc=(pred==yte).mean()*100,ece=ece(prob_te,yte),
        margin_conf=g(wc,-margin[conf]),maha_conf=g(wc,dm[conf]),
        sat=(margin[conf]>0.99).mean()*100,nwrong=int(wc.sum()))

# feature hook: input to last Conv2d/Linear
def feat_logits(net,X):
    mods=[m for m in net.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]; cap={}
    h=mods[-1].register_forward_pre_hook(lambda m,i:cap.__setitem__('f',i[0].detach()))
    F=[];L=[]
    with torch.no_grad():
        for i in range(0,len(X),48):
            o=net(torch.tensor(X[i:i+48]).to(dev));L.append(o.detach().cpu().numpy());F.append(cap['f'].reshape(min(48,len(X)-i),-1).cpu().numpy())
    h.remove();return np.concatenate(F),np.concatenate(L)
def train_bd(ctor,C,T,Xtr,ytr,ep=300):
    net=ctor().to(dev);opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3)
    Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev)
    for e in range(ep):
        p=torch.randperm(len(yt),device=dev)
        for i in range(0,len(yt),48):
            idx=p[i:i+48];loss=nn.functional.cross_entropy(net(Xt[idx])[:, :4] if net(Xt[idx]).shape[1]>4 else net(Xt[idx]),yt[idx])
            opt.zero_grad();loss.backward();opt.step()
    net.eval();return net

from braindecode.models import EEGNetv4, ShallowFBCSPNet, Deep4Net
def mk(cls,**kw):
    for kwargs in [dict(n_chans=22,n_outputs=4,n_times=1000,**kw),dict(n_chans=22,n_outputs=4,n_times=1000),
                   dict(in_chans=22,n_classes=4,input_window_samples=1000,**kw)]:
        try: return lambda kk=kwargs: cls(**kk)
        except Exception: continue
    return None
BACKBONES={
 "EEGNet": lambda C,T: EEGNetv4(22,4,n_times=1000,final_conv_length='auto'),
 "ShallowFBCSP": lambda C,T: ShallowFBCSPNet(22,4,n_times=1000,final_conv_length='auto'),
 "Deep4": lambda C,T: Deep4Net(22,4,n_times=1000,final_conv_length='auto'),
}

results={}  # backbone -> list of per-subject metric dicts
# TCFormer via ckpt
try:
    from models.tcformer.tcformer import TCFormer
    CK="intentflow/offline/results/baseline_5seed_s0_20260309_122106/checkpoints"
    def build_tcf(ck):
        hp=ck["hyper_parameters"]
        m=TCFormer(n_channels=hp["n_channels"],n_classes=hp["n_classes"],F1=hp["F1"],temp_kernel_lengths=tuple(hp["temp_kernel_lengths"]),
            pool_length_1=hp["pool_length_1"],pool_length_2=hp["pool_length_2"],D=hp["D"],dropout_conv=hp["dropout_conv"],d_group=hp["d_group"],
            tcn_depth=hp["tcn_depth"],kernel_length_tcn=hp["kernel_length_tcn"],dropout_tcn=hp["dropout_tcn"],use_group_attn=hp["use_group_attn"],
            q_heads=hp["q_heads"],kv_heads=hp["kv_heads"],trans_depth=hp["trans_depth"],trans_dropout=hp["trans_dropout"])
        m.load_state_dict(ck["state_dict"],strict=False);return m.eval().to(dev)
    cap={}
    def tcf_fl(net,X):
        h=net.model.tcn_head.classifier.register_forward_hook(lambda mod,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach().cpu().float()))
        F=[];L=[]
        with torch.no_grad():
            for i in range(0,len(X),48):
                o=net(torch.from_numpy(X[i:i+48]).float().to(dev));L.append(o.detach().cpu().numpy());Ff=cap['f'];F.append((Ff[:,:,0] if Ff.ndim==3 else Ff).numpy())
        h.remove();return np.concatenate(F),np.concatenate(L)
    rs=[]
    for s in range(1,10):
        ck=torch.load(os.path.join(CK,f"subject_{s}_model.ckpt"),map_location="cpu",weights_only=False);net=build_tcf(ck)
        dm=BCICIV2a(PREP,s);dm.setup();Xtr,ytr=arr(dm.train_dataset);Xte,yte=arr(dm.test_dataset)
        if Xtr.shape[-1]>1000:Xtr=Xtr[...,:1000];Xte=Xte[...,:1000]
        Ftr,Ltr=tcf_fl(net,Xtr);Fte,Lte=tcf_fl(net,Xte)
        rs.append(metrics(sm(Ltr),ytr,Ftr,sm(Lte),yte,Fte));del net;torch.cuda.empty_cache()
    results["TCFormer"]=rs
except Exception: print("TCFormer FAIL:",traceback.format_exc()[-500:])

for name,ctor in BACKBONES.items():
    try:
        rs=[]
        for s in range(1,10):
            dm=BCICIV2a(PREP,s);dm.setup();Xtr,ytr=arr(dm.train_dataset);Xte,yte=arr(dm.test_dataset)
            net=train_bd(lambda:ctor(22,1000),22,1000,Xtr,ytr)
            Ftr,Ltr=feat_logits(net,Xtr);Fte,Lte=feat_logits(net,Xte)
            rs.append(metrics(sm(Ltr[:, :4]),ytr,Ftr,sm(Lte[:, :4]),yte,Fte));del net;torch.cuda.empty_cache()
        results[name]=rs; print(f"[{name}] done",flush=True)
    except Exception: print(f"{name} FAIL:",traceback.format_exc()[-500:])

print(f"\n{'backbone':>13} {'acc':>6} {'ECE':>6} {'sat%':>6} {'margin_conf':>11} {'maha_conf':>10} {'advantage':>10}")
pts=[]
for bb,rs in results.items():
    f=lambda k:np.nanmean([r[k] for r in rs])
    adv=f('maha_conf')-f('margin_conf')
    print(f"{bb:>13} {f('acc'):>6.1f} {f('ece'):>6.1f} {f('sat'):>6.1f} {f('margin_conf'):>11.3f} {f('maha_conf'):>10.3f} {adv:>+10.3f}")
    for r in rs:
        if not np.isnan(r['margin_conf']) and not np.isnan(r['maha_conf']):
            pts.append((r['ece'],r['maha_conf']-r['margin_conf'],r['sat']))
pts=np.array(pts)
if len(pts)>3:
    from numpy import corrcoef
    print(f"\nacross {len(pts)} (backbone x subject) points:")
    print(f"  corr(ECE, geometry-advantage) = {corrcoef(pts[:,0],pts[:,1])[0,1]:+.3f}")
    print(f"  corr(saturation, geometry-advantage) = {corrcoef(pts[:,2],pts[:,1])[0,1]:+.3f}")
print("\nLAW HOLDS if corr(ECE/sat, advantage) strongly POSITIVE: overconfidence -> confidence dead -> geometry needed.")

"""E2: generality of the GeoGate signal (dead-confidence / live-geometry) across
DATASET (2b, 3ch) and BACKBONE (EEGNet on 2a). Clean source features only.
Same diagnostic as verify_geogate_signal: deployed-head proxy = LDA-on-train (sanity vs
known source acc); confident subset = top-50% margin; Maha_AUROC(conf) vs margin_AUROC(conf);
abstain@80% error drop. Run: intentflow env, GPU.
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, torch.nn as nn, yaml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from models.tcformer.tcformer import TCFormer
from braindecode.models import EEGNetv4
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"; torch.manual_seed(0); np.random.seed(0)

def maha_P(Xtr,ytr):
    cls=np.unique(ytr); M=np.stack([Xtr[ytr==c].mean(0) for c in cls])
    res=np.concatenate([Xtr[ytr==c]-Xtr[ytr==c].mean(0) for c in cls],0)
    P=np.linalg.pinv(LedoitWolf().fit(res).covariance_); return M,P
def dmin(X,M,P):
    return np.min(np.stack([np.einsum('ij,jk,ik->i',X-M[c],P,X-M[c]) for c in range(M.shape[0])],1),1)

def diagnostic(tag, TR, TRy, TE, TEy, src_acc_ref):
    print(f"\n=== {tag}  (SANITY: LDAacc ~ {src_acc_ref}) ===")
    print(f"{'subj':>4} {'LDAacc':>7} {'Maha_conf':>10} {'margin_conf':>12} {'abst80→full':>14} {'nWrongConf':>10}")
    rows=[]
    for s in sorted(TR):
        Xtr,ytr,Xte,yte=TR[s],TRy[s],TE[s],TEy[s]
        lda=LDA(solver="lsqr",shrinkage="auto").fit(Xtr,ytr)
        pr=lda.classes_[lda.predict_proba(Xte).argmax(1)]; wrong=(pr!=yte).astype(int)
        pb=lda.predict_proba(Xte); ms=np.sort(pb,1); margin=ms[:,-1]-ms[:,-2]
        M,P=maha_P(Xtr,ytr); dm=dmin(Xte,M,P)
        conf=margin>=np.median(margin); wc=wrong[conf]
        f=lambda yt,sc: roc_auc_score(yt,sc) if yt.sum() not in (0,len(yt)) else np.nan
        amc=f(wc,dm[conf]); amg=f(wc,-margin[conf])
        keep=dm<=np.quantile(dm,0.80); ek=wrong[keep].mean()*100; ef=wrong.mean()*100
        rows.append((s,(pr==yte).mean()*100,amc,amg,ek,ef,int(wc.sum())))
        print(f"S{s:>3} {rows[-1][1]:>7.1f} {amc:>10.3f} {amg:>12.3f} {ek:>6.1f}→{ef:<6.1f} {int(wc.sum()):>10}")
    m=lambda j:np.nanmean([r[j] for r in rows])
    nM=sum(1 for r in rows if r[2]>0.5); nA=sum(1 for r in rows if r[4]<r[5])
    print(f"MEAN LDAacc={m(1):.1f} | Maha_conf={m(2):.3f} vs margin_conf={m(3):.3f} | abstain {m(5):.1f}→{m(4):.1f} ({m(4)-m(5):+.1f}) | Maha>0.5: {nM}/{len(rows)} | abstain-helps: {nA}/{len(rows)}")

# ---------- Part 1: 2b TCFormer (cross-dataset, 3ch) ----------
def build_tcf(ckpt):
    hp=ckpt["hyper_parameters"]
    m=TCFormer(n_channels=hp["n_channels"],n_classes=hp["n_classes"],F1=hp["F1"],
        temp_kernel_lengths=tuple(hp["temp_kernel_lengths"]),pool_length_1=hp["pool_length_1"],
        pool_length_2=hp["pool_length_2"],D=hp["D"],dropout_conv=hp["dropout_conv"],d_group=hp["d_group"],
        tcn_depth=hp["tcn_depth"],kernel_length_tcn=hp["kernel_length_tcn"],dropout_tcn=hp["dropout_tcn"],
        use_group_attn=hp["use_group_attn"],q_heads=hp["q_heads"],kv_heads=hp["kv_heads"],
        trans_depth=hp["trans_depth"],trans_dropout=hp["trans_dropout"])
    m.load_state_dict(ckpt["state_dict"],strict=False); return m.eval().to(dev)
cap={}
def tcf_feats(model,X):
    h=model.model.tcn_head.classifier.register_forward_hook(lambda mod,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach().cpu().float()))
    fs=[]
    with torch.no_grad():
        for i in range(0,len(X),48):
            model(torch.from_numpy(X[i:i+48]).float().to(dev)); F=cap['f']; fs.append((F[:,:,0] if F.ndim==3 else F).clone())
    h.remove(); return torch.cat(fs).numpy()
try:
    from datamodules.bcic4_2b import BCICIV2b
    cfg=yaml.safe_load(open("intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347/config.yaml"))
    prep=dict(cfg["preprocessing"]); prep.update(test_batch_size=48,num_workers=0,eval_label_path=None,data_path=None)
    CK="intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347/checkpoints"
    TR={};TRy={};TE={};TEy={}
    def arr(ds):
        X=np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32);
        if X.shape[-1]>1000: X=X[...,:1000]
        return X,np.array([int(ds[i][1]) for i in range(len(ds))])
    for s in range(1,10):
        ck=torch.load(os.path.join(CK,f"subject_{s}_model.ckpt"),map_location="cpu",weights_only=False)
        mdl=build_tcf(ck); dm=BCICIV2b(prep,s); dm.setup()
        Xtr,ytr=arr(dm.train_dataset); Xte,yte=arr(dm.test_dataset)
        TR[s]=tcf_feats(mdl,Xtr);TRy[s]=ytr;TE[s]=tcf_feats(mdl,Xte);TEy[s]=yte
        del mdl; torch.cuda.empty_cache()
    diagnostic("2b TCFormer (3ch, cross-dataset)",TR,TRy,TE,TEy,"87.7")
except Exception as e:
    import traceback; print("2b PART FAILED:",traceback.format_exc()[-800:])

# ---------- Part 2: EEGNet on 2a (cross-backbone) ----------
try:
    from datamodules.bcic4_2a import BCICIV2a
    DATA="/mnt/data/seiya.narukawa/intentflow/data/raw/BCICIV_2a_gdf/"
    prep2=dict(sfreq=250,low_cut=None,high_cut=None,start=0.0,stop=4.0,batch_size=48,test_batch_size=48,num_workers=0,z_scale=True,data_path=DATA,eval_label_path=DATA+"labels")
    def arr2(ds):
        X=np.stack([ds[i][0].numpy() for i in range(len(ds))]).astype(np.float32);return X,np.array([int(ds[i][1]) for i in range(len(ds))])
    def eeg_feats(net,X):
        convs=[m for m in net.modules() if isinstance(m,nn.Conv2d)]; ff={}
        h=convs[-1].register_forward_pre_hook(lambda mod,i:ff.__setitem__('f',i[0].detach()))
        out=[]
        with torch.no_grad():
            for i in range(0,len(X),48):
                net(torch.tensor(X[i:i+48]).to(dev)); out.append(ff['f'].reshape(min(48,len(X)-i),-1).cpu().numpy())
        h.remove(); return np.concatenate(out)
    TR={};TRy={};TE={};TEy={}
    for s in range(1,10):
        dm=BCICIV2a(prep2,s);dm.setup();Xtr,ytr=arr2(dm.train_dataset);Xte,yte=arr2(dm.test_dataset)
        net=EEGNetv4(Xtr.shape[1],4,n_times=Xtr.shape[2],final_conv_length='auto').to(dev)
        opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-3);Xt=torch.tensor(Xtr).to(dev);yt=torch.tensor(ytr).long().to(dev)
        for ep in range(300):
            p=torch.randperm(len(yt),device=dev)
            for i in range(0,len(yt),48):
                idx=p[i:i+48];loss=nn.functional.cross_entropy(net(Xt[idx]),yt[idx]);opt.zero_grad();loss.backward();opt.step()
        net.eval();TR[s]=eeg_feats(net,Xtr);TRy[s]=ytr;TE[s]=eeg_feats(net,Xte);TEy[s]=yte
        del net;torch.cuda.empty_cache()
    diagnostic("EEGNet 2a (22ch, cross-backbone)",TR,TRy,TE,TEy,"~65-70 (EEGNet weaker)")
except Exception as e:
    import traceback; print("EEGNet PART FAILED:",traceback.format_exc()[-800:])
print("\nLAW HOLDS if Maha_conf>>margin_conf(~0.5) AND abstain helps, on BOTH 2b and EEGNet.")

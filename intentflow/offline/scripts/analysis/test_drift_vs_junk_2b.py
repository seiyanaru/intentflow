"""2b replication of the drift-vs-junk diagnosis (clean no-EA source features).
Extract 2b source penultimate features (train+eval) per subject from the 2b baseline
checkpoint (config has NO 'ea' -> these ARE source features). Then:
  clusterability(eval) vs source_acc  [junk detector predicts reliability]
  drift-magnitude(train->eval centroid) vs EA-benefit(full_ea - source)  [drift detector:
     does LOW drift correctly flag over-adaptation hurt cases e.g. S5 (EA 97.8->64)?]
SANITY: LDA(train)->eval accuracy must ~ 2b source 87.7.
"""
import os, sys, warnings, csv
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np, torch, yaml
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import spearmanr
from models.tcformer.tcformer import TCFormer
from datamodules.bcic4_2b import BCICIV2b
dev="cuda" if torch.cuda.is_available() else "cpu"; assert dev=="cuda"
RUN="intentflow/offline/results/TCFormer_bcic2b_seed-0_aug-True_GPU0_20260422_1347"
cfg=yaml.safe_load(open(RUN+"/config.yaml"))
prep=dict(cfg["preprocessing"]); prep.update(test_batch_size=48,num_workers=0,eval_label_path=None,data_path=None)
assert "ea" not in prep or not prep.get("ea"), "config unexpectedly has EA"
def build(ck):
    hp=ck["hyper_parameters"]
    m=TCFormer(n_channels=hp["n_channels"],n_classes=hp["n_classes"],F1=hp["F1"],temp_kernel_lengths=tuple(hp["temp_kernel_lengths"]),
        pool_length_1=hp["pool_length_1"],pool_length_2=hp["pool_length_2"],D=hp["D"],dropout_conv=hp["dropout_conv"],d_group=hp["d_group"],
        tcn_depth=hp["tcn_depth"],kernel_length_tcn=hp["kernel_length_tcn"],dropout_tcn=hp["dropout_tcn"],use_group_attn=hp["use_group_attn"],
        q_heads=hp["q_heads"],kv_heads=hp["kv_heads"],trans_depth=hp["trans_depth"],trans_dropout=hp["trans_dropout"])
    m.load_state_dict(ck["state_dict"],strict=False);return m.eval().to(dev)
cap={}
def feats(net,X):
    h=net.model.tcn_head.classifier.register_forward_hook(lambda mod,i,o:cap.__setitem__('f',(i[0] if isinstance(i,(tuple,list)) else i).detach().cpu().float()))
    F=[]
    with torch.no_grad():
        for i in range(0,len(X),48):
            net(torch.from_numpy(X[i:i+48]).float().to(dev));Ff=cap['f'];F.append((Ff[:,:,0] if Ff.ndim==3 else Ff).numpy())
    h.remove();return np.concatenate(F)
def arr(ds):
    X=np.stack([ds[i][0] for i in range(len(ds))]).astype(np.float32)
    if X.shape[-1]>1000:X=X[...,:1000]
    return X,np.array([int(ds[i][1]) for i in range(len(ds))])
# portfolio per-subject accs (2b)
rows={int(r["subject"]):r for r in csv.DictReader(open("intentflow/offline/results/research_outputs/260602_bcic2b_portfolio_seed0/subject_summary.csv"))}
clus=[];drift=[];sacc=[];eab=[];lda_acc=[]
print(f"{'S':>3} {'clus':>6} {'drift':>7} {'src':>6} {'full_ea':>7} {'EAΔ':>6} {'LDAchk':>7}")
for s in range(1,10):
    ck=torch.load(os.path.join(RUN,"checkpoints",f"subject_{s}_model.ckpt"),map_location="cpu",weights_only=False)
    net=build(ck);dm=BCICIV2b(prep,s);dm.setup();Xtr,ytr=arr(dm.train_dataset);Xte,yte=arr(dm.test_dataset)
    Ftr=feats(net,Xtr);Fte=feats(net,Xte);del net;torch.cuda.empty_cache()
    Z=(Fte-Fte.mean(0))/(Fte.std(0)+1e-8)
    cl=silhouette_score(Z,KMeans(2,n_init=5,random_state=0).fit_predict(Z))  # 2 classes for 2b
    dr=np.linalg.norm(Ftr.mean(0)-Fte.mean(0))
    so=float(rows[s]["acc_source"]);fea=float(rows[s]["acc_full_ea"])
    lacc=(LDA(solver="lsqr",shrinkage="auto").fit(Ftr,ytr).predict(Fte)==yte).mean()*100
    clus.append(cl);drift.append(dr);sacc.append(so);eab.append(fea-so);lda_acc.append(lacc)
    print(f"S{s:>2} {cl:>6.3f} {dr:>7.2f} {so:>6.1f} {fea:>7.1f} {fea-so:>+6.1f} {lacc:>7.1f}")
print(f"\nSANITY: LDA(train)->eval mean={np.mean(lda_acc):.1f} (must ~87.7=2b source); portfolio source mean={np.mean(sacc):.1f}")
print(f"corr(clusterability, source_acc)   = {spearmanr(clus,sacc).correlation:+.2f}  [junk detector]")
print(f"corr(drift, EA-benefit full_ea-src)= {spearmanr(drift,eab).correlation:+.2f}  [drift detector: predicts where EA helps/HURTS]")
# does low drift flag the EA-hurt subjects?
hurt=[s for s in range(9) if eab[s]<-5]
print(f"\nEA-hurt subjects (Δ<-5): {[f'S{i+1}(drift={drift[i]:.1f},Δ={eab[i]:+.0f})' for i in hurt]}")
print(f"drift of hurt vs helped: hurt={np.mean([drift[i] for i in hurt]):.1f} vs helped={np.mean([drift[i] for i in range(9) if eab[i]>=-5]):.1f}")
print("\nHOLDS if clusterability predicts reliability AND drift separates EA-helps from EA-hurts (low drift -> EA hurts).")

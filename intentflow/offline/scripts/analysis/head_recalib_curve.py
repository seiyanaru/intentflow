"""Head-recalibration unlocking curve (tests the practical accuracy layer of the
Head-Frozen Cross-Family Decoder). Law 2: drift hits the head; features survive.
Q: with K (possibly ErrP-noisy) labels, refit ONLY a linear head on the frozen
64-d TCFormer features -> how much head-headroom is recovered on the UNLABELED
remainder, does it beat DA-DC, and does it survive ErrP-quality label noise?

Honesty guards:
- eval STRICTLY on the unlabeled remainder (labeled trials excluded from scoring).
- K labels sampled at RANDOM, averaged over many draws (report mean).
- ErrP realism: each provided label is corrupted to a random wrong class w.p. eps.
- baselines: deployed softmax (0 labels) and DA-DC blend+0.3R (0 labels, 2a only).
2a (with DA-DC baseline) + 2b (softmax baseline).
"""
import glob, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
RR="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/"
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
def sm(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)

def feats_logits(ds, sid):
    if ds=="2a":
        fd=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz"))
        ld=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy"))
    else:
        fd=sorted(glob.glob(RR+f"ea_aware_tcformer_bcic2b_s*_seed0_*/features_s{sid}_TCFormer.npz"))
        ld=sorted(glob.glob(RR+f"ea_aware_tcformer_bcic2b_s*_seed0_*/logits_s{sid}_TCFormer.npy"))
    if not fd or not ld: return None
    d=np.load(fd[-1]);F=d["features"].reshape(d["features"].shape[0],-1).astype(np.float64)
    return F, d["labels"].astype(int), np.load(ld[-1])

# 2a DA-DC baseline pieces
exp=np.load(RES+"260602_expert_portfolio_table/expert_portfolio_arrays.npz",allow_pickle=True)
IX={n:[str(x) for x in exp["experts"].tolist()].index(n) for n in ["source","full_ea","shrink_0.1"]}
R2a=np.load(RES+"260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
def dadc_2a(i):
    src=exp["probs"][i,IX["source"]];fe=exp["probs"][i,IX["full_ea"]];sh=exp["probs"][i,IX["shrink_0.1"]]
    return (0.3*src+0.4*fe+0.3*sh+0.3*R2a[i]).argmax(1)

EPS=[0.0,0.1,0.2,0.3]; DRAWS=40; NCLS={"2a":4,"2b":2}; KS={"2a":[8,16,32,64],"2b":[4,8,16,32,64]}
for ds in ["2a","2b"]:
    nc=NCLS[ds]; Ks=KS[ds]
    print(f"\n========== {ds} (TCFormer, head-recalib on frozen 64-d features) ==========")
    soft_rem={}; dadc_full=[]; ceil=[]; curve={(e,k):[] for e in EPS for k in Ks}
    for sid in range(1,10):
        fl=feats_logits(ds,sid)
        if fl is None: print(f"S{sid} missing"); continue
        F,y,logit=fl; n=len(y); Fz=(F-F.mean(0))/(F.std(0)+1e-8)
        sp=sm(logit).argmax(1)
        # ceiling: LDA-probe CV (true labels)
        cc=[]
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(Fz,y):
            cc.append((LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr").fit(Fz[tr],y[tr]).predict(Fz[va])==y[va]).mean()*100)
        ceil.append(np.mean(cc))
        if ds=="2a": dadc_full.append((dadc_2a(sid-1)==y).mean()*100)
        # per-subject curve
        srem=[]
        for e in EPS:
            for K in Ks:
                accs=[];brem=[]
                for d in range(DRAWS):
                    rng=np.random.RandomState(d*131+K*7+int(e*100))
                    idx=rng.permutation(n)[:K]; rem=np.setdiff1d(np.arange(n),idx)
                    yl=y[idx].copy()
                    fl_=rng.random(K)<e
                    for j in np.where(fl_)[0]:
                        yl[j]=rng.choice([c for c in range(nc) if c!=y[idx[j]]])
                    if len(np.unique(yl))<2 or K<=len(np.unique(yl)):
                        accs.append((sp[rem]==y[rem]).mean()*100)  # fallback: keep softmax
                    else:
                        try:
                            clf=LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr").fit(Fz[idx],yl)
                            accs.append((clf.predict(Fz[rem])==y[rem]).mean()*100)
                        except Exception:
                            accs.append((sp[rem]==y[rem]).mean()*100)
                    brem.append((sp[rem]==y[rem]).mean()*100)
                curve[(e,K)].append(np.mean(accs))
                if e==0.0 and K==Ks[0]: srem.append(np.mean(brem))
        soft_rem[sid]=np.mean([ (sp==y).mean()*100 ])
    m=lambda L:float(np.mean(L))
    base=m([soft_rem[s] for s in soft_rem])
    print(f"deployed softmax (0 label): {base:.1f} | LDA-probe ceiling(true labels): {m(ceil):.1f}" + (f" | DA-DC: {m(dadc_full):.1f}" if ds=='2a' else ""))
    print(f"\nhead-recalib remainder-accuracy by (label budget K, ErrP noise eps):")
    hdr="  eps\\K  "+"".join(f"{k:>8}" for k in Ks); print(hdr)
    for e in EPS:
        row=f"  {e:>4.2f}  "+"".join(f"{m(curve[(e,k)]):>8.1f}" for k in Ks); print(row)
    bar = m(dadc_full) if ds=='2a' else base
    barname = "DA-DC" if ds=='2a' else "softmax"
    print(f"\n  (bar to beat = {barname} {bar:.1f}. cells above it => minimal-label head-recalib wins, even with ErrP noise.)")
print("\nVERDICT: if recalib at small K and eps~0.2-0.3 exceeds DA-DC/softmax -> ErrP-grade supervision unlocks Law-2 head-headroom = beats label-free ceiling.")

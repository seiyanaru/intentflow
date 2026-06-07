"""Maximal label-free search on TCFormer/2a cached data: can ANY label-free method
beat DA-DC (blend+0.3R = ~87.2)? Tests co-training/stacking that exploit the
head-drift fact (features good, head drifted) + family-law (cross-family decorrelated)
+ agreement seeds (T==R / T==R==S are high-precision label-free pseudo-labels).
All eval-labels used ONLY for scoring. seed0 + seed1.
"""
import glob, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
RR="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/"
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
R=np.load(RES+"260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
C=np.load(RES+"260603_d2_csp_preds.npz",allow_pickle=True)["probs"]
S=np.load(RES+"260603_d3_spectral_preds.npz",allow_pickle=True)["probs"]
def sm(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def load(seed):
    p=RES+f"260602_expert_portfolio_table{'' if seed==0 else '_seed1'}/expert_portfolio_arrays.npz"
    e=np.load(p,allow_pickle=True);ix={n:[str(x) for x in e["experts"].tolist()].index(n) for n in ["source","full_ea","shrink_0.1"]}
    return e["probs"],e["labels"],ix
def feats(sid):  # EA-aware TCFormer penultimate (64-d) on session_E
    fd=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz"))
    if not fd: return None
    d=np.load(fd[-1]);return d["features"].reshape(d["features"].shape[0],-1).astype(np.float64)
m=lambda L:float(np.nanmean(L))
def fit_pred(X,seed_idx,seed_y,Xall):
    if len(np.unique(seed_y))<2: return None
    cl=LogisticRegression(C=0.5,max_iter=2000).fit(X[seed_idx],seed_y); return cl.predict(Xall)

for seed in [0,1]:
    Tp,Lab,IX=load(seed)
    methods={k:[] for k in ["source","full_ea","blend","DA-DC","cotrain_feat","stack_TR","stack_triple","DADC+stack"]}
    perN={k:[] for k in methods}  # per-subject for regression check vs DA-DC
    for i in range(9):
        y=Lab[i];src=Tp[i,IX["source"]];fe=Tp[i,IX["full_ea"]];sh=Tp[i,IX["shrink_0.1"]];Ri=R[i];Si=S[i];Ci=C[i]
        blend=0.3*src+0.4*fe+0.3*sh
        dadc=(blend+0.3*Ri)
        F=feats(i+1)
        Fz=(F-F.mean(0))/(F.std(0)+1e-8) if F is not None else None
        stackX=np.concatenate([Fz,src,fe,Ri,Si,Ci],1) if Fz is not None else None
        # agreement seeds (label-free high-precision pseudo-labels)
        aT=src.argmax(1);aR=Ri.argmax(1);aS=Si.argmax(1)
        tr=np.where(aT==aR)[0]; triple=np.where((aT==aR)&(aR==aS))[0]
        def acc(p): return (p.argmax(1)==y).mean()*100 if p.ndim==2 else (p==y).mean()*100
        out={}
        out["source"]=acc(src);out["full_ea"]=acc(fe);out["blend"]=acc(blend);out["DA-DC"]=acc(dadc)
        # cotrain on features only, triple-agreement seeds
        p=fit_pred(Fz,triple,aT[triple],Fz) if Fz is not None else None
        out["cotrain_feat"]=acc(p) if p is not None else np.nan
        # stacking (feat+all probs), T-R agreement seeds
        p=fit_pred(stackX,tr,aT[tr],stackX) if stackX is not None else None
        out["stack_TR"]=acc(p) if p is not None else np.nan
        # stacking, triple-agreement seeds
        p2=fit_pred(stackX,triple,aT[triple],stackX) if stackX is not None else None
        out["stack_triple"]=acc(p2) if p2 is not None else np.nan
        # DA-DC blended with stack_TR probs
        if stackX is not None and len(np.unique(aT[tr]))>=2:
            cl=LogisticRegression(C=0.5,max_iter=2000).fit(stackX[tr],aT[tr]);sp=cl.predict_proba(stackX)
            spf=np.zeros((len(y),4));spf[:,cl.classes_]=sp
            out["DADC+stack"]=acc(dadc/ dadc.sum(1,keepdims=True)+0.5*spf)
        else: out["DADC+stack"]=np.nan
        for k in methods: methods[k].append(out[k]); perN[k].append(out[k])
    print(f"\n===== seed{seed} (label-free; DA-DC is the bar to beat) =====")
    for k in methods:
        delta=m(methods[k])-m(methods["DA-DC"])
        harmed=sum(1 for i in range(9) if perN[k][i]<perN["DA-DC"][i]-0.01) if not np.isnan(m(methods[k])) else -1
        print(f"  {k:>14}: {m(methods[k]):6.2f}  (vs DA-DC {delta:+.2f}, harmed {harmed}/9)")
print("\nWIN = any method > DA-DC mean AND not harming most subjects. Else DA-DC ~= label-free ceiling.")

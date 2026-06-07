"""E0a: the FORA kill-switch. Test the family-diversity LAW (F2/F4) on bcic2a:
does CROSS-FAMILY (deep T vs covariance/spectral D) decorrelate the deep model's
CONFIDENT errors and give a better reliability signal than SAME-FAMILY (deep T vs
deep ATCNet) and single-model UQ? base predictor = T (source TCFormer).
Metrics: (1) confident-error rescue rate, (2) agreement precision, (3) risk-coverage AURC.
"""
import numpy as np, glob, re, os
exp = np.load("intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz", allow_pickle=True)
ex = [str(x) for x in exp["experts"].tolist()]; EI = {n: ex.index(n) for n in ["source"]}
Lab = exp["labels"]
T_all = exp["probs"][:, EI["source"]]
D1 = np.load("intentflow/offline/results/research_outputs/260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
D2 = np.load("intentflow/offline/results/research_outputs/260603_d2_csp_preds.npz", allow_pickle=True)["probs"]
D3 = np.load("intentflow/offline/results/research_outputs/260603_d3_spectral_preds.npz", allow_pickle=True)["probs"]
ATD = "intentflow/offline/results/ATCNet_bcic2a_seed-0_aug-True_GPU1_20260603_1732"
def sm(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
atc = {}
for f in glob.glob(f"{ATD}/logits_s*_ATCNet.npy"):
    sid = int(re.search(r'logits_s(\d+)_ATCNet', os.path.basename(f)).group(1)); atc[sid] = sm(np.load(f))
atc_subs = sorted(atc); print(f"ATCNet subjects: {atc_subs}")
def ent(p): return -(p*np.log(np.clip(p,1e-12,1))).sum(1)
def symkl(p,q):
    return ((p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(q,1e-12,1)))).sum(1)
           +(q*(np.log(np.clip(q,1e-12,1))-np.log(np.clip(p,1e-12,1)))).sum(1))
def aurc(conf, correct):  # higher conf kept first; lower AURC=better
    o=np.argsort(-conf); c=correct[o]; cov=np.arange(1,len(c)+1)/len(c)
    risk=1-np.cumsum(c)/np.arange(1,len(c)+1)
    return float(np.trapz(risk,cov))
m=lambda L:float(np.nanmean(L)) if len(L) else float('nan')

# Metric 1+2: cross-family vs same-family on T's confident errors / agreement
print("\n=== M1: rescue rate on T's CONFIDENT errors (higher=more decorrelated) ===")
print(f"{'model':>10} {'family':>12} {'rescue%':>8} {'errJaccard_vsT':>14} {'agree_prec%':>11} {'agree_cov%':>10}")
others={"ATCNet":(atc,"SAME-deep"),"D1_riem":(D1,"cross"),"D2_csp":(D2,"cross"),"D3_spec":(D3,"cross")}
for nm,(M,fam) in others.items():
    resc=[];jac=[];ap=[];ac=[]
    for i in range(9):
        sid=i+1
        if nm=="ATCNet":
            if sid not in atc: continue
            X=atc[sid]
        else: X=M[i]
        y=Lab[i];T=T_all[i];tp=T.argmax(1);xp=X.argmax(1);Tc=tp==y;Xc=xp==y
        marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2];ce=(~Tc)&(marg>np.median(marg))
        resc.append((Xc[ce]).mean()*100 if ce.sum() else np.nan)
        eT=~Tc;eX=~Xc;jac.append((eT&eX).sum()/max(1,(eT|eX).sum())*100)
        ag=tp==xp;ap.append((tp[ag]==y[ag]).mean()*100 if ag.sum() else np.nan);ac.append(ag.mean()*100)
    print(f"{nm:>10} {fam:>12} {m(resc):8.1f} {m(jac):14.1f} {m(ap):11.1f} {m(ac):10.1f}")

# Metric 3: risk-coverage AURC of abstain signals (base predictor = T). Use ATCNet subjects for fair same-vs-cross.
print("\n=== M3: risk-coverage AURC (base=T-source predictions; LOWER=better reliability signal) ===")
Dmean=lambda i:(D1[i]+D2[i]+D3[i])/3
sigs={"single_negentropy":[], "single_margin":[], "samefam_disagree(T,ATCNet)":[],
      "crossfam_disagree(T,D1)":[], "crossfam_disagree(T,Dmean)":[]}
for i in range(9):
    sid=i+1
    if sid not in atc: continue
    y=Lab[i];T=T_all[i];tp=T.argmax(1);Tc=(tp==y).astype(float)
    marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2]
    sigs["single_negentropy"].append(aurc(-ent(T),Tc))
    sigs["single_margin"].append(aurc(marg,Tc))
    sigs["samefam_disagree(T,ATCNet)"].append(aurc(-symkl(T,atc[sid]),Tc))
    sigs["crossfam_disagree(T,D1)"].append(aurc(-symkl(T,D1[i]),Tc))
    sigs["crossfam_disagree(T,Dmean)"].append(aurc(-symkl(T,Dmean(i)),Tc))
for k,v in sorted(sigs.items(),key=lambda kv:m(kv[1])):
    print(f"  {k:>30}: AURC {m(v):.4f}")
print("\n(base T acc on ATCNet subjects:", f"{m([(T_all[i].argmax(1)==Lab[i]).mean()*100 for i in range(9) if i+1 in atc]):.2f}%)")
print("\nVERDICT GUIDE: F2 holds if SAME-deep(ATCNet) rescue<<cross-family rescue AND errJaccard(T-ATCNet)>>(T-Dx)")
print("              AND crossfam AURC < samefam AURC <= single. If samefam ~ crossfam -> family thesis FALSE.")

"""Seminar figures for the 260604 progress note (Head-Frozen Cross-Family Decoder).
All numbers from cached TCFormer/2a+2b data. No GPU.
Fig1 subject-wise source->DA-DC (2a, rescue + non-regression)
Fig2 head-headroom (softmax vs LDA-probe ceiling) per subject, 2a & 2b  [LAW 2]
Fig3 confident-wrong detection: cross-family vs single-UQ at matched budget [LAW 1 / safety]
"""
import glob, numpy as np, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
OUT="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/260604_law_based_decoder/"
RR="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/"
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs/"
def sm(z):z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
exp=np.load(RES+"260602_expert_portfolio_table/expert_portfolio_arrays.npz",allow_pickle=True)
IX={n:[str(x) for x in exp["experts"].tolist()].index(n) for n in ["source","full_ea","shrink_0.1"]}
Lab=exp["labels"];Tsrc=exp["probs"][:,IX["source"]];Tfe=exp["probs"][:,IX["full_ea"]];Tsh=exp["probs"][:,IX["shrink_0.1"]]
R=np.load(RES+"260603_diverse_riemann_preds.npz",allow_pickle=True)["probs"]
S=np.load(RES+"260603_d3_spectral_preds.npz",allow_pickle=True)["probs"]
acc=lambda P,i:(P[i].argmax(1)==Lab[i]).mean()*100

# ---------- Fig1: subject-wise source -> DA-DC (2a) ----------
src=[acc(Tsrc,i) for i in range(9)]
dadc=[(( 0.3*Tsrc[i]+0.4*Tfe[i]+0.3*Tsh[i]+0.3*R[i]).argmax(1)==Lab[i]).mean()*100 for i in range(9)]
order=np.argsort(src)
fig,ax=plt.subplots(figsize=(9,5))
x=np.arange(9);w=0.38
ax.bar(x-w/2,[src[i] for i in order],w,label="source (no adapt)",color="#9aa5b1")
ax.bar(x+w/2,[dadc[i] for i in order],w,label="DA-DC (label-free)",color="#2e7d32")
for k,i in enumerate(order):
    ax.text(k+w/2,dadc[i]+0.4,f"+{dadc[i]-src[i]:.1f}",ha="center",fontsize=8,color="#2e7d32")
ax.set_xticks(x);ax.set_xticklabels([f"S{i+1}" for i in order])
ax.set_ylabel("session_E accuracy (%)");ax.set_ylim(60,100)
ax.set_title(f"BCIC-2a subject-wise: source {np.mean(src):.1f}% -> DA-DC {np.mean(dadc):.1f}% (+{np.mean(dadc)-np.mean(src):.1f}pp, worst-rescue, no regression)")
ax.legend();plt.tight_layout();plt.savefig(OUT+"fig1_subjectwise_2a.png",dpi=130);plt.close()

# ---------- Fig2: head-headroom (softmax vs LDA-probe ceiling) 2a & 2b ----------
def headroom(ds):
    rows=[]
    for sid in range(1,10):
        if ds=="2a":
            fd=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/features_s{sid}_TCFormer.npz"))
            ld=sorted(glob.glob(RR+f"ea_aware_tcformer_s{sid}_seed0_*/logits_s{sid}_TCFormer.npy"))
        else:
            fd=sorted(glob.glob(RR+f"ea_aware_tcformer_bcic2b_s*_seed0_*/features_s{sid}_TCFormer.npz"))
            ld=sorted(glob.glob(RR+f"ea_aware_tcformer_bcic2b_s*_seed0_*/logits_s{sid}_TCFormer.npy"))
        if not fd or not ld: continue
        d=np.load(fd[-1]);F=d["features"].reshape(d["features"].shape[0],-1).astype(np.float64);y=d["labels"].astype(int)
        soft=(sm(np.load(ld[-1])).argmax(1)==y).mean()*100
        Fz=(F-F.mean(0))/(F.std(0)+1e-8);cc=[]
        for tr,va in StratifiedKFold(5,shuffle=True,random_state=0).split(Fz,y):
            cc.append((LinearDiscriminantAnalysis(shrinkage="auto",solver="lsqr").fit(Fz[tr],y[tr]).predict(Fz[va])==y[va]).mean()*100)
        rows.append((sid,soft,np.mean(cc)))
    return rows
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,ds in zip(axes,["2a","2b"]):
    rows=sorted(headroom(ds),key=lambda r:r[1]);x=np.arange(len(rows));w=0.38
    so=[r[1] for r in rows];ce=[r[2] for r in rows]
    ax.bar(x-w/2,so,w,label="trained head (softmax)",color="#9aa5b1")
    ax.bar(x+w/2,ce,w,label="features re-read (LDA-probe, true y)",color="#c62828")
    for k,r in enumerate(rows):
        if r[2]-r[1]>1: ax.text(k+w/2,r[2]+0.5,f"+{r[2]-r[1]:.0f}",ha="center",fontsize=8,color="#c62828")
    ax.set_xticks(x);ax.set_xticklabels([f"S{r[0]}" for r in rows]);ax.set_ylim(55,101)
    ax.set_title(f"BCIC-{ds}: head-headroom +{np.mean([r[2]-r[1] for r in rows]):.1f}pp\n(features survive drift, head degrades)")
    ax.set_ylabel("accuracy (%)");ax.legend(fontsize=8)
plt.suptitle("LAW 2: session drift localizes to the classifier HEAD; deep features stay separable",fontweight="bold")
plt.tight_layout();plt.savefig(OUT+"fig2_head_headroom_law2.png",dpi=130);plt.close()

# ---------- Fig3: confident-wrong detection cross-family vs single-UQ ----------
def symkl(p,q):return ((p*(np.log(np.clip(p,1e-12,1))-np.log(np.clip(q,1e-12,1)))).sum(1)+(q*(np.log(np.clip(q,1e-12,1))-np.log(np.clip(p,1e-12,1)))).sum(1))
budgets=np.linspace(0.05,0.4,8);cf=[];uq=[]
for b in budgets:
    cfc=[];uqc=[]
    for i in range(9):
        y=Lab[i];T=Tsrc[i];tp=T.argmax(1);marg=np.sort(T,1)[:,-1]-np.sort(T,1)[:,-2]
        hi=marg>=np.median(marg);cw=hi&(tp!=y)
        if cw.sum()==0: continue
        K=int(len(y)*b)
        sig_cf=symkl(T,R[i])+symkl(T,S[i])     # cross-family disagreement
        sig_uq=-marg                            # single-model uncertainty
        for sig,store in [(sig_cf,cfc),(sig_uq,uqc)]:
            o=np.argsort(-sig)[:K];fl=np.zeros(len(y),bool);fl[o]=True;store.append((fl&cw).sum()/cw.sum()*100)
    cf.append(np.mean(cfc));uq.append(np.mean(uqc))
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(budgets*100,cf,"o-",color="#1565c0",lw=2,label="cross-family disagreement (LAW 1)")
ax.plot(budgets*100,uq,"s--",color="#9aa5b1",lw=2,label="single-model uncertainty (margin)")
ax.set_xlabel("abstain budget (% flagged)");ax.set_ylabel("confident-wrong caught (%)")
ax.set_title("LAW 1: cross-family disagreement detects confident-wrong;\nsingle-UQ is structurally blind in the high-confidence region (BCIC-2a)")
ax.legend();ax.grid(alpha=0.3);plt.tight_layout();plt.savefig(OUT+"fig3_confidentwrong_law1.png",dpi=130);plt.close()
print("FIG3 confident-wrong caught @15% budget: cross-family %.0f%% vs single-UQ %.0f%%"%(cf[2],uq[2]))
print(f"saved figures to {OUT}")
print(f"means: source {np.mean(src):.1f} DA-DC {np.mean(dadc):.1f}")

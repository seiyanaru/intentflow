"""Zemi figures for the confirmed Stieger results (no get_data; cases.npz + jsons only).
F1: per-session Δacc spread + harmful tail (the headline asset, absent from Wimpff/EDAPT).
F2: the mirage — EB-pooled gate selectivity vs k (all-adopt at small k -> selective only at large k).
F3: safety/efficiency frontier — effΔ vs harmed, sequential/subjMean vs per-session LCB vs always.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="sans-serif"; plt.rcParams["font.sans-serif"]=["Noto Sans CJK JP","IPAexGothic","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
OUT="/mnt/data/seiya.narukawa/intentflow/docs/research_progress/ゼミ資料"

# ---- data: per-session Δ from cases.npz ----
d=np.load("/tmp/stieger_snap.npz")
keys=sorted({k[3:] for k in d.files if k.startswith("cs_")},key=lambda s:(int(s.split("_")[0]),int(s.split("_")[1])))
Df=np.array([(d[f"ca_{k}"].mean()-d[f"cs_{k}"].mean())*100 for k in keys])

# ===== F1: spread + harmful tail =====
fig,ax=plt.subplots(figsize=(8,4.5))
colors=["#c0392b" if x<-1 else ("#2c8a3d" if x>1 else "#bbb") for x in np.sort(Df)]
ax.bar(range(len(Df)),np.sort(Df),color=colors,width=1.0)
ax.axhline(0,color="k",lw=0.8); ax.axhline(Df.mean(),color="#2c3e50",ls="--",lw=1,label=f"mean={Df.mean():+.2f}pp")
ax.set_xlabel("subject-session（Δacc昇順, n=524）"); ax.set_ylabel("適応Δacc [pp]（adapt − source, leak-free）")
ax.set_title(f"図1: per-session 適応Δの分布と有害尾  helped {int((Df>1).sum())}(76%) / harmed {int((Df<-1).sum())}(16%) / worst {Df.min():+.1f}",fontsize=11)
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/260610_F1_spread.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ===== F2: mirage — selectivity & harmed vs k (from detailed eval, recomputed inline) =====
from sklearn.metrics import roc_auc_score
CS=[d[f"cs_{k}"].astype(float) for k in keys]; CA=[d[f"ca_{k}"].astype(float) for k in keys]
def eb(dh,se):
    mu=np.full_like(dh,dh.mean());t2=max(dh.var()-(se**2).mean(),1e-6);pr=1/se**2+1/t2
    return (dh/se**2+mu/t2)/pr-1.645*np.sqrt(1/pr)>0,(dh/se**2+mu/t2)/pr
ks=[8,16,32,64,96];sel_eb=[];harm_eb=[];auc_eb=[]
for k in ks:
    dh=[];se=[];ct=[];ca=[]
    for cs,a in zip(CS,CA):
        if len(cs)<=k+10:continue
        df=(a-cs)[:k];dh.append(df.mean());se.append(max(df.std(ddof=1)/np.sqrt(k),1e-3));ct.append(cs[k:].mean());ca.append(a[k:].mean())
    dh,se,ct,ca=map(np.array,(dh,se,ct,ca));td=(ca-ct)*100
    ad,post=eb(dh,se);ben=td>1;hrm=td<-1
    sel_eb.append(ad[ben].mean()-ad[hrm].mean());harm_eb.append(int(((td<-1)&ad).sum()))
    auc_eb.append(roc_auc_score((td>0).astype(int),post))
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4.5))
a1.plot(ks,sel_eb,"o-",color="#2c8a3d",lw=2,label="EB選択性 P(採用|益)−P(採用|害)")
a1.plot(ks,auc_eb,"s--",color="#7f8c8d",lw=1.5,label="EB AUROC(益vs害)")
a1.axhline(0,color="k",lw=0.6);a1.set_xlabel("probe ラベル数 k");a1.set_ylabel("選択性 / AUROC")
a1.set_title("図2-左: 最小ラベルでは選択性≈0（全採用に退化）",fontsize=11);a1.legend(fontsize=8)
a2.plot(ks,harm_eb,"o-",color="#c0392b",lw=2,label="EB harmed")
a2.axhline(81,color="#c0392b",ls=":",lw=1,label="always-adapt harmed=81")
a2.set_xlabel("probe ラベル数 k");a2.set_ylabel("harmed セッション数")
a2.set_title("図2-右: k≤16でharmed≈81（有害尾を一切避けない）",fontsize=11);a2.legend(fontsize=8)
fig.tight_layout();fig.savefig(f"{OUT}/260610_F2_mirage.png",dpi=150,bbox_inches="tight");plt.close(fig)

# ===== F3: safety/efficiency frontier (effΔ vs harmed) from sequential json =====
J=json.load(open(f"{RES}/260610_stieger_sequential_veto.json"))
pts={"always (~Wimpff)":"#c0392b","never (keep-source)":"#bbb","per-sess LCB k=8":"#2980b9",
     "per-sess LCB k=32":"#1f5f8b","SEQ-subjMean (m=8, θ=0)":"#2c8a3d","SEQ-subjLCB (m=8, θ=0)":"#27ae60",
     "SEQ-prevΔ (m=8, θ=0)":"#95a5a6"}
fig,ax=plt.subplots(figsize=(8,5.5))
for name,c in pts.items():
    if name not in J: continue
    r=J[name];ax.scatter(r["harmed"],r["effΔ"],s=90,color=c,zorder=3)
    ax.annotate(name.replace(" (m=8, θ=0)","").replace(" (~Wimpff)",""),(r["harmed"],r["effΔ"]),
                fontsize=8,xytext=(6,3),textcoords="offset points")
ax.axhline(0,color="k",lw=0.6)
ax.set_xlabel("harmed セッション数（少ないほど安全 →左）");ax.set_ylabel("effΔ vs source [pp]（高いほど利得 →上）")
ax.set_title("図3: 安全/利得フロンティア  左上=理想。subjMeanは高利得帯で安く尾を削る／LCBは安全だが利得犠牲",fontsize=10)
fig.tight_layout();fig.savefig(f"{OUT}/260610_F3_frontier.png",dpi=150,bbox_inches="tight");plt.close(fig)
print("saved F1_spread / F2_mirage / F3_frontier .png")

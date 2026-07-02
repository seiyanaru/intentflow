"""Zemi 260622 figures (real data only):
 G_auroc: label-free signals are weak at flagging harmful sessions (single<0.62, fused 0.66).
 G_subject: harm concentrates by subject (per-subject mean Delta; 29/62 never harmed, 7 net-negative) + ICC.
Reads 260610_stieger_signals_auroc.json and 260610_stieger_signals.npy.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"]="sans-serif"; plt.rcParams["font.sans-serif"]=["Noto Sans CJK JP","IPAexGothic","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"]=False
RES="/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
OUT="/mnt/data/seiya.narukawa/intentflow/docs/research_progress/ゼミ資料"

# ---------- G_auroc ----------
a=json.load(open(f"{RES}/260610_stieger_signals_auroc.json"))
order=["H","overrule","conf_drop","entropy","dispersity","riemann","loso_combo"]
lab={"H":"確信オーバールール","overrule":"不一致率","conf_drop":"確信低下","entropy":"エントロピー",
     "dispersity":"分かれ具合","riemann":"共分散ドリフト","loso_combo":"6指標を融合"}
vals=[a[k]["auroc"] for k in order]
cols=["#7f8c8d"]*6+["#2c3e50"]
fig,ax=plt.subplots(figsize=(8.5,4.6))
b=ax.bar([lab[k] for k in order],vals,color=cols)
ax.axhline(0.5,color="#888",ls=":",lw=1); ax.text(6.3,0.505,"偶然(0.5)",fontsize=8,color="#888")
ax.axhline(0.65,color="#c0392b",ls="--",lw=1.2); ax.text(0,0.658,"実用に欲しい水準(0.65)",fontsize=8.5,color="#c0392b")
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+0.006,f"{v:.2f}",ha="center",fontsize=9)
ax.set_ylim(0.45,0.75); ax.set_ylabel("有害セッション判別 AUROC（正解なし）")
ax.set_title("正解なしの指標では、危険セッションを十分に見分けられない（n=536, 有害87）",fontsize=11.5)
plt.xticks(rotation=18,ha="right",fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/260622_G_labelfree_auroc.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- G_subject ----------
rows=np.load(f"{RES}/260610_stieger_signals.npy",allow_pickle=True)
sub=np.array([r['sub'] for r in rows]); td=np.array([r['true_d'] for r in rows])
usub=sorted(set(sub.tolist()))
subj_mean=np.array([td[sub==s].mean() for s in usub])
order_s=np.argsort(subj_mean)
# ICC(1)
grand=td.mean();k=len(usub);n=len(td)
groups=[td[sub==s] for s in usub];ni=np.array([len(g) for g in groups])
MSB=sum(len(g)*(g.mean()-grand)**2 for g in groups)/(k-1)
MSW=sum(((g-g.mean())**2).sum() for g in groups)/(n-k)
k0=(n-(ni**2).sum()/n)/(k-1); icc=(MSB-MSW)/(MSB+(k0-1)*MSW)
never=sum(1 for s in usub if (td[sub==s]<-1).sum()==0)
netneg=int((subj_mean<0).sum())
fig,ax=plt.subplots(figsize=(10,4.6))
sm=subj_mean[order_s]
cols=["#c0392b" if v<0 else ("#2c8a3d" if v>1 else "#bbb") for v in sm]
ax.bar(range(k),sm,color=cols,width=0.9)
ax.axhline(0,color="k",lw=0.8)
ax.set_xlabel(f"被験者（適応の平均効果で昇順, n={k}）"); ax.set_ylabel("その被験者での平均Δacc [pp]")
ax.set_title(f"適応の害は被験者に偏在：平均で損する被験者は{netneg}人 / 一度も壊れない被験者は{never}人  (ICC={icc:.2f})",fontsize=11.5)
ax.annotate(f"赤={netneg}人：平均で損する被験者", xy=(3,sm[3]), xytext=(8,sm.min()-1.5),
            fontsize=9.5,color="#c0392b",arrowprops=dict(arrowstyle="->",color="#c0392b",lw=1))
ax.text(k*0.62,2,"緑：平均で改善する被験者",fontsize=9.5,color="#2c8a3d")
ax.set_ylim(sm.min()-3.5,sm.max()+2)
fig.tight_layout(); fig.savefig(f"{OUT}/260622_F_subject_concentration.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- C2: 平均はほぼ頭打ち / worst-case は大きく割れる（二軸対比） ----------
aniso=json.load(open(f"{RES}/260612_stieger_aniso.json"))
src=0.0  # source基準(Δ=0)
ea=aniso["conditions"]["ea"]; ceil=aniso["oracle_EAvsSrc"]
names=["何もしない\n(source)","常に適応\n(always-adapt)","達成可能上限\n(oracle)"]
meanv=[0.0, ea["meanD"], ceil]
worstv=[0.0, ea["worst"], 0.0]   # source/oracle は壊さない=worst 0
harmedv=[0, ea["harmed"], 0]
fig,(axL,axR)=plt.subplots(1,2,figsize=(12,4.4))
axL.bar(names,meanv,color=["#bbb","#2c3e50","#888"]); axL.axhline(0,color="k",lw=0.6)
for i,v in enumerate(meanv): axL.text(i,v+0.1,f"+{v:.1f}",ha="center",fontsize=10)
axL.set_ylabel("平均Δacc [pp]"); axL.set_title("平均精度：常に適応で上限の93%に到達（ほぼ頭打ち）",fontsize=11)
axR.bar(names,harmedv,color=["#bbb","#c0392b","#888"])
for i,v in enumerate(harmedv): axR.text(i,v+1,f"{v}",ha="center",fontsize=10)
axR.set_ylabel("壊れたセッション数（worstほど悪い）"); axR.set_title("worst-case：常に適応は87セッションを悪化（最悪−12pp）",fontsize=11)
fig.suptitle("平均は解けている一方、worst-caseは大きく割れる → 目標をworst-caseに置く",fontsize=12.5)
fig.tight_layout(); fig.savefig(f"{OUT}/260622_C2_mean_vs_worst.png",dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------- G: 異方的縮約の棄却（meanD vs harmed フロンティア） ----------
C=aniso["conditions"]
def pt(k): return C[k]["harmed"],C[k]["meanD"]
fig,ax=plt.subplots(figsize=(7.5,5.2))
sc=[("scalar_0.3","#2980b9","一律縮約"),("scalar_0.5","#2980b9",None),("scalar_0.7","#2980b9",None)]
am=[("anisoM_0.5","#c0392b","方向別縮約(提案)"),("anisoM_1.0","#c0392b",None),("anisoM_2.0","#c0392b",None)]
for k,c,l in sc: ax.scatter(*pt(k),s=80,color=c,marker="s",label=l,zorder=3)
for k,c,l in am: ax.scatter(*pt(k),s=80,color=c,marker="o",label=l,zorder=3)
ax.scatter(*pt("ea"),s=110,color="#2c3e50",marker="*",label="常に適応(EA)",zorder=4)
# frontier line for scalar
sx=[C[k]["harmed"] for k,_,_ in sc]; sy=[C[k]["meanD"] for k,_,_ in sc]
o=np.argsort(sx); ax.plot(np.array(sx)[o],np.array(sy)[o],color="#2980b9",lw=1,ls="--",alpha=0.6)
ax.set_xlabel("壊れたセッション数（少ないほど安全 →左）"); ax.set_ylabel("平均Δacc [pp]（高いほど良い →上）")
ax.set_title("方向別の縮約(提案)は一律縮約を上回れない＝棄却\n（同じ安全度なら方向別の方が利得が低い）",fontsize=11)
ax.legend(fontsize=8.5,loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/260622_G_aniso_reject.png",dpi=150,bbox_inches="tight"); plt.close(fig)

print("saved: F_subject / G_labelfree_auroc / C2_mean_vs_worst / G_aniso_reject")
print(f"check: ICC={icc:.3f}, never-harmed={never}/{k}, net-negative={netneg}/{k}")

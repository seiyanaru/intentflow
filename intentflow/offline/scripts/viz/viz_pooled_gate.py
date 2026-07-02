"""Fig E: pooled gate result. Left = %oracle-gain captured (k=16) by 4 deciders x 3 adapters
(EB-pooled recovers gain where per-session/global fail). Right = effective Δacc {never/always/
EB-pooled/oracle} showing EB captures 2a's +4.5 AND avoids Lee2019's -11 crash (harmed annotated).
Reads 260609_pooled_gate.json. Leak-free numbers.
"""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
OUT = "/mnt/data/seiya.narukawa/intentflow/docs/research_progress/ゼミ資料"
R = json.load(open(f"{RES}/260609_pooled_gate.json"))
adapters = list(R.keys()); K = "16"
methods = ["per-session LCB", "global", "EB-pooled", "EB+H"]
cols = {"per-session LCB": "#bbb", "global": "#7f8c8d", "EB-pooled": "#2c8a3d", "EB+H": "#27ae60"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.2))

# Left: %oracle captured, grouped bars
x = np.arange(len(adapters)); w = 0.2
for j, m in enumerate(methods):
    vals = [max(R[a][K][m]["pct_oracle"], 0) for a in adapters]
    harm = [R[a][K][m]["harmed"] for a in adapters]
    bars = axL.bar(x + (j-1.5)*w, vals, w, label=m, color=cols[m])
    for i, b in enumerate(bars):
        if harm[i] > 0: axL.text(b.get_x()+b.get_width()/2, b.get_height()+2, f"害{harm[i]}", ha="center", fontsize=7, color="#c0392b")
axL.set_xticks(x); axL.set_xticklabels([a.split(" ")[0]+"\n"+a.split("(")[-1].rstrip(")") for a in adapters], fontsize=9)
axL.set_ylabel("oracle利得の捕捉率 [%]（k=16, リーク無）"); axL.axhline(0, color="k", lw=0.8)
axL.set_title("図E-左: EB-poolingが利得を回収（per-session/globalは0%）", fontsize=12)
axL.legend(fontsize=8, loc="upper right")
for i, a in enumerate(adapters):  # note always-adapt Δ (crash on bad adapters)
    axL.text(x[i], -8, f"常時適応={R[a][K]['per-session LCB']['always']:+.1f}pp", ha="center", fontsize=8, color="#c0392b")

# Right: effective Δacc bars {never, always, EB-pooled, oracle}
w2 = 0.2
nev = [0]*len(adapters); alw = [R[a][K]['EB-pooled']['always'] for a in adapters]
ebv = [R[a][K]['EB-pooled']['effΔ'] for a in adapters]; ora = [R[a][K]['EB-pooled']['oracle'] for a in adapters]
ebh = [R[a][K]['EB-pooled']['harmed'] for a in adapters]
axR.bar(x-1.5*w2, nev, w2, label="何もしない", color="#bbb")
axR.bar(x-0.5*w2, alw, w2, label="常に適応", color="#c0392b")
axR.bar(x+0.5*w2, ebv, w2, label="EB-pooled ゲート(k=16)", color="#2c8a3d")
axR.bar(x+1.5*w2, ora, w2, label="oracle(上限)", color="#2c3e50", alpha=0.5)
for i in range(len(adapters)):
    axR.text(x[i]+0.5*w2, ebv[i], f"害{ebh[i]}", ha="center", va="bottom", fontsize=8, color="#2c8a3d")
axR.axhline(0, color="k", lw=0.8); axR.set_xticks(x)
axR.set_xticklabels([a.split(" ")[0] for a in adapters], fontsize=9)
axR.set_ylabel("source比 平均Δacc [pp]")
axR.set_title("図E-右: EBは2aの+4.5を回収しLee2019の−11崩壊を回避（害0）", fontsize=12)
axR.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/260609_safegate_E_pooled.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("saved", f"{OUT}/260609_safegate_E_pooled.png")

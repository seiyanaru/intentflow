"""Visualize the safe-selective-adaptation findings (de-blackbox). 4 focused figures:
  A: per-session adaptation Δacc SPREAD (2a DA-DC good / Lee2019 Riemann bad / blend) — is there harm to gate?
  B: gate comparison (never / always-adapt / oracle / ATTA-LCB k=16) effective Δ + harmed count — the safety result
  C: ATTA-LCB %oracle-gain captured vs k, per adapter — "safe but conservative"
  D: label-free signal vs Δ (overrule-mass H, LEEP) on Lee2019 — what predicts harm (fusion prior)
Saves PNGs to docs/research_progress/ゼミ資料/. CPU; uses cached probs.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAexGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
OUT = "/mnt/data/seiya.narukawa/intentflow/docs/research_progress/ゼミ資料"
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(0)
EPS = 1e-12

# ---------- load adapters as per-session (c_s, c_a) + label-free signals ----------
def disp(P): sv = np.linalg.svd(P, compute_uv=False); return float(sv.sum()/np.sqrt(P.shape[0]*P.shape[1]))
def leep(Ps, yhat):
    C = Ps.shape[1]; th = np.zeros((C, C))
    for c in range(C): th[c] = Ps[yhat == c].sum(0) if (yhat == c).any() else 0
    th = th/(th.sum(0, keepdims=True)+EPS); pj = Ps @ th.T
    return float(np.mean(np.log(pj[np.arange(len(yhat)), yhat]+EPS)))

# 2a DA-DC (good)
exp = np.load(f"{RES}/260602_expert_portfolio_table/expert_portfolio_arrays.npz", allow_pickle=True)
nm = [str(x) for x in exp["experts"].tolist()]; IX = {n: nm.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
Lab = exp["labels"]; src = exp["probs"][:, IX["source"]]; fe = exp["probs"][:, IX["full_ea"]]; sh = exp["probs"][:, IX["shrink_0.1"]]
R2a = np.load(f"{RES}/260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
def build_2a():
    S = []
    for s in range(9):
        dadc = 0.3*src[s]+0.4*fe[s]+0.3*sh[s]+0.3*R2a[s]
        S.append(dict(c_s=(src[s].argmax(1) == Lab[s]).astype(float), c_a=(dadc.argmax(1) == Lab[s]).astype(float),
                      H=float(((src[s].max(1))*(dadc.argmax(1) != src[s].argmax(1))).mean()), leep=leep(src[s], dadc.argmax(1))))
    return S
# Lee2019 Riemann / blend (bad)
d = np.load(f"{RES}/260609_lee2019_adapter_probs.npz")
subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("Ps_")})
def build_lee(which):
    S = []
    for s in subs:
        Ps = d[f"Ps_{s}"].astype(np.float64); Pr = d[f"Pr_{s}"].astype(np.float64); y = d[f"y_{s}"]
        Pa = Pr if which == "riemann" else 0.5*Ps+0.5*Pr
        S.append(dict(c_s=(Ps.argmax(1) == y).astype(float), c_a=(Pa.argmax(1) == y).astype(float),
                      H=float((Ps.max(1)*(Pa.argmax(1) != Ps.argmax(1))).mean()), leep=leep(Ps, Pa.argmax(1))))
    return S
A2a, ARi, ABl = build_2a(), build_lee("riemann"), build_lee("blend")
ADS = {"2a DA-DC（良いアダプタ）": A2a, "Lee2019 Riemann（危険）": ARi, "Lee2019 blend（やや危険）": ABl}

def delta(S): return np.array([(x['c_a'].mean()-x['c_s'].mean())*100 for x in S])
def clean_eval(S, k):
    """LEAK-FREE: probe = FIRST k trials (cued calibration); decide via LCB; evaluate ONLY on the
    remaining n-k held-out trials. always/oracle also on held-out for a fair comparison."""
    eff = []; nev = []; alw = []; ora = []; harmed = 0; adapt = 0
    for x in S:
        cs, ca = x['c_s'], x['c_a']; n = len(cs)
        if n <= k: continue
        diff = (ca-cs)[:k]; se = diff.std(ddof=1)/np.sqrt(k) if k > 1 else 1.0
        do = (diff.mean()-1.645*se) > 0; adapt += do
        cst, cat = cs[k:].mean(), ca[k:].mean()
        eff.append((cat if do else cst)*100); nev.append(cst*100); alw.append(cat*100); ora.append(max(cst, cat)*100)
        if do and (cat-cst)*100 < -1: harmed += 1
    base = np.mean(nev)
    return dict(eff=np.mean(eff)-base, always=np.mean(alw)-base, oracle=np.mean(ora)-base,
                harmed=harmed, alwaysHarmed=int((np.array(alw)-np.array(nev) < -1).sum()), adaptRate=adapt/len(eff))
def atta(S, k): r = clean_eval(S, k); return r['eff'], r['harmed'], r['adaptRate']
def bounds(S): r = clean_eval(S, 16); return r['always'], r['oracle']

# ---------- Fig A: per-session Δ spread ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, S) in zip(axes, ADS.items()):
    dd = np.sort(delta(S))[::-1]
    colors = ["#2c8a3d" if v > 1 else ("#c0392b" if v < -1 else "#999") for v in dd]
    ax.bar(range(len(dd)), dd, color=colors)
    ax.axhline(0, color="k", lw=0.8); ax.axhline(dd.mean(), color="#2c3e50", ls="--", lw=1.2, label=f"平均 {dd.mean():+.1f}")
    nh = int((dd < -1).sum()); ng = int((dd > 1).sum())
    ax.set_title(f"{name}\n改善{ng} / 悪化{nh} / n={len(dd)}", fontsize=11)
    ax.set_xlabel("被験者（Δ降順）"); ax.set_ylabel("適応Δacc [pp]"); ax.legend(fontsize=9)
fig.suptitle("図A: 適応の利得/害は被験者ごとに大きくばらつく（＝ゲートする対象が実在）", fontsize=13, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/260609_safegate_A_spread.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Fig B: gate comparison ----------
fig, ax = plt.subplots(figsize=(10, 5))
labels = list(ADS.keys()); x = np.arange(len(labels)); w = 0.2
ce = {name: clean_eval(S, 16) for name, S in ADS.items()}
alw = [ce[n]['always'] for n in labels]; ora = [ce[n]['oracle'] for n in labels]
lcb = [ce[n]['eff'] for n in labels]; lcb_h = [ce[n]['harmed'] for n in labels]; alw_h = [ce[n]['alwaysHarmed'] for n in labels]
ax.bar(x-1.5*w, [0]*len(labels), w, label="何もしない(source)", color="#bbb")
ax.bar(x-0.5*w, alw, w, label="常に適応", color="#c0392b")
ax.bar(x+0.5*w, lcb, w, label="ATTA-LCB ゲート(k=16, リーク無)", color="#2c8a3d")
ax.bar(x+1.5*w, ora, w, label="oracle(上限)", color="#2c3e50", alpha=0.5)
for i in range(len(labels)):
    ax.text(x[i]-0.5*w, alw[i], f"害{alw_h[i]}", ha="center", va="top" if alw[i] < 0 else "bottom", fontsize=8, color="#c0392b")
    ax.text(x[i]+0.5*w, lcb[i], f"害{lcb_h[i]}", ha="center", va="bottom", fontsize=8, color="#2c8a3d")
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("source比 平均Δacc [pp]（probe除外・残りで評価）")
ax.set_title("図B(リーク無): ATTA-LCBは害≈0だが、良アダプタ(2a)の利得も取れず ≈『常に不適応』", fontsize=12)
ax.legend(fontsize=9, loc="lower left"); fig.tight_layout()
fig.savefig(f"{OUT}/260609_safegate_B_gate.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Fig C: %oracle-gain captured vs k (safe but conservative) ----------
fig, ax = plt.subplots(figsize=(9, 5)); ks = [8, 16, 32, 64]
for name, S in ADS.items():
    pct = []
    for k in ks:
        r = clean_eval(S, k); pct.append(r['eff']/r['oracle']*100 if r['oracle'] > 1e-6 else np.nan)
    ax.plot(ks, pct, "o-", label=name, lw=2)
ax.axhline(100, color="#2c3e50", ls=":", lw=1, label="oracle(=取りこぼし0)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("セッションあたりラベル数 k"); ax.set_ylabel("oracle利得の捕捉率 [%]（リーク無）")
ax.set_title("図C(リーク無): 最小ラベルでは利得をほぼ取れない — per-session確証が不能で≈不適応", fontsize=12)
ax.set_xticks(ks); ax.legend(fontsize=9); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT}/260609_safegate_C_conservative.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- Fig D: label-free signal vs Δ (Lee2019 Riemann) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
dRi = delta(ARi); H = np.array([x['H'] for x in ARi]); LE = np.array([x['leep'] for x in ARi])
for ax, v, lab, rho in [(axes[0], H, "確信オーバールール質量 H", spearmanr(H, dRi).correlation),
                        (axes[1], LE, "LEEP（転移度）", spearmanr(LE, dRi).correlation)]:
    c = ["#2c8a3d" if t > 1 else ("#c0392b" if t < -1 else "#999") for t in dRi]
    ax.scatter(v, dRi, c=c, s=40, edgecolor="k", lw=0.3); ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(lab); ax.set_ylabel("適応Δacc [pp]"); ax.set_title(f"{lab}\nSpearman ρ={rho:+.2f}", fontsize=11)
fig.suptitle("図D: 実害が大きい時、label-free信号が害を当てる（融合ゲートの事前分布候補）", fontsize=12, y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/260609_safegate_D_labelfree.png", dpi=150, bbox_inches="tight"); plt.close(fig)

print("saved 4 figures to", OUT)
for f in ["A_spread", "B_gate", "C_conservative", "D_labelfree"]:
    print("  260609_safegate_"+f+".png")

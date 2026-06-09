"""FB4 / LEVER-2 salvage precheck: E4 showed clusterability LOSES at predicting ACCURACY.
But the gate's real job is predicting ADAPTATION BENEFIT ("will adapting help?"), which is a
DIFFERENT target. Two questions:
  (1) Is per-session adaptation benefit a NON-DEGENERATE target (enough spread to predict)?
  (2) Does clusterability predict WHO-benefits better than accuracy-estimators (nuc_dispersity/conf)?
If yes, "worthiness != accuracy" is a live salvage even with E4 BAD. If benefit ~ constant, the
target is degenerate and this lever dies (report honestly).
2a: DA-DC gain (blend - source) per subject. 2b: EA-benefit (full_ea - source) per subject (bimodal, S5 hurt).
CPU only.
"""
import numpy as np, warnings, os, csv
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr
RES = "intentflow/offline/results/research_outputs/"
EPS = 1e-12
def disp(P): sv = np.linalg.svd(P, compute_uv=False); return float(sv.sum() / np.sqrt(P.shape[0] * P.shape[1]))
def conf(P): return float(P.max(1).mean())
def negent(P): return float(-(-(P * np.log(P + EPS)).sum(1)).mean())  # -mean entropy
def sil(F, k): Z = (F - F.mean(0)) / (F.std(0) + 1e-8); return float(silhouette_score(Z, KMeans(k, n_init=5, random_state=0).fit_predict(Z)))

print("===== FB4: adaptation-WORTHINESS (predict BENEFIT, not accuracy) =====")

# ---------- 2a: DA-DC gain ----------
try:
    exp = np.load(RES + "260602_expert_portfolio_table/expert_portfolio_arrays.npz", allow_pickle=True)
    names = [str(x) for x in exp["experts"].tolist()]; IX = {n: names.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
    Lab = exp["labels"]; src = exp["probs"][:, IX["source"]]; fe = exp["probs"][:, IX["full_ea"]]; sh = exp["probs"][:, IX["shrink_0.1"]]
    R = np.load(RES + "260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
    d = np.load("intentflow/offline/results/source_train_features_s0.npz", allow_pickle=True)
    cl = []; gain = []; cdisp = []; cconf = []; cneg = []; drift = []
    for s in range(1, 10):
        Fte = d[f"eval_feat_{s}"]; Ftr = d[f"train_feat_{s}"]
        cl.append(sil(Fte, 4)); drift.append(float(np.linalg.norm(Ftr.mean(0) - Fte.mean(0))))
        dadc = 0.3 * src[s - 1] + 0.4 * fe[s - 1] + 0.3 * sh[s - 1] + 0.3 * R[s - 1]
        so = (src[s - 1].argmax(1) == Lab[s - 1]).mean() * 100; da = (dadc.argmax(1) == Lab[s - 1]).mean() * 100
        gain.append(da - so); cdisp.append(disp(dadc)); cconf.append(conf(dadc)); cneg.append(negent(dadc))
    gain = np.array(gain)
    print(f"\n[2a] DA-DC gain: mean={gain.mean():+.2f} std={gain.std():.2f} range[{gain.min():+.1f},{gain.max():+.1f}] harmed={int((gain<0).sum())}/9")
    print(f"     TARGET QUALITY: {'DEGENERATE (std<1.0, all-positive => weak prediction target)' if gain.std()<1.0 else 'usable spread'}")
    for nm, v in [("clusterability", cl), ("nuc_dispersity", cdisp), ("conf", cconf), ("negentropy", cneg), ("drift", drift)]:
        print(f"     Spearman({nm:>15}, gain) = {spearmanr(v, gain).correlation:+.3f}")
except Exception as e:
    print("[2a] FB4 FAILED:", str(e)[:140])

# ---------- 2b: EA-benefit (bimodal: S5 catastrophe) ----------
try:
    rows = {int(r["subject"]): r for r in csv.DictReader(open(RES + "260602_bcic2b_portfolio_seed0/subject_summary.csv"))}
    npz2b = RES + "260608_2b_pertrial.npz"
    if os.path.exists(npz2b):
        d2 = np.load(npz2b)
        cl = []; eab = []
        for s in range(1, 10):
            cl.append(sil(d2[f"F1_{s}"], 2))
            eab.append(float(rows[s]["acc_full_ea"]) - float(rows[s]["acc_source"]))
        eab = np.array(eab)
        print(f"\n[2b] EA-benefit: mean={eab.mean():+.2f} std={eab.std():.2f} range[{eab.min():+.1f},{eab.max():+.1f}] hurt(<-5)={int((eab<-5).sum())}/9")
        print(f"     TARGET QUALITY: {'usable spread (bimodal harm => good worthiness target)' if eab.std()>=2 else 'low spread'}")
        print(f"     Spearman(clusterability, EA-benefit) = {spearmanr(cl, eab).correlation:+.3f}  [does clusterability flag where EA HURTS?]")
    else:
        print("\n[2b] skipped (260608_2b_pertrial.npz not yet built)")
except Exception as e:
    print("[2b] FB4 FAILED:", str(e)[:140])

print("\nINTERPRET: salvage ALIVE if clusterability-vs-benefit Spearman BEATS nuc_dispersity/conf-vs-benefit")
print("           (worthiness != accuracy). DEAD if benefit is degenerate or clusterability still loses.")

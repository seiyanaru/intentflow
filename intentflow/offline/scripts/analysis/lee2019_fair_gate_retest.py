"""Fair GATE re-test against a REAL adapter's per-session benefit/harm (GPT holes #2,#4):
reads 260609_lee2019_adapter_probs.npz (Ps source / Pr Riemann / y per subject) from the spread run.
For adapter in {riemann, blend(0.5/0.5)} tests, on Lee2019 n=54:
  LABEL-FREE gates (predict 'should adapt' = Δ>0): confident-overrule mass H, conf, dispersity, LEEP
     -> AUROC(helped) / AUROC(harmed) / Spearman(score,Δ).  [transferability LEEP added per GPT]
  ATTA-style PROBE-VETO (k labels): estimate per-session Δ on k trials, one-sided lower confidence
     bound LCB=Δ̂-1.645*SE, ADAPT iff LCB>0 -> effective accuracy + harmed-count vs always/never/oracle.
This is the principled probe-veto GPT flagged (statistical LCB, not simple agreement).
CPU. Run after lee2019_riemann_adapter_spread.py produces the probs npz.
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
from scipy.stats import spearmanr
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
P = f"{RES}/260609_lee2019_adapter_probs.npz"
if not os.path.exists(P):
    raise SystemExit(f"{P} not found -- run lee2019_riemann_adapter_spread.py first")
d = np.load(P); DUMP = np.load(f"{RES}/260608_lee2019_pertrial_tcformer.npz")
subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("Ps_")})
rng = np.random.RandomState(0)
EPS = 1e-12

def auroc(score, lab):
    lab = np.asarray(lab)
    if lab.sum() == 0 or lab.sum() == len(lab): return float('nan')
    order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score)+1)
    n1 = lab.sum(); n0 = len(lab)-n1
    return float((r[lab == 1].sum() - n1*(n1+1)/2)/(n1*n0))

def leep(Ps, yhat):  # LEEP transferability: target pseudo-labels yhat vs source soft preds Ps
    C = Ps.shape[1]; n = len(yhat)
    theta = np.zeros((C, C))
    for c in range(C): theta[c] = Ps[yhat == c].sum(0) if (yhat == c).any() else 0
    theta = theta / (theta.sum(0, keepdims=True) + EPS)        # p(target_class | source_pred dist)
    pj = Ps @ theta.T                                          # p(y|x) marginalized
    return float(np.mean(np.log(pj[np.arange(n), yhat] + EPS)))

def disp(Pp): sv = np.linalg.svd(Pp, compute_uv=False); return float(sv.sum()/np.sqrt(Pp.shape[0]*Pp.shape[1]))

for ADAP in ["riemann", "blend"]:
    sess = []
    for s in subs:
        Ps = d[f"Ps_{s}"].astype(np.float64); Pr = d[f"Pr_{s}"].astype(np.float64); y = d[f"y_{s}"]
        Pa = Pr if ADAP == "riemann" else 0.5*Ps + 0.5*Pr
        c_s = (Ps.argmax(1) == y).astype(float); c_a = (Pa.argmax(1) == y).astype(float)
        diff = c_a - c_s
        delta = diff.mean()*100
        ov = Pa.argmax(1) != Ps.argmax(1)
        H = float((Ps.max(1)*ov).mean())
        sess.append(dict(s=s, delta=delta, H=H, conf=float(Ps.max(1).mean()), disp=disp(Ps),
                         leep=leep(Ps, Pa.argmax(1)), diff=diff, c_s=c_s, c_a=c_a))
    Δ = np.array([x["delta"] for x in sess])
    helped = (Δ > 1).astype(int); harmed = (Δ < -1).astype(int)
    print("="*80)
    print(f"ADAPTER = {ADAP}:  meanΔ={Δ.mean():+.2f} helped={int(helped.sum())} harmed={int(harmed.sum())} "
          f"worst={Δ.min():+.1f} best={Δ.max():+.1f}  (n={len(subs)})")
    print("  LABEL-FREE gate -> can it pick helped / avoid harmed sessions?")
    print(f"  {'gate':>14} {'AUROC(helped)':>14} {'AUROC(harmed↓)':>15} {'rho(score,Δ)':>13}")
    for g in ["H", "conf", "disp", "leep"]:
        v = np.array([x[g] for x in sess])
        # for harmed we want LOW score=harmed, so AUROC(-v, harmed); for helped high score=helped
        print(f"  {g:>14} {auroc(v,helped):>14.3f} {auroc(-v,harmed):>15.3f} {spearmanr(v,Δ).correlation:>+13.3f}")
    # ATTA-LCB probe-veto
    print("  ATTA-LCB PROBE-VETO (adapt iff lower-conf-bound of per-session Δ > 0):")
    print(f"  {'k':>4} {'effAcc':>7} {'Δvssrc':>7} {'harmed':>7} {'adaptRate':>9}  (vs never={0:.0f} always/oracle below)")
    base_s = np.mean([x['c_s'].mean() for x in sess])*100
    base_a = np.mean([x['c_a'].mean() for x in sess])*100
    oracle = np.mean([max(x['c_s'].mean(), x['c_a'].mean()) for x in sess])*100
    for k in [4, 8, 16, 32]:
        eff = []; harmed_ct = 0; adapt_ct = 0
        for x in sess:
            n = len(x['diff'])
            idx = rng.permutation(n)[:min(k, n)]
            dk = x['diff'][idx]
            se = dk.std(ddof=1)/np.sqrt(len(dk)) if len(dk) > 1 else 1.0
            lcb = dk.mean() - 1.645*se
            adapt = lcb > 0
            adapt_ct += adapt
            acc = x['c_a'].mean() if adapt else x['c_s'].mean()
            eff.append(acc*100)
            if adapt and x['delta'] < -1: harmed_ct += 1
        print(f"  {k:>4} {np.mean(eff):>7.2f} {np.mean(eff)-base_s:>+7.2f} {harmed_ct:>7d} {adapt_ct/len(sess):>9.2f}")
    print(f"  baselines: never-adapt(source)={base_s:.2f}  always-adapt={base_a:+.2f}({base_a-base_s:+.2f})  oracle={oracle:.2f}(+{oracle-base_s:.2f})")
print("="*80)
print("READ: label-free gate works if AUROC(helped)>>0.5 AND it avoids harmed. ATTA-LCB works if effAcc")
print(">= always-adapt with harmed≈0 (it safely captures gain). If ATTA-LCB only matches never-adapt, the")
print("k-probe can't separate either => harm/benefit is genuinely label-light-unidentifiable here.")

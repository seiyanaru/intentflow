"""Is the k-shot ATTA-LCB probe-veto SELECTIVE (adapt good, veto bad) or merely CONSERVATIVE
(always reject)? Decisive cross-adapter test:
  - 2a DA-DC = a GOOD adapter (+4.4, 0/9 harmed). Does ATTA-LCB KEEP most of the gain?
  - Lee2019 Riemann = a BAD adapter (-11.24, 43/54 harmed). Does ATTA-LCB VETO it (harmed 0)?
If 2a Δ stays near always-adapt (+4.4) AND Lee2019 harmed=0, the gate is genuinely selective+safe
(= minimal-label monotone-safe selective adaptation). CPU; uses cached probs.
"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
rng = np.random.RandomState(0)

def atta_lcb(sessions, ks=(4, 8, 16, 32)):
    """sessions: list of dict(c_s, c_a) per-trial correctness arrays. Returns table."""
    base_s = np.mean([x['c_s'].mean() for x in sessions]) * 100
    base_a = np.mean([x['c_a'].mean() for x in sessions]) * 100
    oracle = np.mean([max(x['c_s'].mean(), x['c_a'].mean()) for x in sessions]) * 100
    out = {"never": base_s, "always": base_a, "oracle": oracle, "k": {}}
    for k in ks:
        eff = []; harmed = 0; adapt = 0
        for x in sessions:
            diff = x['c_a'] - x['c_s']; n = len(diff)
            idx = rng.permutation(n)[:min(k, n)]; dk = diff[idx]
            se = dk.std(ddof=1) / np.sqrt(len(dk)) if len(dk) > 1 else 1.0
            do = (dk.mean() - 1.645 * se) > 0
            adapt += do
            eff.append((x['c_a'].mean() if do else x['c_s'].mean()) * 100)
            if do and (x['c_a'].mean() - x['c_s'].mean()) * 100 < -1: harmed += 1
        out["k"][k] = dict(effAcc=float(np.mean(eff)), dvs=float(np.mean(eff) - base_s),
                           harmed=harmed, adaptRate=adapt / len(sessions))
    return out

def report(name, out):
    print(f"\n===== {name} =====")
    print(f"  never(source)={out['never']:.2f}  always-adapt={out['always']:.2f}({out['always']-out['never']:+.2f})  "
          f"oracle={out['oracle']:.2f}(+{out['oracle']-out['never']:.2f})")
    print(f"  {'k':>4} {'effAcc':>7} {'Δvssrc':>7} {'harmed':>7} {'adaptRate':>9} {'%oracle-gain':>12}")
    og = out['oracle'] - out['never']
    for k, m in out["k"].items():
        pct = (m['dvs'] / og * 100) if og > 1e-6 else float('nan')
        print(f"  {k:>4} {m['effAcc']:>7.2f} {m['dvs']:>+7.2f} {m['harmed']:>7d} {m['adaptRate']:>9.2f} {pct:>11.0f}%")

# ---- 2a DA-DC (GOOD adapter) ----
exp = np.load(f"{RES}/260602_expert_portfolio_table/expert_portfolio_arrays.npz", allow_pickle=True)
names = [str(x) for x in exp["experts"].tolist()]; IX = {n: names.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
Lab = exp["labels"]; src = exp["probs"][:, IX["source"]]; fe = exp["probs"][:, IX["full_ea"]]; sh = exp["probs"][:, IX["shrink_0.1"]]
R = np.load(f"{RES}/260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
sess2a = []
for s in range(9):
    dadc = 0.3 * src[s] + 0.4 * fe[s] + 0.3 * sh[s] + 0.3 * R[s]
    sess2a.append(dict(c_s=(src[s].argmax(1) == Lab[s]).astype(float),
                       c_a=(dadc.argmax(1) == Lab[s]).astype(float)))
report("2a DA-DC (GOOD adapter, n=9): does ATTA-LCB KEEP the gain?", atta_lcb(sess2a))

# ---- Lee2019 Riemann + blend (BAD adapter) ----
d = np.load(f"{RES}/260609_lee2019_adapter_probs.npz")
subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("Ps_")})
for ADAP in ["riemann", "blend"]:
    sess = []
    for s in subs:
        Ps = d[f"Ps_{s}"].astype(np.float64); Pr = d[f"Pr_{s}"].astype(np.float64); y = d[f"y_{s}"]
        Pa = Pr if ADAP == "riemann" else 0.5 * Ps + 0.5 * Pr
        sess.append(dict(c_s=(Ps.argmax(1) == y).astype(float), c_a=(Pa.argmax(1) == y).astype(float)))
    report(f"Lee2019 {ADAP} (BAD adapter, n={len(subs)}): does ATTA-LCB VETO it (harmed->0)?", atta_lcb(sess))

print("\n" + "="*70)
print("VERDICT: gate is SELECTIVE+safe if 2a captures a high %oracle-gain (keeps DA-DC's +4.4)")
print("AND Lee2019 harmed->0 (vetoes the disaster). If 2a %oracle-gain is also ~0, it's just CONSERVATIVE.")

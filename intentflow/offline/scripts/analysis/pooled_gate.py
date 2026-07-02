"""POOLED gate (borrow strength across sessions) vs per-session LCB. Leak-free:
probe = FIRST k trials (cued calibration), evaluate ONLY on remaining n-k. Compares 4 deciders:
  per-session LCB : Δ̂_i - 1.645*SE_i > 0           (the one that degenerated to never-adapt)
  global          : adapt ALL iff cohort-mean LCB>0 (a t-test on the dataset)
  EB-pooled       : empirical-Bayes shrink Δ̂_i toward population μ; posterior LCB>0
  EB+H            : prior mean depends on label-free overrule-mass H_i (OLS Δ̂~H); posterior LCB>0
Goal: capture gain on the GOOD adapter (2a, fix the +0.00) while keeping harmed≈0 on BAD (Lee2019).
CPU; cached probs.
"""
import numpy as np, warnings, json
warnings.filterwarnings("ignore")
RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"

# ---- build per-session (c_s, c_a, H) in trial order ----
exp = np.load(f"{RES}/260602_expert_portfolio_table/expert_portfolio_arrays.npz", allow_pickle=True)
nm = [str(x) for x in exp["experts"].tolist()]; IX = {n: nm.index(n) for n in ["source", "full_ea", "shrink_0.1"]}
Lab = exp["labels"]; src = exp["probs"][:, IX["source"]]; fe = exp["probs"][:, IX["full_ea"]]; sh = exp["probs"][:, IX["shrink_0.1"]]
R2a = np.load(f"{RES}/260603_diverse_riemann_preds.npz", allow_pickle=True)["probs"]
def H_of(Ps, Pa): return float((Ps.max(1) * (Pa.argmax(1) != Ps.argmax(1))).mean())
def sess_2a():
    S = []
    for s in range(9):
        dd = 0.3*src[s]+0.4*fe[s]+0.3*sh[s]+0.3*R2a[s]
        S.append(dict(c_s=(src[s].argmax(1) == Lab[s]).astype(float), c_a=(dd.argmax(1) == Lab[s]).astype(float), H=H_of(src[s], dd)))
    return S
d = np.load(f"{RES}/260609_lee2019_adapter_probs.npz"); subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("Ps_")})
def sess_lee(which):
    S = []
    for s in subs:
        Ps = d[f"Ps_{s}"].astype(float); Pr = d[f"Pr_{s}"].astype(float); y = d[f"y_{s}"]
        Pa = Pr if which == "riemann" else 0.5*Ps+0.5*Pr
        S.append(dict(c_s=(Ps.argmax(1) == y).astype(float), c_a=(Pa.argmax(1) == y).astype(float), H=H_of(Ps, Pa)))
    return S

def probe_stats(S, k):
    """leak-free probe estimates (fraction) + held-out source/adapter acc per session."""
    dhat = []; se = []; cst = []; cat = []; H = []
    for x in S:
        cs, ca, n = x['c_s'], x['c_a'], len(x['c_s'])
        diff = (ca - cs)[:k]
        dhat.append(diff.mean()); se.append(max(diff.std(ddof=1)/np.sqrt(k) if k > 1 else 0.5, 1e-3))
        cst.append(cs[k:].mean()); cat.append(ca[k:].mean()); H.append(x['H'])
    return map(np.array, (dhat, se, cst, cat, H))

def eb(dhat, se, H=None):
    if H is not None:
        X = np.c_[np.ones_like(H), (H - H.mean())/(H.std()+1e-9)]
        beta = np.linalg.lstsq(X, dhat, rcond=None)[0]; mu = X @ beta; resid = dhat - mu
        tau2 = max(resid.var() - (se**2).mean(), 1e-6)
    else:
        mu = np.full_like(dhat, dhat.mean()); tau2 = max(dhat.var() - (se**2).mean(), 1e-6)
    prec = 1/se**2 + 1/tau2; post = (dhat/se**2 + mu/tau2)/prec; pse = np.sqrt(1/prec)
    return post - 1.645*pse > 0

def evaluate(adapt, cst, cat):
    eff = np.where(adapt, cat, cst); base = cst.mean()
    harmed = int(((cat - cst < -0.01) & adapt).sum())
    return (eff.mean()-base)*100, harmed, adapt.mean(), (cat.mean()-base)*100, (np.maximum(cst, cat).mean()-base)*100

ADS = {"2a DA-DC (良)": sess_2a(), "Lee2019 Riemann (危険)": sess_lee("riemann"), "Lee2019 blend": sess_lee("blend")}
report = {}
for name, S in ADS.items():
    print("="*78); print(f"{name}  (n={len(S)})")
    print(f"  {'k':>3} {'method':>14} {'effΔ':>7} {'%oracle':>8} {'harmed':>7} {'adaptRate':>9}  always/oracle")
    report[name] = {}
    for k in [8, 16, 32]:
        dhat, se, cst, cat, H = probe_stats(S, k)
        lcb = dhat - 1.645*se > 0
        gmean = dhat.mean(); gse = np.sqrt((se**2).mean()/len(dhat))
        glob = np.full(len(dhat), (gmean - 1.645*gse) > 0)
        deciders = {"per-session LCB": lcb, "global": glob, "EB-pooled": eb(dhat, se), "EB+H": eb(dhat, se, H)}
        for mname, adapt in deciders.items():
            effd, harmed, ar, alw, ora = evaluate(adapt, cst, cat)
            pct = effd/ora*100 if ora > 1e-6 else float('nan')
            tag = f"  always={alw:+.2f} oracle={ora:+.2f}" if mname == "per-session LCB" else ""
            print(f"  {k:>3} {mname:>14} {effd:>+7.2f} {pct:>7.0f}% {harmed:>7d} {ar:>9.2f}{tag}")
            report[name].setdefault(str(k), {})[mname] = dict(effΔ=effd, pct_oracle=pct, harmed=harmed, adaptRate=ar, always=alw, oracle=ora)
json.dump(report, open(f"{RES}/260609_pooled_gate.json", "w"), indent=2)
print("="*78)
print("WIN if EB / EB+H: 2a %oracle >> per-session-LCB's ~0%  AND  Lee2019 harmed stays 0.")
print(f"saved {RES}/260609_pooled_gate.json")

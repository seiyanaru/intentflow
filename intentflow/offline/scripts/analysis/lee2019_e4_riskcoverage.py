"""E4 (make-or-break): does the clusterability GATE beat existing label-free reliability
predictors at SELECTIVE PREDICTION (risk-coverage / AURC), on a STRONG decoder?

Loads a per-trial dump (default: 260608_lee2019_pertrial_tcformer.npz from
lee2019_e4_pertrial_dump.py) with per held-out subject:
  F0/F1 (feats), P0/P1 (net softmax), L0/L1 (net logits), y0/y1 (labels), sessions 0(source)/1(target).

Predictors (ALL label-free, per subject, higher = more reliable):
  clusterability (OURS) = silhouette(KMeans k=#classes, z-scored target feats)   [feature space]
  feat_db / feat_ch / feat_nuc = FAIRNESS controls: feature-space separability (Davies-Bouldin/
        Calinski-Harabasz of the same KMeans) + feature-space nuclear-norm. Tests "is silhouette
        special, or does any feature-geometry metric work?" and controls the output-vs-feature confound.
  conf (MSP)           = mean max-prob
  negentropy           = -mean predictive entropy
  nuc_dispersity (Deng ICML23) = nuclear norm of target prob matrix (confidence+diversity)  <- PRIMARY competitor
  mano (Xie NeurIPS24) = mean Lq-norm of normalized logits/probs
  maxlogit / maxlogit_pnorm (Cattelan UAI24) = (p-normalized) max logit          [needs logits]
  atc (Garg ICLR22)    = source-threshold transfer of confidence                 [needs source]
  neg_mahalanobis      = -median min-source-class Mahalanobis of target feats     [needs source]
Riemannian Potato/Field is a RAW-EEG covariance quality gate (different input space) -> separate dump, not here.

PRE-REGISTERED DECISION RULE (frozen 2026-06-08, BEFORE strong-decoder numbers seen):
  PRIMARY endpoint = paired subject cluster-bootstrap of session-AURC DIFFERENCE
        Delta = AURC(nuc_dispersity) - AURC(clusterability), mode=net (deployed decoder).
  GOOD     : clusterability rank-1 session-AURC AND ci_lo(Delta)>0 AND (Delta>=0.010 or g>=0.05)
             AND p(not-better)<0.05 AND partial-Spearman(control source-acc)>=+0.30.
  BAD      : a baseline beats clusterability with the paired CI strictly favoring it (ci_hi<0),
             OR clusterability rank>=3, OR clusterability session-AURC within 0.01 of random.
  MARGINAL : everything else (rank-1/2 but CI straddles 0, or significant-but-negligible, or mode-fragile).
  g = Delta / (random_aurc - oracle_aurc). net is primary; lda is confirmatory; ties NEVER count GOOD.
The verdict is auto-computed below so it cannot be moved post-hoc.

Run: intentflow env. CPU only (no GPU/training).
"""
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import spearmanr, rankdata

RES = "/mnt/data/seiya.narukawa/intentflow/intentflow/offline/results/research_outputs"
EPS = 1e-12
PRIMARY = "nuc_dispersity"  # the named competitor (R1 risk); frozen
NAME_ORDER = ["clusterability", "feat_db", "feat_ch", "feat_nuc", "conf", "negentropy",
              "nuc_dispersity", "mano", "maxlogit", "maxlogit_pnorm", "atc", "neg_mahalanobis"]

def softmax_rows(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z); return e / (e.sum(1, keepdims=True) + EPS)

def entropy_rows(p):
    return -(p * np.log(p + EPS)).sum(1)

def maha_scores(F0, y0, F1):
    cls = np.unique(y0); M = np.stack([F0[y0 == c].mean(0) for c in cls])
    resid = np.concatenate([F0[y0 == c] - F0[y0 == c].mean(0) for c in cls], 0)
    P = np.linalg.pinv(LedoitWolf().fit(resid).covariance_)
    return np.min(np.stack([np.einsum('ij,jk,ik->i', F1 - M[c], P, F1 - M[c]) for c in range(len(cls))], 1), 1)

def partial_spearman(x, y, z):  # corr(x,y | z) on ranks via residualization
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        b1 = np.c_[np.ones_like(b), b]; beta = np.linalg.lstsq(b1, a, rcond=None)[0]; return a - b1 @ beta
    return float(np.corrcoef(resid(rx, rz), resid(ry, rz))[0, 1])

def subject_predictors(d, s, mode):
    """Return (scores dict, acc, per_trial dict, source_acc or None)."""
    F1 = d[f"F1_{s}"].astype(np.float64); y1 = d[f"y1_{s}"]
    P1 = d.get(f"P1_{s}"); L1 = d.get(f"L1_{s}")
    haveF0 = f"F0_{s}" in d
    F0 = d[f"F0_{s}"].astype(np.float64) if haveF0 else None
    y0 = d[f"y0_{s}"] if haveF0 else None
    P0 = d.get(f"P0_{s}"); L0 = d.get(f"L0_{s}")
    k = len(np.unique(y0)) if haveF0 else (P1.shape[1] if P1 is not None else 2)

    if mode == "lda":
        if not haveF0: return None
        lda = LDA(solver="lsqr", shrinkage="auto").fit(F0, y0)
        proba1 = lda.predict_proba(F1); pred1 = lda.predict(F1)
        proba0 = lda.predict_proba(F0); pred0 = lda.predict(F0)
        logit1 = None  # binary LDA decision_function is 1-D -> maxlogit ill-defined; skip
    else:  # net
        if P1 is None: raise ValueError(f"net mode needs P1 for subject {s}")
        proba1 = P1.astype(np.float64); pred1 = proba1.argmax(1)
        proba0 = P0.astype(np.float64) if P0 is not None else None
        pred0 = proba0.argmax(1) if proba0 is not None else None
        logit1 = L1.astype(np.float64) if L1 is not None else None

    correct = (pred1 == y1).astype(float); acc = correct.mean() * 100.0
    sacc = float((pred0 == y0).mean() * 100.0) if (pred0 is not None and y0 is not None) else None
    sc = {}; pt = {"correct": correct}

    # clusterability (OURS) -- feature-space, decoder-agnostic
    Z = (F1 - F1.mean(0)) / (F1.std(0) + 1e-8)
    km = KMeans(k, n_init=5, random_state=0).fit_predict(Z)
    sc["clusterability"] = float(silhouette_score(Z, km))
    # FEATURE-SPACE separability controls (fairness vs output-space predictors)
    if len(np.unique(km)) > 1:
        try: sc["feat_db"] = float(-davies_bouldin_score(Z, km))     # -DB: higher=better
        except Exception: pass
        try: sc["feat_ch"] = float(calinski_harabasz_score(Z, km))   # higher=better
        except Exception: pass
    sc["feat_nuc"] = float(np.linalg.svd(Z, compute_uv=False).sum() / np.sqrt(len(Z) * Z.shape[1]))

    conf_t = proba1.max(1); sc["conf"] = float(conf_t.mean()); pt["conf"] = conf_t
    negent_t = -entropy_rows(proba1); sc["negentropy"] = float(negent_t.mean()); pt["negentropy"] = negent_t

    sv = np.linalg.svd(proba1, compute_uv=False)
    sc["nuc_dispersity"] = float(sv.sum() / np.sqrt(len(F1) * proba1.shape[1]))

    pm = softmax_rows(logit1) if logit1 is not None else proba1
    mano_t = (pm ** 4).sum(1) ** 0.25; sc["mano"] = float(mano_t.mean()); pt["mano"] = mano_t

    if logit1 is not None:
        maxl_t = logit1.max(1); sc["maxlogit"] = float(maxl_t.mean()); pt["maxlogit"] = maxl_t
        ln = logit1 / (np.linalg.norm(logit1, ord=3, axis=1, keepdims=True) + EPS)
        mlpn_t = ln.max(1); sc["maxlogit_pnorm"] = float(mlpn_t.mean()); pt["maxlogit_pnorm"] = mlpn_t

    if haveF0 and proba0 is not None and pred0 is not None:
        s0 = proba0.max(1); src_err = 1.0 - (pred0 == y0).mean()
        t = np.quantile(s0, np.clip(src_err, 0, 1))
        s1 = proba1.max(1); sc["atc"] = float((s1 >= t).mean()); pt["atc"] = (s1 >= t).astype(float)
        mh = maha_scores(F0, y0, F1)
        sc["neg_mahalanobis"] = float(-np.median(mh)); pt["neg_mahalanobis"] = -mh

    return sc, acc, pt, sacc


def aurc_from_order(correct_ordered):
    """Selective risk-coverage AURC (lower=better): mean over coverage of cumulative selective risk."""
    n = len(correct_ordered)
    cum_err = np.cumsum(1.0 - correct_ordered) / np.arange(1, n + 1)
    return float(cum_err.mean())

def selacc_at(correct_ordered, cov):
    n = len(correct_ordered); k = max(1, int(round(cov * n)))
    return float(correct_ordered[:k].mean() * 100.0)

def _session_aurc(rows, idx, nm):
    sub = [rows[i] for i in idx]
    order = np.argsort(-np.array([sub[j][1][nm] for j in range(len(sub))]))
    correct = np.concatenate([sub[j][3]["correct"] for j in order])
    return aurc_from_order(correct)

def paired_bootstrap(rows, names, ref="clusterability", B=5000, seed=0):
    """Paired SUBJECT cluster-bootstrap of Delta=AURC(nm)-AURC(ref) (>0 => ref better=lower risk)."""
    if ref not in names: return {}
    N = len(rows); rng = np.random.RandomState(seed)
    others = [nm for nm in names if nm != ref]
    ref_full = _session_aurc(rows, list(range(N)), ref)
    hat = {nm: _session_aurc(rows, list(range(N)), nm) - ref_full for nm in others}
    boot = {nm: [] for nm in others}
    for _ in range(B):
        idx = rng.randint(0, N, N); aref = _session_aurc(rows, idx, ref)
        for nm in others:
            boot[nm].append(_session_aurc(rows, idx, nm) - aref)
    res = {}
    for nm in others:
        b = np.array(boot[nm]); lo, hi = np.percentile(b, [2.5, 97.5])
        res[nm] = {"delta_hat": float(hat[nm]), "ci_lo": float(lo), "ci_hi": float(hi),
                   "p_ref_not_better": float(np.mean(b <= 0))}  # P(Delta<=0) = clusterability not better than nm
    return res


def evaluate(d, subs, mode):
    rows = []
    for s in subs:
        r = subject_predictors(d, s, mode)
        if r is None: continue
        sc, acc, pt, sacc = r
        rows.append((s, sc, acc, pt, sacc))
    if not rows: return None
    accs = np.array([r[2] for r in rows])
    saccs = [r[4] for r in rows]; have_sacc = all(v is not None for v in saccs)
    sacc_arr = np.array([v for v in saccs]) if have_sacc else None
    names = set(rows[0][1].keys())
    for r in rows: names &= set(r[1].keys())
    names = [n for n in NAME_ORDER if n in names]

    out = {"mode": mode, "n_subjects": len(rows), "mean_acc": float(accs.mean()),
           "acc_range": [float(accs.min()), float(accs.max())], "predictors": {}}
    all_correct = np.concatenate([r[3]["correct"] for r in rows])
    rng = np.random.RandomState(0)
    rand_aurc = float(np.mean([aurc_from_order(all_correct[rng.permutation(len(all_correct))]) for _ in range(200)]))
    oracle_aurc = aurc_from_order(np.sort(all_correct)[::-1])
    out["bounds"] = {"random_aurc": rand_aurc, "oracle_aurc": oracle_aurc, "overall_error": float(1 - all_correct.mean())}

    for nm in names:
        subj_score = np.array([r[1][nm] for r in rows])
        rho = spearmanr(subj_score, accs).correlation
        partial = partial_spearman(subj_score, accs, sacc_arr) if have_sacc else None
        order_subj = np.argsort(-subj_score)
        sess_correct = np.concatenate([rows[i][3]["correct"] for i in order_subj])
        sess_aurc = aurc_from_order(sess_correct)
        sess_sel = {f"selacc@{int(c*100)}": selacc_at(sess_correct, c) for c in (0.9, 0.8, 0.7, 0.5)}
        per_trial_avail = all(nm in r[3] for r in rows)
        if per_trial_avail:
            tscore = np.concatenate([r[3][nm] for r in rows])
        else:
            tscore = np.concatenate([np.full(len(r[3]["correct"]), r[1][nm]) for r in rows])
        tcorrect = np.concatenate([r[3]["correct"] for r in rows])
        trial_aurc = aurc_from_order(tcorrect[np.argsort(-tscore)])
        out["predictors"][nm] = {"spearman": None if rho is None else float(rho),
                                 "partial_spearman": partial, "session_aurc": sess_aurc,
                                 "trial_aurc": trial_aurc, "session_selacc": sess_sel,
                                 "per_trial": per_trial_avail}
    out["bootstrap_vs_clusterability"] = paired_bootstrap(rows, names, ref="clusterability")
    out["verdict"] = compute_verdict(out)
    return out


def compute_verdict(out):
    """Apply the FROZEN pre-registered rule (mode=net is primary; lda is confirmatory)."""
    P = out["predictors"]
    if "clusterability" not in P: return {"label": "N/A", "reason": "clusterability absent"}
    cl = P["clusterability"]
    ranked = sorted(P.items(), key=lambda kv: kv[1]["session_aurc"])  # lowest AURC first
    rank = [n for n, _ in ranked].index("clusterability") + 1
    bs = out.get("bootstrap_vs_clusterability", {})
    prim = bs.get(PRIMARY)
    rnd, ora = out["bounds"]["random_aurc"], out["bounds"]["oracle_aurc"]
    scale = max(rnd - ora, 1e-6)
    near_random = cl["session_aurc"] >= rnd - 0.01
    partial_ok = (cl["partial_spearman"] is None) or (cl["partial_spearman"] >= 0.30)
    reasons = [f"rank={rank}", f"sessAURC={cl['session_aurc']:.3f}", f"random={rnd:.3f}",
               f"partial={cl['partial_spearman']}"]
    # BAD checks
    bad_competitor = any(m["ci_hi"] < 0 for m in bs.values())  # some baseline strictly better (paired)
    if bad_competitor or rank >= 3 or near_random:
        why = ("a baseline strictly better (paired CI ci_hi<0)" if bad_competitor else
               ("rank>=3" if rank >= 3 else "AURC ~ random (no signal)"))
        return {"label": "BAD", "reason": why + " | " + ", ".join(reasons), "clusterability_rank": rank}
    # GOOD checks (need the primary bootstrap)
    if prim is not None:
        g = prim["delta_hat"] / scale
        good = (rank == 1 and prim["ci_lo"] > 0 and (prim["delta_hat"] >= 0.010 or g >= 0.05)
                and prim["p_ref_not_better"] < 0.05 and partial_ok)
        if good:
            return {"label": "GOOD", "reason": f"rank1, vs {PRIMARY}: Delta={prim['delta_hat']:+.3f} "
                    f"CI[{prim['ci_lo']:+.3f},{prim['ci_hi']:+.3f}] g={g:+.3f} p={prim['p_ref_not_better']:.3f}, "
                    + ", ".join(reasons)}
    return {"label": "MARGINAL", "reason": "rank1/2 but not a clean CI-separated, practically-meaningful, "
            "partial-robust win | " + ", ".join(reasons), "clusterability_rank": rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=f"{RES}/260608_lee2019_pertrial_tcformer.npz")
    ap.add_argument("--out", default=f"{RES}/260608_lee2019_e4_riskcoverage.json")
    args = ap.parse_args()
    path = args.npz; dev = False
    if not os.path.exists(path):
        alt = f"{RES}/260607_lee2019_pertrial_eegnet.npz"
        print(f"[WARN] {os.path.basename(path)} not found -> DEV fallback to weak eegnet {os.path.basename(alt)}")
        print("[WARN] eegnet is the WEAK F7-confounded decoder; HARNESS DEBUG ONLY, not a valid E4 verdict.")
        path = alt; dev = True
    d = np.load(path)
    subs = sorted({int(k.split("_")[1]) for k in d.files if k.startswith("F1_")})
    haveF0 = any(k.startswith("F0_") for k in d.files)
    haveP1 = any(k.startswith("P1_") for k in d.files)
    modes = ([ "net"] if haveP1 else []) + (["lda"] if haveF0 else [])
    if not modes:
        raise SystemExit("npz has neither P1 (net mode) nor F0 (lda mode) -> cannot evaluate")
    print(f"loaded {os.path.basename(path)}: {len(subs)} subjects, F0={haveF0} P1={haveP1}, modes={modes}")

    report = {"npz": os.path.basename(path), "dev_weak_decoder": dev, "results": []}
    for mode in modes:
        res = evaluate(d, subs, mode)
        if res is None: continue
        report["results"].append(res)
        b = res["bounds"]
        print(f"\n===== MODE={mode}  n={res['n_subjects']}  mean_acc={res['mean_acc']:.1f}"
              f"  bounds: random={b['random_aurc']:.3f} oracle={b['oracle_aurc']:.3f} =====")
        print(f"{'predictor':>16} {'Spear':>7} {'partial':>7} {'sessAURC':>9} {'trAURC':>7}  {'selacc@80/70/50':>15}")
        for nm, m in sorted(res["predictors"].items(), key=lambda kv: kv[1]["session_aurc"]):
            sp = " n/a " if m["spearman"] is None else f"{m['spearman']:+.3f}"
            pa = " n/a " if m["partial_spearman"] is None else f"{m['partial_spearman']:+.3f}"
            ss = m["session_selacc"]; tag = " <-OURS" if nm == "clusterability" else ""
            print(f"{nm:>16} {sp:>7} {pa:>7} {m['session_aurc']:>9.3f} {m['trial_aurc']:>7.3f}  "
                  f"{ss['selacc@80']:5.1f}/{ss['selacc@70']:4.1f}/{ss['selacc@50']:4.1f}{tag}")
        bs = res.get("bootstrap_vs_clusterability", {})
        if bs:
            print("  paired subject-bootstrap  Delta=AURC(baseline)-AURC(clusterability)  (Delta>0 & ci_lo>0 => clusterability better):")
            for nm in sorted(bs, key=lambda n: -bs[n]["delta_hat"]):
                m = bs[nm]; star = "  *PRIMARY (R1)*" if nm == PRIMARY else ""
                print(f"     vs {nm:>16}: Delta={m['delta_hat']:+.3f} CI[{m['ci_lo']:+.3f},{m['ci_hi']:+.3f}] "
                      f"p(cl not better)={m['p_ref_not_better']:.3f}{star}")
        v = res["verdict"]
        print(f"  >>> VERDICT [{mode}]: {v['label']}  ({v['reason']})")
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nsaved {args.out}")
    if dev:
        print("REMINDER: weak-eegnet DEV run is INVALID for the verdict; rerun on the strong tcformer dump.")
    else:
        net = next((r for r in report["results"] if r["mode"] == "net"), None)
        if net:
            print(f"\nPRIMARY (net) VERDICT = {net['verdict']['label']}. "
                  f"Sanity: clusterability Spearman={net['predictors']['clusterability']['spearman']:+.3f} "
                  f"(E1 was +0.638; if outside [+0.45,+0.80] treat dump as SUSPECT before trusting verdict).")

if __name__ == "__main__":
    main()

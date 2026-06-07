"""Settle the contradiction: does an INVERTED (confident x disagreement) active
query criterion beat STANDARD uncertainty (entropy / margin) at matched label
budget, for unlocking the per-trial routing headroom?

Two verification subagents reported opposite results. This runs a clean,
apples-to-apples comparison: same base, same expert pool, same harm-first update
rule; ONLY the query-selection criterion differs.

Protocol (Option A, cued-calibration style): base = fixed blend
0.3*source + 0.4*full_ea + 0.3*shrink_0.1. For budget B labels/subject, each
criterion ranks trials and queries top-B; on a queried trial, if any pool expert
{source, full_ea, shrink_0.1} predicts the revealed true label, output it (harm-
first: never worse), else keep base. Non-queried trials: base. Score ALL trials.
Also reports AUC of each signal for predicting 'rescuable' (base wrong AND some
pool expert right) — this directly explains which signal finds the headroom.
Caveat (reported, not hidden): Option A 'fixes what it labels', so gains on
queried trials are partly definitional; the criterion comparison is still valid
because budget+update are identical across criteria.
"""

from __future__ import annotations

import numpy as np

NPZ = {
    "seed0": "intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz",
    "seed1": "intentflow/offline/results/research_outputs/260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz",
}
POOL = ["source", "full_ea", "shrink_0.1"]
W = np.array([0.3, 0.4, 0.3])
EPS = 1e-12
BUDGETS = [0, 5, 10, 20, 30, 50]


def load(seed):
    d = np.load(NPZ[seed], allow_pickle=True)
    ex = [str(x) for x in d["experts"].tolist()]
    idx = [ex.index(n) for n in POOL]
    return d["probs"][:, idx], d["labels"], d["subjects"].astype(int)


def auc(score, target):
    # rank-based AUC; higher score should mean target=1
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    pos = target == 1; npos = pos.sum(); nneg = (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def run(seed):
    probs, labels, subs = load(seed)
    # criteria -> per-subject accuracy at each budget
    crits = ["random", "entropy", "margin", "disagree", "inverted"]
    acc = {c: {B: [] for B in BUDGETS} for c in crits}
    src_acc, base_acc = [], []
    aucs = {"entropy": [], "margin": [], "disagree": [], "inverted": []}
    rng = np.random.RandomState(0)
    for si, sid in enumerate(subs):
        y = labels[si]
        pr = probs[si]                      # [3, T, C]
        preds = pr.argmax(-1)               # [3, T]
        base_p = np.tensordot(W, pr, axes=(0, 0))   # [T, C]
        base_pred = base_p.argmax(-1)
        base_conf = base_p.max(-1)
        base_ent = -(base_p * np.log(np.clip(base_p, EPS, 1))).sum(-1)
        top2 = np.sort(base_p, -1)[:, -2:]
        base_margin = top2[:, 1] - top2[:, 0]
        disagree = (preds != base_pred[None, :]).sum(0).astype(float)  # 0..3
        T = len(y)
        base_correct = base_pred == y
        pool_any_correct = (preds == y[None, :]).any(0)
        rescuable = (~base_correct) & pool_any_correct
        # signals (higher => more likely to query)
        sig = {
            "entropy": base_ent,
            "margin": -base_margin,            # low margin => query
            "disagree": disagree + 1e-3 * rng.rand(T),
            "inverted": base_conf * disagree,  # confident AND disagreeing
        }
        for k in aucs:
            aucs[k].append(auc(sig[k], rescuable.astype(int)))
        src_acc.append((preds[0] == y).mean() * 100)
        base_acc.append(base_correct.mean() * 100)
        # oracle-expert prediction per trial (for the harm-first update on queried)
        def apply_query(qidx):
            out = base_pred.copy()
            for t in qidx:
                hit = np.where(preds[:, t] == y[t])[0]
                if len(hit):
                    out[t] = y[t]            # some expert correct -> output it
                # else keep base
            return (out == y).mean() * 100
        for B in BUDGETS:
            if B == 0:
                for c in crits:
                    acc[c][B].append(base_correct.mean() * 100)
                continue
            for c in crits:
                if c == "random":
                    q = rng.permutation(T)[:B]
                else:
                    q = np.argsort(-sig[c])[:B]
                acc[c][B].append(apply_query(q))
    return acc, np.mean(src_acc), np.mean(base_acc), {k: np.nanmean(v) for k, v in aucs.items()}, len(subs)


def main():
    for seed in ("seed0", "seed1"):
        acc, src, base, aucs, n = run(seed)
        print(f"\n========== {seed} (n={n}) ==========")
        print(f"source={src:.2f}  base-blend(0.3/0.4/0.3)={base:.2f} (+{base-src:.2f} vs source)")
        print(f"AUC(signal -> rescuable):  " + "  ".join(f"{k}={v:.3f}" for k, v in aucs.items()))
        print(f"{'B':>4} " + " ".join(f"{c:>9}" for c in ["random","entropy","margin","disagree","inverted"]))
        for B in BUDGETS:
            row = " ".join(f"{np.mean(acc[c][B]):9.2f}" for c in ["random","entropy","margin","disagree","inverted"])
            print(f"{B:>4} {row}")
        print("  (gain over base at each B = value - base; the winner is the highest column per row)")
        # explicit verdict at B=30
        B = 30
        m = {c: np.mean(acc[c][B]) for c in ["random","entropy","margin","disagree","inverted"]}
        best = max(m, key=m.get)
        print(f"  B=30 winner: {best} ({m[best]:.2f}). inverted={m['inverted']:.2f} vs margin={m['margin']:.2f} vs entropy={m['entropy']:.2f}")


if __name__ == "__main__":
    main()

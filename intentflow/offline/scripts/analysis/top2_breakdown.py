"""Break down the BCIC2a top2 ceiling: what are the "true class is 2nd" trials?

oracle_ceiling.py showed a +11.84pp top2 ceiling on 2a (the true class sits in
the source top-2 but is not ranked 1st). This script characterises those
"rescuable" trials to decide whether a feature-fixed re-ranking can recover them:

- rescuable rate: of all misclassified trials, what fraction have the true class
  as the 2nd-ranked (i.e. reachable by re-ranking without touching features).
- margin = p(pred) - p(true) on rescuable trials. Small margin => near-tie =>
  easy for re-ranking; large margin => the model is confidently wrong.
- confusion pairs (pred -> true) on rescuable trials: systematic pairs suggest a
  structured fix (alignment / pairwise re-ranking) is plausible.

Read-only on source-only logits already on disk; no GPU.
"""

from __future__ import annotations

import csv
import glob
from collections import Counter
from pathlib import Path

import numpy as np

R_2A = "intentflow/offline/results/c_aug_true_9subj_20260506_004923"
OUT = Path("docs/research_progress/260526_oracle_ceiling/top2_breakdown_2a.csv")
CLASSES = ["left_hand", "right_hand", "feet", "tongue"]


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def labels_for(s):
    hits = sorted(glob.glob(f"{R_2A}/eval/s{s}/*/*.npz"))
    return np.load(hits[0], allow_pickle=True)["label"].astype(int)


def main():
    rows = []
    all_resc_margins = []
    all_wrong_margins = []
    pair_counter = Counter()
    tot_n = tot_wrong = tot_resc = 0

    for s in range(1, 10):
        lg = np.load(f"{R_2A}/sources/s{s}/logits_s{s}_tcformer_policy_safe_otta.npy")
        lab = labels_for(s)
        n = min(len(lg), len(lab))
        lg, lab = lg[:n], lab[:n]
        probs = softmax(lg)
        pred = probs.argmax(1)
        order = np.argsort(-probs, axis=1)
        top2 = order[:, 1]

        wrong = pred != lab
        rescuable = wrong & (top2 == lab)  # true class is exactly 2nd
        p_pred = probs[np.arange(n), pred]
        p_true = probs[np.arange(n), lab]
        margin = p_pred - p_true  # how far ahead the wrong top1 is over the true

        resc_margins = margin[rescuable]
        all_resc_margins.extend(resc_margins.tolist())
        all_wrong_margins.extend(margin[wrong].tolist())
        for i in np.where(rescuable)[0]:
            pair_counter[(int(pred[i]), int(lab[i]))] += 1

        tot_n += n
        tot_wrong += int(wrong.sum())
        tot_resc += int(rescuable.sum())
        rows.append(dict(
            subject=s, n=n, acc=100 * (~wrong).mean(),
            n_wrong=int(wrong.sum()), n_rescuable=int(rescuable.sum()),
            rescuable_of_wrong=100 * rescuable.sum() / max(wrong.sum(), 1),
            resc_margin_median=float(np.median(resc_margins)) if len(resc_margins) else float("nan"),
        ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print(f"{'subj':>4} {'acc':>6} {'wrong':>6} {'resc':>5} {'resc/wrong%':>11} {'margin_med':>10}")
    for r in rows:
        print(f"{r['subject']:>4} {r['acc']:>6.2f} {r['n_wrong']:>6} {r['n_rescuable']:>5} "
              f"{r['rescuable_of_wrong']:>11.1f} {r['resc_margin_median']:>10.3f}")
    print("-" * 50)
    am = np.array(all_resc_margins)
    wm = np.array(all_wrong_margins)
    print(f"TOTAL trials={tot_n}  wrong={tot_wrong}  rescuable(true=2nd)={tot_resc} "
          f"({100*tot_resc/tot_n:.2f}% of all, {100*tot_resc/tot_wrong:.1f}% of wrong)")
    print(f"\nRescuable-trial margin p(pred)-p(true) [small=near-tie, easy to re-rank]:")
    print(f"  q25={np.quantile(am,.25):.3f}  median={np.median(am):.3f}  q75={np.quantile(am,.75):.3f}  max={am.max():.3f}")
    print(f"  fraction with margin<0.10 (near-tie): {100*(am<0.10).mean():.1f}%")
    print(f"  fraction with margin<0.20          : {100*(am<0.20).mean():.1f}%")
    print(f"\nTop confusion pairs among rescuable (pred -> true):")
    for (p, t), c in pair_counter.most_common(6):
        print(f"  {CLASSES[p]:>10} -> {CLASSES[t]:<10}  {c}")
    print(f"\n[top2] wrote {OUT}")


if __name__ == "__main__":
    main()

"""Calibration analysis of BCIC2a source-only logits.

Goal: decide whether the large top2 margins (top2_breakdown.py: median 0.334,
only 18.6% near-tie) are *genuine confident errors* or *miscalibration*
(over-confident logits). This drives the re-ranking vs EA/feature-adaptation call.

Metrics (per subject + overall):
- accuracy, ECE(15-bin), Brier (multiclass), NLL.
- best temperature T* (grid, minimises NLL) and ECE/NLL after T*.
- mean confidence on wrong-but-rescuable trials (true class is 2nd).
- sanity: temperature scaling does NOT change argmax => accuracy unchanged
  (calibration cannot raise accuracy on its own).

Read-only on source-only logits; no GPU.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

R_2A = "intentflow/offline/results/c_aug_true_9subj_20260506_004923"
OUT = Path("docs/research_progress/260526_oracle_ceiling/calibration_2a.csv")


def softmax_T(z, T=1.0):
    z = (z / T)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    n = len(conf)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return e


def nll(probs, lab):
    return float(-np.log(probs[np.arange(len(lab)), lab] + 1e-12).mean())


def brier(probs, lab):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(lab)), lab] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def best_temp(logits, lab):
    grid = np.linspace(0.5, 5.0, 91)
    best_T, best = 1.0, 1e9
    for T in grid:
        v = nll(softmax_T(logits, T), lab)
        if v < best:
            best, best_T = v, T
    return best_T


def labels_for(s):
    hits = sorted(glob.glob(f"{R_2A}/eval/s{s}/*/*.npz"))
    return np.load(hits[0], allow_pickle=True)["label"].astype(int)


def main():
    rows = []
    for s in range(1, 10):
        lg = np.load(f"{R_2A}/sources/s{s}/logits_s{s}_tcformer_policy_safe_otta.npy")
        lab = labels_for(s)
        n = min(len(lg), len(lab))
        lg, lab = lg[:n], lab[:n]

        p1 = softmax_T(lg, 1.0)
        pred = p1.argmax(1)
        correct = (pred == lab).astype(float)
        conf = p1.max(1)

        T = best_temp(lg, lab)
        pT = softmax_T(lg, T)
        predT = pT.argmax(1)
        acc_unchanged = bool((pred == predT).all())

        # wrong-but-rescuable (true class is 2nd) confidence at T=1
        order = np.argsort(-p1, axis=1)
        rescuable = (pred != lab) & (order[:, 1] == lab)
        resc_conf = conf[rescuable]

        rows.append(dict(
            subject=s, n=n, acc=100 * correct.mean(),
            ECE=ece(conf, correct), Brier=brier(p1, lab), NLL=nll(p1, lab),
            T_star=T, ECE_T=ece(pT.max(1), (predT == lab).astype(float)),
            NLL_T=nll(pT, lab), acc_unchanged_by_T=acc_unchanged,
            resc_conf_mean=float(resc_conf.mean()) if len(resc_conf) else float("nan"),
        ))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"{'subj':>4} {'acc':>6} {'ECE':>6} {'Brier':>6} {'NLL':>6} {'T*':>5} "
          f"{'ECE@T':>6} {'NLL@T':>6} {'accSame':>7} {'rescConf':>8}")
    for r in rows:
        print(f"{r['subject']:>4} {r['acc']:>6.2f} {r['ECE']:>6.3f} {r['Brier']:>6.3f} "
              f"{r['NLL']:>6.3f} {r['T_star']:>5.2f} {r['ECE_T']:>6.3f} {r['NLL_T']:>6.3f} "
              f"{str(r['acc_unchanged_by_T']):>7} {r['resc_conf_mean']:>8.3f}")
    print("-" * 78)
    print(f"{'MEAN':>4} {m('acc'):>6.2f} {m('ECE'):>6.3f} {m('Brier'):>6.3f} {m('NLL'):>6.3f} "
          f"{m('T_star'):>5.2f} {m('ECE_T'):>6.3f} {m('NLL_T'):>6.3f} {'':>7} {m('resc_conf_mean'):>8.3f}")
    print(f"\nInterpretation hooks:")
    print(f"  - T* > 1 => over-confident (logits too sharp). ECE drop after T* = how much was miscalibration.")
    print(f"  - resc_conf_mean = avg confidence on confident-but-wrong (true=2nd) trials.")
    print(f"  - acc_unchanged_by_T must be all True (temperature cannot change accuracy).")
    print(f"\n[calib] wrote {OUT}")


if __name__ == "__main__":
    main()

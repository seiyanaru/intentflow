"""Oracle ceiling for posterior/prior correction across datasets (2a/2b/HGD).

Question: how much head-room does a *prediction-correction* OTTA (DC-Replay L1 /
CMC) have in principle? We bound it with source-only logits + true labels:

- source_acc      : argmax of frozen source logits (no adaptation).
- top2_oracle     : fraction where the true class is in the source top-2. Absolute
                    ceiling of ANY re-ranking that does not change the features.
                    (Meaningless for 2-class data: always 100%.)
- prior_shift_oracle : best accuracy reachable by adding a per-class constant bias
                    b, chosen in hindsight (coordinate ascent). Ceiling of
                    L1-style prior/logit-bias correction.
- uniform_prior_acc : argmax(softmax * pi_test / pi_train). For class-balanced
                    test sets ~= source_acc (global prior correction is useless).

All inputs are read-only npy/npz already on disk; no GPU, no re-run. The hindsight
oracles are UPPER BOUNDS; online unsupervised correction reaches strictly less.
"""

from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np

OUT_DIR = Path("intentflow/offline/results/research_outputs/260526_oracle_ceiling")

R_2A = "intentflow/offline/results/c_aug_true_9subj_20260506_004923"
R_2B = "intentflow/offline/results/phaseC_2b_firstpass_20260422_164615_seed0"
R_HGD = "intentflow/offline/results/phaseC_hgd_firstpass_20260422_210714_seed0"

DATASETS = {
    "2a": dict(
        logits=R_2A + "/sources/s{s}/logits_s{s}_tcformer_policy_safe_otta.npy",
        label_glob=R_2A + "/eval/s{s}/*/*.npz",
        subjects=list(range(1, 10)),
    ),
    "2b": dict(
        logits=R_2B + "/source_only/logits_s{s}_tcformer_otta.npy",
        label_glob=R_2B + "/hybrid_mom001/otta_stats_s{s}_tcformer_otta.npz",
        subjects=list(range(1, 10)),
    ),
    "hgd": dict(
        logits=R_HGD + "/source_only/logits_s{s}_tcformer_otta.npy",
        label_glob=R_HGD + "/hybrid_mom001/otta_stats_s{s}_tcformer_otta.npz",
        subjects=list(range(1, 15)),
    ),
}


def load_labels(pattern: str) -> np.ndarray | None:
    hits = sorted(glob.glob(pattern))
    if not hits:
        return None
    d = np.load(hits[0], allow_pickle=True)
    return d["label"].astype(int)


def top2_ceiling(logits, labels):
    true_logit = logits[np.arange(len(labels)), labels]
    rank = (logits > true_logit[:, None]).sum(axis=1)
    return float((rank <= 1).mean())


def prior_shift_oracle(logits, labels):
    n_classes = logits.shape[1]
    grid = np.linspace(-6.0, 6.0, 121)
    b = np.zeros(n_classes)
    acc = lambda bias: float((np.argmax(logits + bias, axis=1) == labels).mean())
    cur = acc(b)
    for _ in range(50):
        improved = False
        for c in range(n_classes):
            best_v, best_a = b[c], cur
            for v in grid:
                bb = b.copy(); bb[c] = v
                a = acc(bb)
                if a > best_a:
                    best_a, best_v = a, v
            if best_v != b[c]:
                b[c], cur = best_v, best_a
                improved = True
        if not improved:
            break
    return cur


def uniform_prior_acc(logits, labels):
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    n_classes = logits.shape[1]
    pi_train = np.full(n_classes, 1.0 / n_classes)
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    pi_test = counts / counts.sum()
    adj = probs * (pi_test / pi_train)[None, :]
    return float((np.argmax(adj, axis=1) == labels).mean())


def run_dataset(name, cfg):
    rows = []
    for s in cfg["subjects"]:
        lp = cfg["logits"].format(s=s)
        if not Path(lp).exists():
            continue
        lab = load_labels(cfg["label_glob"].format(s=s))
        if lab is None:
            continue
        lg = np.load(lp)
        n = min(len(lg), len(lab))
        lg, lab = lg[:n], lab[:n]
        nc = lg.shape[1]
        src = (np.argmax(lg, 1) == lab).mean() * 100
        top2 = top2_ceiling(lg, lab) * 100
        ps = prior_shift_oracle(lg, lab) * 100
        unif = uniform_prior_acc(lg, lab) * 100
        rows.append(dict(dataset=name, subject=s, n=n, n_classes=nc,
                         source_acc=src, top2_oracle=top2, prior_shift_oracle=ps,
                         uniform_prior_acc=unif, d_top2=top2 - src,
                         d_prior_shift=ps - src, d_uniform=unif - src))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = []
    for name, cfg in DATASETS.items():
        rows = run_dataset(name, cfg)
        if not rows:
            print(f"[{name}] no data found, skipped")
            continue
        all_rows += rows
        nc = rows[0]["n_classes"]
        m = lambda k: float(np.mean([r[k] for r in rows]))
        top2_note = " (N/A, 2-class)" if nc == 2 else ""
        print(f"\n=== {name}  ({len(rows)} subjects, {nc}-class) ===")
        print(f"{'subj':>4} {'src':>7} {'top2':>7} {'priorS':>7} {'unif':>7} "
              f"{'Δtop2':>7} {'Δprior':>7} {'Δunif':>7}")
        for r in rows:
            print(f"{r['subject']:>4} {r['source_acc']:>7.2f} {r['top2_oracle']:>7.2f} "
                  f"{r['prior_shift_oracle']:>7.2f} {r['uniform_prior_acc']:>7.2f} "
                  f"{r['d_top2']:>7.2f} {r['d_prior_shift']:>7.2f} {r['d_uniform']:>7.2f}")
        print("-" * 64)
        print(f"{'MEAN':>4} {m('source_acc'):>7.2f} {m('top2_oracle'):>7.2f} "
              f"{m('prior_shift_oracle'):>7.2f} {m('uniform_prior_acc'):>7.2f} "
              f"{m('d_top2'):>7.2f} {m('d_prior_shift'):>7.2f} {m('d_uniform'):>7.2f}")
        summary.append((name, nc, len(rows), m('source_acc'), m('d_top2'),
                        m('d_prior_shift'), m('d_uniform'), top2_note))

    keys = list(all_rows[0].keys())
    with open(OUT_DIR / "oracle_ceiling_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    print("\n========== CROSS-DATASET CEILING (mean Δ vs source) ==========")
    print(f"{'ds':>5} {'cls':>4} {'subj':>5} {'src':>7} {'Δtop2':>8} {'Δprior':>8} {'Δunif':>7}")
    for name, nc, k, src, dt, dp, du, note in summary:
        print(f"{name:>5} {nc:>4} {k:>5} {src:>7.2f} {dt:>8.2f} {dp:>8.2f} {du:>7.2f}{note}")
    print(f"\n[oracle] wrote {OUT_DIR/'oracle_ceiling_all.csv'}")


if __name__ == "__main__":
    main()

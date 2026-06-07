"""Frozen-threshold multi-seed validation of the adaptive EA selector.

Test 1 (go/no-go for the harm-first veto-to-source contribution):
take the seed0-tuned selector thresholds, FREEZE them, and apply the exact same
rule to held-out seed1 from the prebuilt expert_portfolio_arrays.npz. No label
tuning on seed1. Reproduces seed0 first as a drift check, then reports seed1.

The selection logic is imported from eval_adaptive_ea_selector.py so it cannot
drift; only the disk-loading fusion branch is reimplemented in-memory because
here candidates come from npz probs, not per-file logits.

Covariance diagnostics are a property of the test EEG session (not the model
seed); build_selector_feature_table.py uses the same json for both seeds, so we
do too.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_adaptive_ea_selector import (  # noqa: E402
    candidate_from_probs,
    reject_by_covariance,
    reject_candidate,
)

FROZEN = dict(
    max_delta_kl=0.05,
    max_dominance=0.42,
    max_entropy_increase=0.11,
    min_conf_drop=0.05,
    max_shrink_condition=30000.0,
    prefer_shrink_diag_cv=0.15,
)

CAND_NAMES = ["source", "full_ea", "shrink_0.1", "partial_0.5_shrink_0.1"]


def select_in_memory(candidates, probs_by_name, cov_row, args, labels):
    """Faithful copy of eval_adaptive_ea_selector.select_candidate, with the
    full+shrink fusion computed from in-memory probs instead of disk."""
    source = candidates["source"]
    for name, cand in candidates.items():
        if name != "source":
            reject_candidate(cand, source, args)
    reject_by_covariance(candidates, cov_row, args)

    shrink = candidates.get("shrink_0.1")
    if (
        cov_row is not None
        and cov_row["test"]["diag_cv"] > args.prefer_shrink_diag_cv
        and shrink is not None
        and not shrink.rejected
    ):
        return shrink

    full = candidates.get("full_ea")
    if (
        args.fuse_full_shrink
        and full is not None
        and shrink is not None
        and not full.rejected
        and not shrink.rejected
    ):
        fused_probs = 0.5 * (probs_by_name["full_ea"] + probs_by_name["shrink_0.1"])
        fused = candidate_from_probs("full_shrink_mean", "mem", fused_probs, labels)
        reject_candidate(fused, source, args)
        if not fused.rejected:
            candidates[fused.name] = fused
            return fused

    if full is not None and not full.rejected:
        return full

    for fallback_name in ("shrink_0.1", "partial_0.5_shrink_0.1"):
        cand = candidates.get(fallback_name)
        if cand is not None and not cand.rejected:
            return cand

    return source


def eval_seed(npz_path, cov_by_subject, fuse):
    data = np.load(npz_path, allow_pickle=True)
    experts = [str(x) for x in data["experts"].tolist()]
    eidx = {name: experts.index(name) for name in CAND_NAMES if name in experts}
    subjects = data["subjects"].astype(int)
    args = SimpleNamespace(**FROZEN, fuse_full_shrink=fuse)

    rows = []
    for sidx, sid in enumerate(subjects):
        labels = data["labels"][sidx]
        probs_by_name = {n: data["probs"][sidx, eidx[n]] for n in eidx}
        candidates = {}
        for n in CAND_NAMES:
            if n in probs_by_name:
                candidates[n] = candidate_from_probs(n, f"mem:{n}", probs_by_name[n], labels)
        cov_row = cov_by_subject.get(int(sid))
        selected = select_in_memory(candidates, probs_by_name, cov_row, args, labels)
        src = candidates["source"].acc
        full = candidates["full_ea"].acc
        shrink = candidates["shrink_0.1"].acc
        oracle = max(src, full, shrink)  # ceiling over the 3-candidate set
        rows.append(
            dict(
                subject=int(sid),
                selected=selected.name,
                selected_acc=selected.acc,
                source_acc=src,
                full_ea_acc=full,
                shrink_acc=shrink,
                oracle3_acc=oracle,
                delta_vs_source=selected.acc - src,
                delta_vs_full=selected.acc - full,
            )
        )
    return rows


def summarize(rows, tag):
    m = lambda k: float(np.mean([r[k] for r in rows]))
    harmed = [r for r in rows if r["delta_vs_source"] < -1e-9]
    worst = min(r["delta_vs_source"] for r in rows)
    print(f"\n===== {tag} =====")
    print(f"{'S':>2} {'sel':>16} {'acc':>6} {'src':>6} {'full':>6} {'shr':>6} {'orac':>6} {'dSrc':>6}")
    for r in rows:
        print(
            f"{r['subject']:>2} {r['selected']:>16} {r['selected_acc']:6.2f} "
            f"{r['source_acc']:6.2f} {r['full_ea_acc']:6.2f} {r['shrink_acc']:6.2f} "
            f"{r['oracle3_acc']:6.2f} {r['delta_vs_source']:+6.2f}"
        )
    print(
        f"means: src={m('source_acc'):.2f} full={m('full_ea_acc'):.2f} "
        f"shrink={m('shrink_acc'):.2f} oracle3={m('oracle3_acc'):.2f} "
        f"SELECTED={m('selected_acc'):.2f}"
    )
    print(
        f"delta selected-source: {m('selected_acc')-m('source_acc'):+.2f}pp | "
        f"delta selected-full: {m('selected_acc')-m('full_ea_acc'):+.2f}pp"
    )
    print(
        f"HARMED subjects (selected<source): {len(harmed)}/{len(rows)} "
        f"{[ (r['subject'], round(r['delta_vs_source'],2)) for r in harmed]} | worst drop {worst:+.2f}pp"
    )
    cap = (m("selected_acc") - m("source_acc")) / max(1e-9, m("oracle3_acc") - m("source_acc"))
    print(f"headroom capture vs 3-cand oracle: {100*cap:.1f}%")
    return dict(
        tag=tag,
        n=len(rows),
        source_mean=m("source_acc"),
        full_mean=m("full_ea_acc"),
        shrink_mean=m("shrink_acc"),
        oracle3_mean=m("oracle3_acc"),
        selected_mean=m("selected_acc"),
        delta_selected_source=m("selected_acc") - m("source_acc"),
        harmed_count=len(harmed),
        worst_drop=worst,
        rows=rows,
    )


def veto_roc(all_rows):
    """GPT Q3: is the rule a high-precision veto against harmful full-EA?
    Harmful := full_ea worse than source by >2pp. Rejected := selector did NOT
    pick full_ea (it vetoed full to source/shrink/fusion)."""
    tp = fp = tn = fn = 0
    for r in all_rows:
        harmful = (r["full_ea_acc"] - r["source_acc"]) < -2.0
        vetoed = r["selected"] != "full_ea"
        if harmful and vetoed:
            tp += 1
        elif (not harmful) and vetoed:
            fp += 1
        elif (not harmful) and (not vetoed):
            tn += 1
        else:
            fn += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    print("\n===== veto-ROC (full-EA harmful by >2pp vs vetoed) =====")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} | precision={prec:.2f} recall={rec:.2f}")
    print("  (FP = vetoed a full-EA that was actually fine; that is the cost of the veto)")
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed0_npz", default="intentflow/offline/results/research_outputs/260602_expert_portfolio_table/expert_portfolio_arrays.npz")
    ap.add_argument("--seed1_npz", default="intentflow/offline/results/research_outputs/260602_expert_portfolio_table_seed1/expert_portfolio_arrays.npz")
    ap.add_argument("--cov", default="intentflow/offline/results/research_outputs/260601_ea_cov_diagnostics.json")
    ap.add_argument("--output", default="intentflow/offline/results/research_outputs/260602_selector_multiseed_frozen.json")
    args = ap.parse_args()

    cov_obj = json.load(open(args.cov))
    cov_by_subject = {int(r["subject"]): r for r in cov_obj["rows"]}

    out = {"frozen_thresholds": FROZEN, "results": {}}
    for fuse, label in [(False, "v1_no_fusion"), (True, "v2_fusion")]:
        s0 = summarize(eval_seed(args.seed0_npz, cov_by_subject, fuse), f"seed0 {label} (DRIFT CHECK)")
        s1 = summarize(eval_seed(args.seed1_npz, cov_by_subject, fuse), f"seed1 {label} (HELD-OUT TEST)")
        out["results"][label] = {"seed0": s0, "seed1": s1}

    # veto-ROC over both seeds, v2 selection
    all_rows = (
        out["results"]["v2_fusion"]["seed0"]["rows"]
        + out["results"]["v2_fusion"]["seed1"]["rows"]
    )
    out["veto_roc_v2_both_seeds"] = veto_roc(all_rows)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()

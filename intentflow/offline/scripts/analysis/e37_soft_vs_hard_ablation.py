"""E37-A mechanism ablation: soft weighting vs hard top-k vs full models.

This analysis tests whether the E36 gain is plausibly due to soft reliability
weighting plus a source anchor, rather than just dimensionality reduction or
another broad selector.

Inputs are existing E32 fixed-method records and E36 frozen guarded records.
No new model fitting or tuning is performed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e37_soft_vs_hard_ablation"


E32_DEFAULTS = [
    RESULTS_DIR
    / "260701_e32_lee2019_sensorimotor20_weighted_ridge"
    / "weighted_ridge_transfer_records.csv",
    RESULTS_DIR
    / "260701_e32_bnci2014_001_full_source_weighted_ridge"
    / "weighted_ridge_transfer_records.csv",
    RESULTS_DIR
    / "260701_e32_bnci2014_001_m8_weighted_ridge"
    / "weighted_ridge_transfer_records.csv",
]
E36_DEFAULT = (
    RESULTS_DIR
    / "260701_e36_frozen_guarded_reliability_audit"
    / "e36_frozen_combined_outer_records.csv"
)


FAMILY_BY_METHOD = {
    "lda_full": "source_anchor",
    "ridge_full_a100": "full_ridge",
    "lda_source_q25": "hard_topk_lda",
    "lda_long_q10": "hard_topk_lda",
    "lda_long_q70": "hard_topk_lda",
    "ridge_hard_source_q25_a100": "hard_topk_ridge",
    "ridge_hard_long_q10_a100": "hard_topk_ridge",
    "ridge_hard_long_q70_a100": "hard_topk_ridge",
    "ridge_source_rank_g1_a100": "soft_weighted_ridge",
    "ridge_source_rank_g2_a100": "soft_weighted_ridge",
    "ridge_longitudinal_rank_g2_a100": "soft_weighted_ridge",
    "e36_frozen_guarded_repeat": "frozen_guarded",
    "e36_frozen_guarded_stable": "frozen_guarded",
}


KEY_METHODS = [
    "lda_full",
    "ridge_full_a100",
    "lda_long_q70",
    "ridge_hard_source_q25_a100",
    "ridge_hard_long_q10_a100",
    "ridge_source_rank_g2_a100",
    "ridge_longitudinal_rank_g2_a100",
    "e36_frozen_guarded_repeat",
    "e36_frozen_guarded_stable",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e32-records", nargs="*", type=Path, default=E32_DEFAULTS)
    parser.add_argument("--e36-records", type=Path, default=E36_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def bootstrap_ci(values: Iterable[float], rng: np.random.Generator, repeats: int) -> list[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(arr, size=(int(repeats), arr.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def load_records(e32_paths: Sequence[Path], e36_path: Path) -> pd.DataFrame:
    fixed_frames = []
    for path in e32_paths:
        df = pd.read_csv(path)
        df["record_source"] = "e32_fixed"
        fixed_frames.append(df)
    fixed_base = pd.concat(fixed_frames, ignore_index=True, sort=False)
    frames = []
    for audit_mode in ["fixed", "repeat", "stable"]:
        copy = fixed_base.copy()
        copy["audit_mode"] = audit_mode
        copy["record_source"] = f"e32_reference_for_{audit_mode}"
        frames.append(copy)
    e36 = pd.read_csv(e36_path)
    e36 = e36[e36["method"].isin(["e36_frozen_guarded_repeat", "e36_frozen_guarded_stable"])].copy()
    e36["source_per_class"] = np.nan
    e36["source_total"] = np.nan
    e36["p"] = np.nan
    e36["p_over_source_total"] = np.nan
    e36["record_source"] = "e36_frozen_guarded"
    frames.append(e36)
    records = pd.concat(frames, ignore_index=True, sort=False)
    records["family"] = records["method"].map(FAMILY_BY_METHOD).fillna("other")
    return records


def subject_method_matrix(records: pd.DataFrame, *, regime: str, audit_mode: str) -> pd.DataFrame:
    sub = records[(records["regime_label"] == regime) & (records["audit_mode"] == audit_mode)].copy()
    if len(sub) == 0:
        return pd.DataFrame()
    by_subject = sub.groupby(["subject", "method"], as_index=False)["accuracy"].mean()
    return by_subject.pivot(index="subject", columns="method", values="accuracy")


def summarize_methods(records: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (regime, audit_mode, method), sub in records.groupby(
        ["regime_label", "audit_mode", "method"], sort=True
    ):
        if method not in FAMILY_BY_METHOD:
            continue
        matrix = subject_method_matrix(records, regime=regime, audit_mode=audit_mode)
        if method not in matrix.columns or "lda_full" not in matrix.columns:
            continue
        acc = matrix[method].dropna().to_numpy(dtype=np.float64)
        gain = (matrix[method] - matrix["lda_full"]).dropna().to_numpy(dtype=np.float64)
        n_selected = np.nan
        if "n_selected" in sub.columns and len(sub["n_selected"].dropna()) > 0:
            n_selected = float(sub["n_selected"].dropna().mean())
        rows.append(
            {
                "regime_label": regime,
                "audit_mode": audit_mode,
                "method": method,
                "family": FAMILY_BY_METHOD[method],
                "n_subjects": int(len(acc)),
                "n_repeats_total": int(sub[["subject", "repeat"]].drop_duplicates().shape[0]),
                "n_selected_mean": n_selected,
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_subject_bootstrap_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(np.mean(gain)),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < -5.0)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["regime_label", "audit_mode", "accuracy_mean"], ascending=[True, True, False]
    )


def best_by_family(method_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, audit_mode, family), sub in method_summary.groupby(
        ["regime_label", "audit_mode", "family"], sort=True
    ):
        if family == "frozen_guarded":
            # Keep the actual guarded methods separate, not as a target-picked family.
            continue
        best = sub.sort_values("accuracy_mean", ascending=False).iloc[0].to_dict()
        best["family_best_note"] = "descriptive fixed-family best; not an honest selector"
        rows.append(best)
    return pd.DataFrame(rows).sort_values(
        ["regime_label", "audit_mode", "accuracy_mean"], ascending=[True, True, False]
    )


def method_diff(
    matrix: pd.DataFrame,
    method_a: str,
    method_b: str,
    *,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict[str, object] | None:
    if method_a not in matrix.columns or method_b not in matrix.columns:
        return None
    diff = (matrix[method_a] - matrix[method_b]).dropna().to_numpy(dtype=np.float64)
    if diff.size == 0:
        return None
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_diff_pp": float(np.mean(diff)),
        "subject_bootstrap_95ci": bootstrap_ci(diff, rng, bootstrap),
        "q05_diff_pp": float(np.quantile(diff, 0.05)),
        "p_diff_lt_minus5": float(np.mean(diff < -5.0)),
        "p_diff_gt_0": float(np.mean(diff > 0.0)),
    }


def key_contrasts(records: pd.DataFrame, method_summary: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for (regime, audit_mode), _ in records.groupby(["regime_label", "audit_mode"], sort=True):
        matrix = subject_method_matrix(records, regime=regime, audit_mode=audit_mode)
        if matrix.empty or "lda_full" not in matrix.columns:
            continue

        regime_summary = method_summary[
            (method_summary["regime_label"] == regime)
            & (method_summary["audit_mode"] == audit_mode)
        ]

        def best_method(family: str) -> str | None:
            sub = regime_summary[regime_summary["family"] == family]
            if len(sub) == 0:
                return None
            return str(sub.sort_values("accuracy_mean", ascending=False).iloc[0]["method"])

        best_soft = best_method("soft_weighted_ridge")
        best_hard_lda = best_method("hard_topk_lda")
        best_hard_ridge = best_method("hard_topk_ridge")
        hard_candidates = [m for m in [best_hard_lda, best_hard_ridge] if m is not None]
        best_hard = None
        if hard_candidates:
            best_hard = (
                regime_summary[regime_summary["method"].isin(hard_candidates)]
                .sort_values("accuracy_mean", ascending=False)
                .iloc[0]["method"]
            )
            best_hard = str(best_hard)
        guarded = None
        if audit_mode == "repeat":
            guarded = "e36_frozen_guarded_repeat"
        elif audit_mode == "stable":
            guarded = "e36_frozen_guarded_stable"

        contrasts = [
            ("best_soft_vs_lda_full", best_soft, "lda_full"),
            ("best_hard_vs_lda_full", best_hard, "lda_full"),
            ("best_soft_vs_best_hard", best_soft, best_hard),
            ("ridge_full_vs_lda_full", "ridge_full_a100", "lda_full"),
            ("guarded_vs_lda_full", guarded, "lda_full"),
            ("guarded_vs_best_soft", guarded, best_soft),
            ("guarded_vs_best_hard", guarded, best_hard),
        ]
        for contrast, a, b in contrasts:
            if a is None or b is None:
                continue
            result = method_diff(matrix, a, b, rng=rng, bootstrap=bootstrap)
            if result is None:
                continue
            result.update(
                {
                    "regime_label": regime,
                    "audit_mode": audit_mode,
                    "contrast": contrast,
                }
            )
            rows.append(result)
    return pd.DataFrame(rows).sort_values(["regime_label", "audit_mode", "contrast"])


def family_interpretation(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (regime, audit_mode), sub in contrasts.groupby(["regime_label", "audit_mode"], sort=True):
        def get_mean(name: str) -> float:
            hit = sub[sub["contrast"] == name]
            if len(hit) == 0:
                return float("nan")
            return float(hit.iloc[0]["mean_diff_pp"])

        soft_vs_hard = get_mean("best_soft_vs_best_hard")
        guarded_vs_lda = get_mean("guarded_vs_lda_full")
        ridge_full_vs_lda = get_mean("ridge_full_vs_lda_full")
        if np.isfinite(soft_vs_hard) and soft_vs_hard > 0.5:
            soft_claim = "soft beats hard top-k"
        elif np.isfinite(soft_vs_hard) and soft_vs_hard < -0.5:
            soft_claim = "hard top-k beats soft"
        else:
            soft_claim = "soft and hard are close"
        if np.isfinite(guarded_vs_lda) and guarded_vs_lda > 1.5:
            guard_claim = "guarded improves over LDA"
        elif np.isfinite(guarded_vs_lda) and guarded_vs_lda >= -0.2:
            guard_claim = "guarded preserves LDA"
        else:
            guard_claim = "guarded underperforms LDA"
        if np.isfinite(ridge_full_vs_lda) and ridge_full_vs_lda < -0.5:
            full_ridge_claim = "full Ridge is weak"
        elif np.isfinite(ridge_full_vs_lda) and ridge_full_vs_lda > 0.5:
            full_ridge_claim = "full Ridge helps"
        else:
            full_ridge_claim = "full Ridge close to LDA"
        rows.append(
            {
                "regime_label": regime,
                "audit_mode": audit_mode,
                "soft_vs_hard_mean_pp": soft_vs_hard,
                "guarded_vs_lda_mean_pp": guarded_vs_lda,
                "ridge_full_vs_lda_mean_pp": ridge_full_vs_lda,
                "soft_claim": soft_claim,
                "guard_claim": guard_claim,
                "full_ridge_claim": full_ridge_claim,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args.e32_records, args.e36_records)
    method_summary = summarize_methods(records, bootstrap=int(args.bootstrap), seed=int(args.seed))
    family_best = best_by_family(method_summary)
    contrasts = key_contrasts(records, method_summary, bootstrap=int(args.bootstrap), seed=int(args.seed))
    interpretation = family_interpretation(contrasts)

    records.to_csv(args.output_dir / "e37_ablation_records.csv", index=False)
    method_summary.to_csv(args.output_dir / "e37_method_summary.csv", index=False)
    family_best.to_csv(args.output_dir / "e37_family_best_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "e37_key_contrasts.csv", index=False)
    interpretation.to_csv(args.output_dir / "e37_interpretation.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "e32_records": [str(p) for p in args.e32_records],
                    "e36_records": str(args.e36_records),
                },
                "method_summary": method_summary.to_dict(orient="records"),
                "family_best_summary": family_best.to_dict(orient="records"),
                "key_contrasts": contrasts.to_dict(orient="records"),
                "interpretation": interpretation.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "method_summary": str(args.output_dir / "e37_method_summary.csv"),
                "key_contrasts": str(args.output_dir / "e37_key_contrasts.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

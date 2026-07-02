"""E42 final frozen comparison for source-anchored interpolation.

E41 showed that source validation supports beta around 0.7, but local
subject-level family/beta switching is noisy.  E42 freezes the final rule:

    regime-level family + regime-level beta=0.7

and compares it against the key baselines:

* LDA full
* regime-family beta=1.0 weighted Ridge
* beta=0.6 risk-first alternative
* E36 frozen guarded selector
* E38 binary fallback
* E41 local source-selected beta
* best hard top-k baseline (descriptive family-best)

No new models are fitted here.  E42 is a consolidation/audit table over E39 and
E41 prediction records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_E39 = RESULTS_DIR / "260701_e39_frozen_method_comparison" / "e39_unit_records.csv"
DEFAULT_E41 = RESULTS_DIR / "260701_e41_nested_source_beta_selection" / "e41_combined_outer_records.csv"
DEFAULT_E41_SELECTION = (
    RESULTS_DIR / "260701_e41_nested_source_beta_selection" / "e41_combined_selection_records.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e42_final_rule_comparison"

HARM_THRESHOLD_PP = -5.0

REGIME_METHODS = {
    "lee2019_sensorimotor20": {
        "final_beta07": "fixed_source_interp_b0p70",
        "risk_beta06": "fixed_source_interp_b0p60",
        "regime_beta1": "fixed_source_interp_b1p00",
    },
    "bnci2014_001_full_source": {
        "final_beta07": "lda_full",
        "risk_beta06": "lda_full",
        "regime_beta1": "lda_full",
    },
    "bnci2014_001_m8_per_class": {
        "final_beta07": "fixed_longitudinal_interp_b0p70",
        "risk_beta06": "fixed_longitudinal_interp_b0p60",
        "regime_beta1": "fixed_longitudinal_interp_b1p00",
    },
}

E39_METHODS = {
    "e36_frozen_guarded_repeat": "e36_frozen_guarded_repeat",
    "e38_binary_fallback": "e38_local_guard_global_family_repeat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e39-records", type=Path, default=DEFAULT_E39)
    parser.add_argument("--e41-records", type=Path, default=DEFAULT_E41)
    parser.add_argument("--e41-selection", type=Path, default=DEFAULT_E41_SELECTION)
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


def cvar_lower_10(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.10 * arr.size)))
    return float(arr[:k].mean())


def add_records(
    rows: list[dict[str, object]],
    source: pd.DataFrame,
    *,
    regime: str,
    source_method: str,
    final_method: str,
    record_source: str,
    note: str,
) -> None:
    sub = source[(source["regime_label"] == regime) & (source["method"] == source_method)].copy()
    if len(sub) == 0:
        raise ValueError((regime, source_method, record_source))
    for row in sub.itertuples():
        rows.append(
            {
                "regime_label": regime,
                "subject": int(row.subject),
                "repeat": int(row.repeat),
                "method": final_method,
                "source_method": source_method,
                "record_source": record_source,
                "note": note,
                "accuracy": float(row.accuracy),
                "lda_full_accuracy": float(row.lda_full_accuracy),
                "gain_vs_lda_full_pp": float(row.gain_vs_lda_full_pp),
            }
        )


def best_hard_by_regime(e39: pd.DataFrame) -> dict[str, str]:
    hard_prefixes = ("lda_source_", "lda_long_", "ridge_hard_")
    rows = []
    for (regime, method), sub in e39.groupby(["regime_label", "method"], sort=True):
        if not str(method).startswith(hard_prefixes):
            continue
        by_subject = sub.groupby("subject", sort=True)["gain_vs_lda_full_pp"].mean()
        rows.append(
            {
                "regime_label": regime,
                "method": method,
                "subject_mean_gain": float(by_subject.mean()),
            }
        )
    hard = pd.DataFrame(rows)
    hard = hard.sort_values(["regime_label", "subject_mean_gain"], ascending=[True, False])
    return {
        str(row.regime_label): str(row.method)
        for row in hard.groupby("regime_label", sort=True).head(1).itertuples()
    }


def build_records(e39: pd.DataFrame, e41: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hard_map = best_hard_by_regime(e39)
    for regime, mapping in REGIME_METHODS.items():
        add_records(
            rows,
            e41,
            regime=regime,
            source_method="lda_full",
            final_method="lda_full",
            record_source="e41",
            note="source anchor",
        )
        for final_method, source_method in mapping.items():
            add_records(
                rows,
                e41,
                regime=regime,
                source_method=source_method,
                final_method=final_method,
                record_source="e41",
                note="frozen regime rule",
            )
        add_records(
            rows,
            e41,
            regime=regime,
            source_method="e41_source_selected_beta",
            final_method="e41_local_selected",
            record_source="e41",
            note="local source-validation selector",
        )
        for final_method, source_method in E39_METHODS.items():
            add_records(
                rows,
                e39,
                regime=regime,
                source_method=source_method,
                final_method=final_method,
                record_source="e39",
                note="previous guarded baseline",
            )
        if regime in hard_map:
            add_records(
                rows,
                e39,
                regime=regime,
                source_method=hard_map[regime],
                final_method="best_hard_topk_descriptive",
                record_source="e39",
                note=f"descriptive best hard top-k: {hard_map[regime]}",
            )
    records = pd.DataFrame(rows)
    records = records.drop_duplicates(["regime_label", "subject", "repeat", "method"], keep="first")
    return records


def summarize(records: pd.DataFrame, *, level: str, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (regime, method), sub in records.groupby(["regime_label", "method"], sort=True):
        if level == "unit":
            acc = sub["accuracy"].to_numpy(dtype=np.float64)
            gain = sub["gain_vs_lda_full_pp"].to_numpy(dtype=np.float64)
            n_units = len(sub)
        elif level == "subject":
            grouped = (
                sub.groupby("subject", sort=True)
                .agg(accuracy=("accuracy", "mean"), gain_vs_lda_full_pp=("gain_vs_lda_full_pp", "mean"))
                .reset_index()
            )
            acc = grouped["accuracy"].to_numpy(dtype=np.float64)
            gain = grouped["gain_vs_lda_full_pp"].to_numpy(dtype=np.float64)
            n_units = len(grouped)
        else:
            raise ValueError(level)
        first = sub.iloc[0]
        rows.append(
            {
                "regime_label": regime,
                "summary_level": level,
                "method": method,
                "n_units": int(n_units),
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(np.mean(gain)),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "gain_vs_lda_full_cvar10_pp": cvar_lower_10(gain),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_vs_lda_full_lt_0": float(np.mean(gain < 0.0)),
                "source_methods": json.dumps(sorted(sub["source_method"].unique().tolist())),
                "note": str(first["note"]),
            }
        )
    return pd.DataFrame(rows)


def values_for(records: pd.DataFrame, regime: str, method: str, level: str) -> pd.Series:
    sub = records[(records["regime_label"] == regime) & (records["method"] == method)].copy()
    if level == "unit":
        return sub.set_index(["subject", "repeat"])["gain_vs_lda_full_pp"].sort_index()
    if level == "subject":
        return sub.groupby("subject", sort=True)["gain_vs_lda_full_pp"].mean()
    raise ValueError(level)


def paired_contrast(
    records: pd.DataFrame,
    *,
    regime: str,
    level: str,
    method_a: str,
    method_b: str,
    label: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object] | None:
    a = values_for(records, regime, method_a, level)
    b = values_for(records, regime, method_b, level)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    diff = (a.loc[common] - b.loc[common]).to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    return {
        "regime_label": regime,
        "summary_level": level,
        "contrast": label,
        "method_a": method_a,
        "method_b": method_b,
        "n_units": int(len(diff)),
        "mean_diff_pp": float(np.mean(diff)),
        "diff_95ci": bootstrap_ci(diff, rng, bootstrap),
        "q05_diff_pp": float(np.quantile(diff, 0.05)),
        "cvar10_diff_pp": cvar_lower_10(diff),
        "p_diff_lt_minus5": float(np.mean(diff < HARM_THRESHOLD_PP)),
        "p_diff_lt_0": float(np.mean(diff < 0.0)),
        "p_diff_gt_0": float(np.mean(diff > 0.0)),
    }


def build_contrasts(records: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("final_beta07", "lda_full", "final_vs_lda"),
        ("final_beta07", "regime_beta1", "final_vs_beta1"),
        ("final_beta07", "risk_beta06", "final_vs_risk_beta06"),
        ("final_beta07", "e36_frozen_guarded_repeat", "final_vs_e36"),
        ("final_beta07", "e38_binary_fallback", "final_vs_e38_binary"),
        ("final_beta07", "e41_local_selected", "final_vs_e41_local"),
        ("final_beta07", "best_hard_topk_descriptive", "final_vs_best_hard"),
        ("risk_beta06", "regime_beta1", "risk_beta06_vs_beta1"),
    ]
    for regime in sorted(records["regime_label"].unique()):
        for level in ["subject", "unit"]:
            for method_a, method_b, label in comparisons:
                result = paired_contrast(
                    records,
                    regime=regime,
                    level=level,
                    method_a=method_a,
                    method_b=method_b,
                    label=label,
                    bootstrap=bootstrap,
                    seed=seed,
                )
                if result is not None:
                    rows.append(result)
    return pd.DataFrame(rows)


def selection_support(selection_path: Path) -> pd.DataFrame:
    selection = pd.read_csv(selection_path)
    rows = []
    for regime, sub in selection.groupby("regime_label", sort=True):
        rows.append(
            {
                "regime_label": regime,
                "n_subjects": int(len(sub)),
                "mode_family": str(sub["selected_family"].mode().iloc[0]),
                "mode_beta": float(sub["selected_beta"].mode().iloc[0]),
                "mean_beta": float(sub["selected_beta"].mean()),
                "family_counts": json.dumps(sub["selected_family"].value_counts().to_dict(), sort_keys=True),
                "beta_counts": json.dumps(sub["selected_beta"].round(3).value_counts().to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def pass_fail(summary: pd.DataFrame, contrasts: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def metric(regime: str, level: str, method: str, col: str) -> float:
        sub = summary[
            (summary["regime_label"] == regime)
            & (summary["summary_level"] == level)
            & (summary["method"] == method)
        ]
        if len(sub) != 1:
            raise ValueError((regime, level, method, col, len(sub)))
        return float(sub.iloc[0][col])

    def contrast(regime: str, level: str, label: str, col: str) -> float:
        sub = contrasts[
            (contrasts["regime_label"] == regime)
            & (contrasts["summary_level"] == level)
            & (contrasts["contrast"] == label)
        ]
        if len(sub) != 1:
            raise ValueError((regime, level, label, col, len(sub)))
        return float(sub.iloc[0][col])

    rows.append(
        {
            "criterion": "Lee final beta=0.7 gain >= +2.4pp",
            "value": metric("lee2019_sensorimotor20", "subject", "final_beta07", "gain_vs_lda_full_mean_pp"),
            "threshold": ">= 2.4",
            "pass": metric("lee2019_sensorimotor20", "subject", "final_beta07", "gain_vs_lda_full_mean_pp") >= 2.4,
        }
    )
    rows.append(
        {
            "criterion": "Lee final harm P<-5 no worse than beta=1",
            "value": contrast("lee2019_sensorimotor20", "subject", "final_vs_beta1", "p_diff_lt_minus5"),
            "threshold": "paired P(final-beta1 < -5) <= 0.03",
            "pass": contrast("lee2019_sensorimotor20", "subject", "final_vs_beta1", "p_diff_lt_minus5") <= 0.03,
        }
    )
    rows.append(
        {
            "criterion": "BNCI full final stays exactly LDA",
            "value": metric("bnci2014_001_full_source", "subject", "final_beta07", "gain_vs_lda_full_mean_pp"),
            "threshold": "abs(gain) <= 0.01",
            "pass": abs(metric("bnci2014_001_full_source", "subject", "final_beta07", "gain_vs_lda_full_mean_pp")) <= 0.01,
        }
    )
    rows.append(
        {
            "criterion": "BNCI m8 unit final gain >= +2.3pp",
            "value": metric("bnci2014_001_m8_per_class", "unit", "final_beta07", "gain_vs_lda_full_mean_pp"),
            "threshold": ">= 2.3",
            "pass": metric("bnci2014_001_m8_per_class", "unit", "final_beta07", "gain_vs_lda_full_mean_pp") >= 2.3,
        }
    )
    rows.append(
        {
            "criterion": "BNCI m8 unit final P<-5 <= E38 binary fallback",
            "value": metric("bnci2014_001_m8_per_class", "unit", "final_beta07", "p_gain_vs_lda_full_lt_minus5")
            - metric("bnci2014_001_m8_per_class", "unit", "e38_binary_fallback", "p_gain_vs_lda_full_lt_minus5"),
            "threshold": "<= 0.0",
            "pass": (
                metric("bnci2014_001_m8_per_class", "unit", "final_beta07", "p_gain_vs_lda_full_lt_minus5")
                <= metric("bnci2014_001_m8_per_class", "unit", "e38_binary_fallback", "p_gain_vs_lda_full_lt_minus5")
            ),
        }
    )
    for regime in ["lee2019_sensorimotor20", "bnci2014_001_m8_per_class"]:
        sub = support[support["regime_label"] == regime]
        mode_beta = float(sub.iloc[0]["mode_beta"]) if len(sub) else float("nan")
        rows.append(
            {
                "criterion": f"{regime} source-validation modal beta is 0.7",
                "value": mode_beta,
                "threshold": "== 0.7",
                "pass": bool(np.isclose(mode_beta, 0.7)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    e39 = pd.read_csv(args.e39_records)
    e41 = pd.read_csv(args.e41_records)
    records = build_records(e39, e41)
    summary = pd.concat(
        [
            summarize(records, level="subject", bootstrap=int(args.bootstrap), seed=int(args.seed)),
            summarize(records, level="unit", bootstrap=int(args.bootstrap), seed=int(args.seed)),
        ],
        ignore_index=True,
    )
    contrasts = build_contrasts(records, bootstrap=int(args.bootstrap), seed=int(args.seed))
    support = selection_support(args.e41_selection)
    checks = pass_fail(summary, contrasts, support)

    records.to_csv(args.output_dir / "e42_final_records.csv", index=False)
    summary.to_csv(args.output_dir / "e42_final_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "e42_key_contrasts.csv", index=False)
    support.to_csv(args.output_dir / "e42_source_validation_support.csv", index=False)
    checks.to_csv(args.output_dir / "e42_pass_fail.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "e39_records": str(args.e39_records),
                    "e41_records": str(args.e41_records),
                    "e41_selection": str(args.e41_selection),
                },
                "frozen_rule": REGIME_METHODS,
                "pass_fail": checks.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary": str(args.output_dir / "e42_final_summary.csv"),
                "contrasts": str(args.output_dir / "e42_key_contrasts.csv"),
                "pass_fail": str(args.output_dir / "e42_pass_fail.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

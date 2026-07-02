"""E39 frozen comparison for regime-level reliability-weighted Ridge.

E38 suggested that the cleaner deployable control structure is:

    choose a reliability-weighted family at the regime level,
    then keep a local LDA fallback.

E39 freezes that interpretation and compares it against the previously tested
baselines in one table:

* LDA full source anchor
* fixed soft reliability-weighted Ridge
* fixed hard top-k baselines
* E36 frozen guarded selector
* E38 regime-level family variants

This script does not fit new models.  It re-aggregates existing outer records.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "research_outputs"
DEFAULT_E37 = RESULTS_DIR / "260701_e37_soft_vs_hard_ablation" / "e37_ablation_records.csv"
DEFAULT_E38 = RESULTS_DIR / "260701_e38_regime_level_family_rule"
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e39_frozen_method_comparison"

LDA = "lda_full"
PRACTICAL = "e38_local_guard_global_family_repeat"
GLOBAL = "e38_global_regime_family"
E36_REPEAT = "e36_frozen_guarded_repeat"
E36_STABLE = "e36_frozen_guarded_stable"
HARM_THRESHOLD_PP = -5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e37-records", type=Path, default=DEFAULT_E37)
    parser.add_argument("--e38-dir", type=Path, default=DEFAULT_E38)
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


def normalize_e37(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path)
    records = records.copy()
    records["record_family"] = records.get("family", "")
    records["record_source"] = records.get("record_source", "e37")
    records["audit_mode"] = records.get("audit_mode", "fixed")
    records["selected_candidate"] = records.get("selected_candidate", records["method"])
    records.loc[records["selected_candidate"].isna(), "selected_candidate"] = records.loc[
        records["selected_candidate"].isna(), "method"
    ]
    return records[
        [
            "regime_label",
            "subject",
            "repeat",
            "method",
            "selected_candidate",
            "accuracy",
            "audit_mode",
            "record_family",
            "record_source",
        ]
    ]


def normalize_e38(root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob("*/e38_records.csv")):
        df = pd.read_csv(path)
        keep = df[df["method"].isin([GLOBAL, PRACTICAL, "e38_local_guard_global_family_stable"])].copy()
        keep["audit_mode"] = keep["method"].map(
            {
                GLOBAL: "global",
                PRACTICAL: "repeat",
                "e38_local_guard_global_family_stable": "stable",
            }
        )
        keep["record_family"] = "regime_family_rule"
        keep["record_source"] = "e38"
        frames.append(
            keep[
                [
                    "regime_label",
                    "subject",
                    "repeat",
                    "method",
                    "selected_candidate",
                    "accuracy",
                    "audit_mode",
                    "record_family",
                    "record_source",
                ]
            ]
        )
    if not frames:
        raise FileNotFoundError(f"no e38_records.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def baseline_lookup(records: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    lda = records[records["method"] == LDA].copy()
    lda = lda.sort_values(["regime_label", "subject", "repeat", "audit_mode"]).drop_duplicates(
        ["regime_label", "subject", "repeat"]
    )
    return {
        (str(row.regime_label), int(row.subject), int(row.repeat)): float(row.accuracy)
        for row in lda.itertuples()
    }


def unit_table(records: pd.DataFrame, baseline: dict[tuple[str, int, int], float]) -> pd.DataFrame:
    rows = []
    for row in records.itertuples():
        key = (str(row.regime_label), int(row.subject), int(row.repeat))
        if key not in baseline:
            continue
        rows.append(
            {
                "regime_label": str(row.regime_label),
                "subject": int(row.subject),
                "repeat": int(row.repeat),
                "method": str(row.method),
                "selected_candidate": str(row.selected_candidate),
                "accuracy": float(row.accuracy),
                "lda_full_accuracy": float(baseline[key]),
                "gain_vs_lda_full_pp": float(row.accuracy) - float(baseline[key]),
                "audit_mode": str(row.audit_mode),
                "record_family": str(row.record_family),
                "record_source": str(row.record_source),
            }
        )
    return pd.DataFrame(rows)


def summarize(unit_records: pd.DataFrame, *, level: str, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (regime, method), sub in unit_records.groupby(["regime_label", "method"], sort=True):
        if level == "unit":
            values = sub[["accuracy", "gain_vs_lda_full_pp"]].to_numpy(dtype=np.float64)
            acc = values[:, 0]
            gain = values[:, 1]
            n_units = len(sub)
        elif level == "subject":
            by_subject = (
                sub.groupby("subject", sort=True)
                .agg(accuracy=("accuracy", "mean"), gain_vs_lda_full_pp=("gain_vs_lda_full_pp", "mean"))
                .reset_index()
            )
            acc = by_subject["accuracy"].to_numpy(dtype=np.float64)
            gain = by_subject["gain_vs_lda_full_pp"].to_numpy(dtype=np.float64)
            n_units = len(by_subject)
        else:
            raise ValueError(level)
        rows.append(
            {
                "regime_label": regime,
                "summary_level": level,
                "method": method,
                "n_units": int(n_units),
                "accuracy_mean": float(acc.mean()),
                "accuracy_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(gain.mean()),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "gain_vs_lda_full_cvar10_pp": cvar_lower_10(gain),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_vs_lda_full_lt_0": float(np.mean(gain < 0.0)),
                "selection_counts": json.dumps(
                    sub["selected_candidate"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "record_sources": json.dumps(sorted(sub["record_source"].unique().tolist())),
            }
        )
    return pd.DataFrame(rows)


def subject_or_unit_values(unit_records: pd.DataFrame, regime: str, method: str, level: str) -> pd.Series:
    sub = unit_records[(unit_records["regime_label"] == regime) & (unit_records["method"] == method)].copy()
    if level == "unit":
        return sub.set_index(["subject", "repeat"])["gain_vs_lda_full_pp"].sort_index()
    if level == "subject":
        return sub.groupby("subject", sort=True)["gain_vs_lda_full_pp"].mean()
    raise ValueError(level)


def paired_contrast(
    unit_records: pd.DataFrame,
    *,
    regime: str,
    level: str,
    method_a: str,
    method_b: str,
    bootstrap: int,
    seed: int,
    label: str,
) -> dict[str, object] | None:
    a = subject_or_unit_values(unit_records, regime, method_a, level)
    b = subject_or_unit_values(unit_records, regime, method_b, level)
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
        "mean_diff_pp": float(diff.mean()),
        "diff_95ci": bootstrap_ci(diff, rng, bootstrap),
        "q05_diff_pp": float(np.quantile(diff, 0.05)),
        "cvar10_diff_pp": cvar_lower_10(diff),
        "p_diff_lt_minus5": float(np.mean(diff < HARM_THRESHOLD_PP)),
        "p_diff_lt_0": float(np.mean(diff < 0.0)),
        "p_diff_gt_0": float(np.mean(diff > 0.0)),
    }


def best_hard_methods(summary: pd.DataFrame) -> pd.DataFrame:
    hard_prefixes = ("lda_source_", "lda_long_", "ridge_hard_")
    hard = summary[
        (summary["summary_level"] == "subject")
        & (summary["method"].map(lambda name: str(name).startswith(hard_prefixes)))
    ].copy()
    hard = hard.sort_values(["regime_label", "gain_vs_lda_full_mean_pp"], ascending=[True, False])
    return hard.groupby("regime_label", sort=True).head(1)[
        ["regime_label", "method", "gain_vs_lda_full_mean_pp"]
    ].rename(columns={"method": "best_hard_method", "gain_vs_lda_full_mean_pp": "best_hard_gain_pp"})


def build_contrasts(unit_records: pd.DataFrame, summary: pd.DataFrame, *, bootstrap: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    regimes = sorted(unit_records["regime_label"].unique().tolist())
    hard_map = dict(zip(best_hard_methods(summary)["regime_label"], best_hard_methods(summary)["best_hard_method"]))
    requested = [
        (PRACTICAL, LDA, "practical_vs_lda_full"),
        (GLOBAL, LDA, "global_vs_lda_full"),
        (PRACTICAL, E36_REPEAT, "practical_vs_e36_repeat"),
        (GLOBAL, E36_REPEAT, "global_vs_e36_repeat"),
        (PRACTICAL, E36_STABLE, "practical_vs_e36_stable"),
    ]
    for regime in regimes:
        if regime in hard_map:
            requested_with_hard = requested + [(PRACTICAL, hard_map[regime], "practical_vs_best_hard")]
        else:
            requested_with_hard = requested
        for level in ["subject", "unit"]:
            for method_a, method_b, label in requested_with_hard:
                result = paired_contrast(
                    unit_records,
                    regime=regime,
                    level=level,
                    method_a=method_a,
                    method_b=method_b,
                    bootstrap=bootstrap,
                    seed=seed,
                    label=label,
                )
                if result is not None:
                    rows.append(result)
    return pd.DataFrame(rows)


def pass_fail(summary: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def summary_row(regime: str, level: str, method: str) -> pd.Series:
        sub = summary[
            (summary["regime_label"] == regime)
            & (summary["summary_level"] == level)
            & (summary["method"] == method)
        ]
        if len(sub) != 1:
            raise ValueError((regime, level, method, len(sub)))
        return sub.iloc[0]

    def contrast_row(regime: str, level: str, label: str) -> pd.Series:
        sub = contrasts[
            (contrasts["regime_label"] == regime)
            & (contrasts["summary_level"] == level)
            & (contrasts["contrast"] == label)
        ]
        if len(sub) != 1:
            raise ValueError((regime, level, label, len(sub)))
        return sub.iloc[0]

    lee = contrast_row("lee2019_sensorimotor20", "subject", "practical_vs_e36_repeat")
    rows.append(
        {
            "criterion": "Lee subject: practical beats E36 repeat",
            "value": float(lee["mean_diff_pp"]),
            "threshold": ">= 0.0 pp",
            "pass": bool(float(lee["mean_diff_pp"]) >= 0.0),
        }
    )
    bnci_full = summary_row("bnci2014_001_full_source", "subject", PRACTICAL)
    rows.append(
        {
            "criterion": "BNCI full subject: practical keeps LDA anchor",
            "value": float(bnci_full["gain_vs_lda_full_mean_pp"]),
            "threshold": "abs(gain) <= 0.1 pp and P<-5 = 0",
            "pass": bool(
                abs(float(bnci_full["gain_vs_lda_full_mean_pp"])) <= 0.1
                and float(bnci_full["p_gain_vs_lda_full_lt_minus5"]) == 0.0
            ),
        }
    )
    m8 = contrast_row("bnci2014_001_m8_per_class", "unit", "practical_vs_e36_repeat")
    m8_practical = summary_row("bnci2014_001_m8_per_class", "unit", PRACTICAL)
    m8_e36 = summary_row("bnci2014_001_m8_per_class", "unit", E36_REPEAT)
    harm_delta = float(m8_practical["p_gain_vs_lda_full_lt_minus5"]) - float(
        m8_e36["p_gain_vs_lda_full_lt_minus5"]
    )
    rows.append(
        {
            "criterion": "BNCI m8 unit: practical improves E36 mean by >= +0.1pp",
            "value": float(m8["mean_diff_pp"]),
            "threshold": ">= 0.1 pp",
            "pass": bool(float(m8["mean_diff_pp"]) >= 0.1),
        }
    )
    rows.append(
        {
            "criterion": "BNCI m8 unit: practical harm not worse than E36 by > +1pp",
            "value": harm_delta,
            "threshold": "<= 0.01 probability",
            "pass": bool(harm_delta <= 0.01),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    e37 = normalize_e37(args.e37_records)
    e38 = normalize_e38(args.e38_dir)
    records = pd.concat([e37, e38], ignore_index=True, sort=False)
    # E37 stores fixed baselines under fixed/repeat/stable audit views.  They
    # are identical prediction records, so each method must appear once per
    # regime/subject/repeat before paired contrasts are formed.
    records = records.drop_duplicates(["regime_label", "subject", "repeat", "method"], keep="first")

    unit_records = unit_table(records, baseline_lookup(records))
    summaries = pd.concat(
        [
            summarize(unit_records, level="subject", bootstrap=int(args.bootstrap), seed=int(args.seed)),
            summarize(unit_records, level="unit", bootstrap=int(args.bootstrap), seed=int(args.seed)),
        ],
        ignore_index=True,
    )
    contrasts = build_contrasts(unit_records, summaries, bootstrap=int(args.bootstrap), seed=int(args.seed))
    checks = pass_fail(summaries, contrasts)

    unit_records.to_csv(args.output_dir / "e39_unit_records.csv", index=False)
    summaries.to_csv(args.output_dir / "e39_method_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "e39_key_contrasts.csv", index=False)
    checks.to_csv(args.output_dir / "e39_pass_fail.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "inputs": {
                    "e37_records": str(args.e37_records),
                    "e38_dir": str(args.e38_dir),
                },
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
                "method_summary": str(args.output_dir / "e39_method_summary.csv"),
                "key_contrasts": str(args.output_dir / "e39_key_contrasts.csv"),
                "pass_fail": str(args.output_dir / "e39_pass_fail.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

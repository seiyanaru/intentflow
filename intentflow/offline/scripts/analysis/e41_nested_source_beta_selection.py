"""E41 nested source-validation beta selection.

E40 found that source-anchored interpolation can dominate the binary
LDA-vs-weighted-Ridge fallback, but beta=0.6/0.7 was interpreted after seeing
held target results.  E41 asks whether beta can be selected using source-side
validation only.

Protocol for each outer held subject:

1. Exclude the held subject from all selection.
2. On the remaining source subjects, run leave-one-source-subject validation.
3. Select family among {LDA, source-weighted Ridge, longitudinal-weighted Ridge}.
4. If a weighted family is selected, select beta from {0.0, ..., 1.0} using
   only the inner validation risk-utility frontier.
5. Fit on the held subject's source session and evaluate once on its target
   session.

This is stricter than E38/E40's global same-regime diagnostic because the held
subject never contributes target labels to family/beta selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from e40_source_anchor_interpolation import (
    BNCI_CACHE,
    HARM_THRESHOLD_PP,
    LEE_CACHE,
    accuracy_from_margin,
    beta_token,
    cvar_lower_10,
    fit_binary_lda_margins,
    fit_binary_ridge_margins,
    scale_by_source_margin,
)
from weighted_ridge_nested_selector_e33 import aggregate_stats_excluding
from weighted_ridge_regime_transfer_e32 import (
    RESULTS_DIR,
    balanced_subset,
    bootstrap_ci,
    load_payload,
    longitudinal_score,
    parse_subjects,
    rank_weights,
    source_score,
)


DEFAULT_OUTPUT = RESULTS_DIR / "260701_e41_nested_source_beta_selection"

REGIMES = [
    {
        "regime_label": "lee2019_sensorimotor20",
        "cache_dir": LEE_CACHE,
        "e33_dir": RESULTS_DIR / "260701_e33_lee2019_sensorimotor20_nested_selector",
        "subjects": "1-54",
        "per_class": 0,
        "repeats": 1,
    },
    {
        "regime_label": "bnci2014_001_full_source",
        "cache_dir": BNCI_CACHE,
        "e33_dir": RESULTS_DIR / "260701_e33_bnci2014_001_full_source_nested_selector",
        "subjects": "1-9",
        "per_class": 0,
        "repeats": 1,
    },
    {
        "regime_label": "bnci2014_001_m8_per_class",
        "cache_dir": BNCI_CACHE,
        "e33_dir": RESULTS_DIR / "260701_e33_bnci2014_001_m8_nested_selector",
        "subjects": "1-9",
        "per_class": 8,
        "repeats": 32,
    },
]

FAMILIES = ["source", "longitudinal"]
FAMILY_TO_METHOD = {
    "source": "ridge_source_rank_g2_a100",
    "longitudinal": "ridge_longitudinal_rank_g2_a100",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--betas", nargs="*", type=float, default=[i / 10 for i in range(11)])
    parser.add_argument("--family-margin", type=float, default=0.5)
    parser.add_argument("--family-risk-limit", type=float, default=0.20)
    parser.add_argument("--beta-risk-limit", type=float, default=0.08)
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def source_indices(labels: np.ndarray, *, subject: int, repeat: int, per_class: int, seed: int) -> np.ndarray:
    if int(per_class) <= 0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1000003 * int(subject) + int(repeat))
    return balanced_subset(labels, int(per_class), rng)


def family_weights(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    *,
    excluded_subjects: Sequence[int],
    family: str,
) -> np.ndarray:
    stats = aggregate_stats_excluding(payloads, excluded_subjects)
    if family == "source":
        return rank_weights(source_score(stats), gamma=2.0)
    if family == "longitudinal":
        return rank_weights(longitudinal_score(stats), gamma=2.0)
    raise ValueError(family)


def margins_for_unit(
    payload: Mapping[str, np.ndarray],
    source_idx: np.ndarray,
    *,
    weights_by_family: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    xs_all = np.asarray(payload["source_features"], dtype=np.float64)
    ys_all = np.asarray(payload["source_labels"], dtype=np.int64)
    xt = np.asarray(payload["target_features"], dtype=np.float64)
    yt = np.asarray(payload["target_labels"], dtype=np.int64)
    xs = xs_all[source_idx]
    ys = ys_all[source_idx]

    lda_source_margin, lda_target_margin, classes = fit_binary_lda_margins(xs, ys, xt)
    lda_margin = scale_by_source_margin(lda_source_margin, lda_target_margin)
    ridge_margins = {}
    for family, weights in weights_by_family.items():
        ridge_source_margin, ridge_target_margin, ridge_classes = fit_binary_ridge_margins(
            xs,
            ys,
            xt,
            weights=np.asarray(weights, dtype=np.float64),
        )
        if not np.array_equal(classes, ridge_classes):
            raise ValueError((classes, ridge_classes))
        ridge_margins[family] = scale_by_source_margin(ridge_source_margin, ridge_target_margin)
    return lda_margin, ridge_margins, classes, yt


def interpolation_accuracy(
    lda_margin: np.ndarray,
    ridge_margin: np.ndarray,
    *,
    beta: float,
    classes: np.ndarray,
    labels: np.ndarray,
) -> float:
    beta = float(beta)
    margin = (1.0 - beta) * lda_margin + beta * ridge_margin
    return accuracy_from_margin(margin, classes, labels)


def build_inner_validation(
    *,
    regime: str,
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    subjects: Sequence[int],
    held_subject: int,
    per_class: int,
    repeats: int,
    betas: Sequence[float],
    families: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    rows = []
    validation_subjects = [subject for subject in subjects if int(subject) != int(held_subject)]
    for inner_subject in validation_subjects:
        payload = payloads[int(inner_subject)]
        weights_by_family = {
            family: family_weights(
                payloads,
                excluded_subjects=[int(held_subject), int(inner_subject)],
                family=family,
            )
            for family in families
        }
        labels = np.asarray(payload["source_labels"], dtype=np.int64)
        for repeat in range(int(repeats)):
            idx = source_indices(
                labels,
                subject=int(inner_subject),
                repeat=int(repeat),
                per_class=int(per_class),
                seed=int(seed),
            )
            lda_margin, ridge_margins, classes, yt = margins_for_unit(
                payload,
                idx,
                weights_by_family=weights_by_family,
            )
            lda_acc = accuracy_from_margin(lda_margin, classes, yt)
            rows.append(
                {
                    "regime_label": regime,
                    "outer_subject": int(held_subject),
                    "inner_subject": int(inner_subject),
                    "repeat": int(repeat),
                    "family": "lda",
                    "beta": 0.0,
                    "method": "lda_full",
                    "accuracy": lda_acc,
                    "lda_accuracy": lda_acc,
                    "gain_vs_lda_full_pp": 0.0,
                }
            )
            for family, ridge_margin in ridge_margins.items():
                for beta in betas:
                    acc = interpolation_accuracy(
                        lda_margin,
                        ridge_margin,
                        beta=float(beta),
                        classes=classes,
                        labels=yt,
                    )
                    rows.append(
                        {
                            "regime_label": regime,
                            "outer_subject": int(held_subject),
                            "inner_subject": int(inner_subject),
                            "repeat": int(repeat),
                            "family": family,
                            "beta": float(beta),
                            "method": f"{family}_interp_{beta_token(float(beta))}",
                            "accuracy": acc,
                            "lda_accuracy": lda_acc,
                            "gain_vs_lda_full_pp": acc - lda_acc,
                        }
                    )
    return pd.DataFrame(rows)


def family_summary_from_e33(e33_dir: Path, *, held_subject: int) -> pd.DataFrame:
    validation = pd.read_csv(e33_dir / "nested_inner_validation_records.csv")
    validation = validation[validation["outer_subject"] == int(held_subject)].copy()
    lda = validation[validation["candidate"] == "lda_full"][
        ["repeat", "inner_subject", "accuracy"]
    ].rename(columns={"accuracy": "lda_accuracy"})
    rows = []
    for family, candidate in {
        "source": "ridge_source_rank_g2_a100",
        "longitudinal": "ridge_longitudinal_rank_g2_a100",
    }.items():
        sub = validation[validation["candidate"] == candidate].copy()
        merged = sub.merge(lda, on=["repeat", "inner_subject"], how="left")
        gain = merged["accuracy"].to_numpy(dtype=np.float64) - merged["lda_accuracy"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "family": family,
                "n_validation_units": int(len(gain)),
                "mean_gain_pp": float(np.mean(gain)),
                "q05_gain_pp": float(np.quantile(gain, 0.05)),
                "cvar10_gain_pp": cvar_lower_10(gain),
                "p_gain_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_lt_0": float(np.mean(gain < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def choose_family_from_summary(
    family_summary: pd.DataFrame,
    *,
    family_margin: float,
    family_risk_limit: float,
) -> dict[str, object]:
    ordered = family_summary.sort_values(["mean_gain_pp", "family"], ascending=[False, True])
    best = ordered.iloc[0]
    best_family = str(best["family"])
    best_mean = float(best["mean_gain_pp"])
    best_p_harm = float(best["p_gain_lt_minus5"])
    if best_mean >= float(family_margin) and best_p_harm <= float(family_risk_limit):
        selected_family = best_family
    else:
        selected_family = "lda"
    return {
        "selected_family": selected_family,
        "best_beta1_family": best_family,
        "best_beta1_mean_gain_pp": float(best_mean),
        "best_beta1_p_harm": float(best_p_harm),
        "family_pass_margin": bool(best_mean >= float(family_margin)),
        "family_pass_risk": bool(best_p_harm <= float(family_risk_limit)),
    }


def gain_summary(records: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for key, sub in records.groupby(list(group_cols), sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        gain = sub["gain_vs_lda_full_pp"].to_numpy(dtype=np.float64)
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "n_validation_units": int(len(gain)),
                "mean_gain_pp": float(np.mean(gain)),
                "q05_gain_pp": float(np.quantile(gain, 0.05)),
                "cvar10_gain_pp": cvar_lower_10(gain),
                "p_gain_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_lt_0": float(np.mean(gain < 0.0)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def choose_beta_from_validation(
    validation: pd.DataFrame,
    *,
    selected_family: str,
    beta_risk_limit: float,
) -> tuple[float, float, float, pd.DataFrame]:
    if selected_family == "lda":
        return 0.0, 0.0, 0.0, pd.DataFrame()
    beta_records = validation[validation["family"] == selected_family].copy()
    beta_summary = gain_summary(beta_records, group_cols=["family", "beta"])
    safe = beta_summary[beta_summary["p_gain_lt_minus5"] <= float(beta_risk_limit)].copy()
    if len(safe) == 0:
        safe = beta_summary[np.isclose(beta_summary["beta"], 0.0)].copy()
    safe = safe.sort_values(
        ["mean_gain_pp", "p_gain_lt_minus5", "beta"],
        ascending=[False, True, True],
    )
    selected = safe.iloc[0]
    return (
        float(selected["beta"]),
        float(selected["mean_gain_pp"]),
        float(selected["p_gain_lt_minus5"]),
        beta_summary,
    )


def choose_family_and_beta(
    validation: pd.DataFrame,
    *,
    betas: Sequence[float],
    family_margin: float,
    family_risk_limit: float,
    beta_risk_limit: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    weighted_beta1 = validation[
        (validation["family"].isin(FAMILIES)) & (np.isclose(validation["beta"], 1.0))
    ].copy()
    family_summary = gain_summary(weighted_beta1, group_cols=["family"])
    if len(family_summary) == 0:
        raise ValueError("empty family summary")
    family_summary = family_summary.sort_values(["mean_gain_pp", "family"], ascending=[False, True])
    best_family = str(family_summary.iloc[0]["family"])
    best_mean = float(family_summary.iloc[0]["mean_gain_pp"])
    best_p_harm = float(family_summary.iloc[0]["p_gain_lt_minus5"])
    if best_mean >= float(family_margin) and best_p_harm <= float(family_risk_limit):
        selected_family = best_family
    else:
        selected_family = "lda"

    beta_summary = pd.DataFrame()
    if selected_family == "lda":
        selected_beta = 0.0
        selected_beta_mean = 0.0
        selected_beta_p_harm = 0.0
    else:
        beta_records = validation[validation["family"] == selected_family].copy()
        beta_summary = gain_summary(beta_records, group_cols=["family", "beta"])
        safe = beta_summary[beta_summary["p_gain_lt_minus5"] <= float(beta_risk_limit)].copy()
        if len(safe) == 0:
            safe = beta_summary[np.isclose(beta_summary["beta"], 0.0)].copy()
        # Tie-breaking is deliberately source-anchor conservative after mean/risk.
        safe = safe.sort_values(
            ["mean_gain_pp", "p_gain_lt_minus5", "beta"],
            ascending=[False, True, True],
        )
        selected = safe.iloc[0]
        selected_beta = float(selected["beta"])
        selected_beta_mean = float(selected["mean_gain_pp"])
        selected_beta_p_harm = float(selected["p_gain_lt_minus5"])

    return (
        {
            "selected_family": selected_family,
            "selected_beta": float(selected_beta),
            "selected_beta_mean_gain_pp": float(selected_beta_mean),
            "selected_beta_p_harm": float(selected_beta_p_harm),
            "best_beta1_family": best_family,
            "best_beta1_mean_gain_pp": float(best_mean),
            "best_beta1_p_harm": float(best_p_harm),
            "family_pass_margin": bool(best_mean >= float(family_margin)),
            "family_pass_risk": bool(best_p_harm <= float(family_risk_limit)),
        },
        family_summary,
        beta_summary,
    )


def evaluate_outer(
    *,
    regime: str,
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
    per_class: int,
    repeats: int,
    betas: Sequence[float],
    selected_family: str,
    selected_beta: float,
    seed: int,
) -> pd.DataFrame:
    payload = payloads[int(held_subject)]
    labels = np.asarray(payload["source_labels"], dtype=np.int64)
    weights_by_family = {
        family: family_weights(payloads, excluded_subjects=[int(held_subject)], family=family)
        for family in FAMILIES
    }
    rows = []
    for repeat in range(int(repeats)):
        idx = source_indices(
            labels,
            subject=int(held_subject),
            repeat=int(repeat),
            per_class=int(per_class),
            seed=int(seed),
        )
        lda_margin, ridge_margins, classes, yt = margins_for_unit(
            payload,
            idx,
            weights_by_family=weights_by_family,
        )
        lda_acc = accuracy_from_margin(lda_margin, classes, yt)
        rows.append(
            {
                "regime_label": regime,
                "subject": int(held_subject),
                "repeat": int(repeat),
                "method": "lda_full",
                "family": "lda",
                "beta": 0.0,
                "selected_family": selected_family,
                "selected_beta": float(selected_beta),
                "accuracy": lda_acc,
                "lda_full_accuracy": lda_acc,
                "gain_vs_lda_full_pp": 0.0,
            }
        )
        # Fixed references.
        for family, ridge_margin in ridge_margins.items():
            for beta in betas:
                acc = interpolation_accuracy(
                    lda_margin,
                    ridge_margin,
                    beta=float(beta),
                    classes=classes,
                    labels=yt,
                )
                rows.append(
                    {
                        "regime_label": regime,
                        "subject": int(held_subject),
                        "repeat": int(repeat),
                        "method": f"fixed_{family}_interp_{beta_token(float(beta))}",
                        "family": family,
                        "beta": float(beta),
                        "selected_family": selected_family,
                        "selected_beta": float(selected_beta),
                        "accuracy": acc,
                        "lda_full_accuracy": lda_acc,
                        "gain_vs_lda_full_pp": acc - lda_acc,
                    }
                )
        # Selected source-validation method.
        if selected_family == "lda":
            selected_acc = lda_acc
        else:
            selected_acc = interpolation_accuracy(
                lda_margin,
                ridge_margins[selected_family],
                beta=float(selected_beta),
                classes=classes,
                labels=yt,
            )
        rows.append(
            {
                "regime_label": regime,
                "subject": int(held_subject),
                "repeat": int(repeat),
                "method": "e41_source_selected_beta",
                "family": selected_family,
                "beta": float(selected_beta),
                "selected_family": selected_family,
                "selected_beta": float(selected_beta),
                "accuracy": selected_acc,
                "lda_full_accuracy": lda_acc,
                "gain_vs_lda_full_pp": selected_acc - lda_acc,
            }
        )
    return pd.DataFrame(rows)


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
                "selected_family_counts": json.dumps(
                    sub["selected_family"].value_counts().to_dict(),
                    sort_keys=True,
                ),
                "selected_beta_counts": json.dumps(
                    sub["selected_beta"].round(3).value_counts().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def process_regime(
    config: Mapping[str, object],
    *,
    betas: Sequence[float],
    family_margin: float,
    family_risk_limit: float,
    beta_risk_limit: float,
    bootstrap: int,
    seed: int,
    quiet: bool,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regime = str(config["regime_label"])
    subjects = parse_subjects(str(config["subjects"]))
    payloads = {subject: load_payload(Path(config["cache_dir"]), subject) for subject in subjects}
    e33_dir = Path(config["e33_dir"])
    per_class = int(config["per_class"])
    repeats = int(config["repeats"]) if per_class > 0 else 1

    validation_frames = []
    selection_rows = []
    family_summary_frames = []
    beta_summary_frames = []
    outer_frames = []

    for held_subject in subjects:
        family_summary = family_summary_from_e33(e33_dir, held_subject=int(held_subject))
        family_choice = choose_family_from_summary(
            family_summary,
            family_margin=family_margin,
            family_risk_limit=family_risk_limit,
        )
        selected_family = str(family_choice["selected_family"])
        if selected_family == "lda":
            validation = pd.DataFrame()
            selected_beta = 0.0
            selected_beta_mean = 0.0
            selected_beta_p_harm = 0.0
            beta_summary = pd.DataFrame()
        else:
            validation = build_inner_validation(
                regime=regime,
                payloads=payloads,
                subjects=subjects,
                held_subject=int(held_subject),
                per_class=per_class,
                repeats=repeats,
                betas=betas,
                families=[selected_family],
                seed=seed,
            )
            (
                selected_beta,
                selected_beta_mean,
                selected_beta_p_harm,
                beta_summary,
            ) = choose_beta_from_validation(
                validation,
                selected_family=selected_family,
                beta_risk_limit=beta_risk_limit,
            )
        if len(validation) > 0:
            validation_frames.append(validation)
        selection_rows.append(
            {
                "regime_label": regime,
                "subject": int(held_subject),
                "per_class": per_class,
                "repeats": repeats,
                "n_validation_rows": int(len(validation)),
                **family_choice,
                "selected_beta": float(selected_beta),
                "selected_beta_mean_gain_pp": float(selected_beta_mean),
                "selected_beta_p_harm": float(selected_beta_p_harm),
            }
        )
        family_summary["regime_label"] = regime
        family_summary["outer_subject"] = int(held_subject)
        family_summary_frames.append(family_summary)
        if len(beta_summary) > 0:
            beta_summary["regime_label"] = regime
            beta_summary["outer_subject"] = int(held_subject)
            beta_summary_frames.append(beta_summary)
        outer_frames.append(
            evaluate_outer(
                regime=regime,
                payloads=payloads,
                held_subject=int(held_subject),
                per_class=per_class,
                repeats=repeats,
                betas=betas,
                selected_family=selected_family,
                selected_beta=float(selected_beta),
                seed=seed,
            )
        )
        if not quiet:
            print(
                f"{regime} S{held_subject}: family={selected_family} beta={selected_beta}",
                flush=True,
            )

    validation_records = (
        pd.concat(validation_frames, ignore_index=True, sort=False)
        if validation_frames
        else pd.DataFrame()
    )
    selection_records = pd.DataFrame(selection_rows)
    family_summaries = pd.concat(family_summary_frames, ignore_index=True, sort=False)
    beta_summaries = (
        pd.concat(beta_summary_frames, ignore_index=True, sort=False)
        if beta_summary_frames
        else pd.DataFrame()
    )
    outer_records = pd.concat(outer_frames, ignore_index=True, sort=False)
    summary = pd.concat(
        [
            summarize(outer_records, level="subject", bootstrap=bootstrap, seed=seed),
            summarize(outer_records, level="unit", bootstrap=bootstrap, seed=seed),
        ],
        ignore_index=True,
    )

    regime_dir = output_dir / regime
    regime_dir.mkdir(parents=True, exist_ok=True)
    validation_records.to_csv(regime_dir / "e41_inner_validation_records.csv", index=False)
    selection_records.to_csv(regime_dir / "e41_selection_records.csv", index=False)
    family_summaries.to_csv(regime_dir / "e41_family_summaries.csv", index=False)
    beta_summaries.to_csv(regime_dir / "e41_beta_summaries.csv", index=False)
    outer_records.to_csv(regime_dir / "e41_outer_records.csv", index=False)
    summary.to_csv(regime_dir / "e41_summary.csv", index=False)

    return outer_records, selection_records, family_summaries, beta_summaries


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(dict.fromkeys(float(beta) for beta in args.betas))

    outer_frames = []
    selection_frames = []
    family_frames = []
    beta_frames = []
    for config in REGIMES:
        outer, selection, family_summary, beta_summary = process_regime(
            config,
            betas=betas,
            family_margin=float(args.family_margin),
            family_risk_limit=float(args.family_risk_limit),
            beta_risk_limit=float(args.beta_risk_limit),
            bootstrap=int(args.bootstrap),
            seed=int(args.seed),
            quiet=bool(args.quiet),
            output_dir=args.output_dir,
        )
        outer_frames.append(outer)
        selection_frames.append(selection)
        family_frames.append(family_summary)
        if len(beta_summary) > 0:
            beta_frames.append(beta_summary)

    combined_outer = pd.concat(outer_frames, ignore_index=True, sort=False)
    combined_selection = pd.concat(selection_frames, ignore_index=True, sort=False)
    combined_family = pd.concat(family_frames, ignore_index=True, sort=False)
    combined_beta = pd.concat(beta_frames, ignore_index=True, sort=False) if beta_frames else pd.DataFrame()
    combined_summary = pd.concat(
        [
            summarize(combined_outer, level="subject", bootstrap=int(args.bootstrap), seed=int(args.seed)),
            summarize(combined_outer, level="unit", bootstrap=int(args.bootstrap), seed=int(args.seed)),
        ],
        ignore_index=True,
    )

    combined_outer.to_csv(args.output_dir / "e41_combined_outer_records.csv", index=False)
    combined_selection.to_csv(args.output_dir / "e41_combined_selection_records.csv", index=False)
    combined_family.to_csv(args.output_dir / "e41_combined_family_summaries.csv", index=False)
    combined_beta.to_csv(args.output_dir / "e41_combined_beta_summaries.csv", index=False)
    combined_summary.to_csv(args.output_dir / "e41_combined_summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "betas": betas,
                "family_margin": float(args.family_margin),
                "family_risk_limit": float(args.family_risk_limit),
                "beta_risk_limit": float(args.beta_risk_limit),
                "regimes": [
                    {
                        **config,
                        "cache_dir": str(config["cache_dir"]),
                        "e33_dir": str(config["e33_dir"]),
                    }
                    for config in REGIMES
                ],
                "selection_counts": combined_selection.groupby("regime_label")[
                    ["selected_family", "selected_beta"]
                ].value_counts().reset_index(name="n").to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "summary": str(args.output_dir / "e41_combined_summary.csv"),
                "selection": str(args.output_dir / "e41_combined_selection_records.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

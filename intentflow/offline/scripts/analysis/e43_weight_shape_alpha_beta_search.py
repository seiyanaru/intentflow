"""E43 reliability-weight shape / alpha / beta search.

This is a method-exploration script for the current strongest direction:

    reliability-weighted Ridge + source-margin-calibrated interpolation.

It deliberately does *not* build another local selector over the old expert
set.  Instead it searches the internals of the weighted classifier:

* reliability family: source / longitudinal
* rank-weight shape: floor and gamma
* Ridge alpha
* source-anchor interpolation beta

The script writes both descriptive target-side grid results and a nested
source-validation selection audit where each held target subject is excluded
from the validation pool.  The latter is the deployable-style check; the former
is only for hypothesis generation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

from weighted_ridge_regime_transfer_e32 import (
    RESULTS_DIR,
    aggregate_stats,
    balanced_subset,
    bootstrap_ci,
    load_payload,
    longitudinal_score,
    parse_subjects,
    percentile_rank,
    source_score,
)


LEE_CACHE = (
    RESULTS_DIR
    / "260629_lee2019_channel_dimension_ablation_e20"
    / "sensorimotor20"
    / "subject_cache"
)
BNCI_CACHE = RESULTS_DIR / "260628_bnci2014_001_fraction_sweep_e18" / "subject_cache"
DEFAULT_OUTPUT = RESULTS_DIR / "260702_e43_weight_shape_alpha_beta_search"

HARM_THRESHOLD_PP = -5.0


REGIMES = [
    {
        "regime_label": "lee2019_sensorimotor20",
        "cache_dir": LEE_CACHE,
        "subjects": "1-54",
        "per_class": 0,
        "repeats": 1,
    },
    {
        "regime_label": "bnci2014_001_full_source",
        "cache_dir": BNCI_CACHE,
        "subjects": "1-9",
        "per_class": 0,
        "repeats": 1,
    },
    {
        "regime_label": "bnci2014_001_m8_per_class",
        "cache_dir": BNCI_CACHE,
        "subjects": "1-9",
        "per_class": 8,
        "repeats": 32,
    },
]


@dataclass(frozen=True)
class WeightSpec:
    family: str
    transform: str
    gamma: float
    floor: float

    @property
    def id(self) -> str:
        return (
            f"{self.family}_{self.transform}"
            f"_g{float_token(self.gamma)}_f{float_token(self.floor)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--families", nargs="*", default=["source", "longitudinal"])
    parser.add_argument("--gammas", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    parser.add_argument("--floors", nargs="*", type=float, default=[0.02, 0.05, 0.10, 0.20])
    parser.add_argument("--alphas", nargs="*", type=float, default=[10.0, 30.0, 100.0, 300.0, 1000.0])
    parser.add_argument("--betas", nargs="*", type=float, default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--transforms", nargs="*", default=["rank"])
    parser.add_argument("--risk-constraint", type=float, default=0.08)
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def float_token(value: float) -> str:
    text = f"{float(value):.3g}".replace("-", "m").replace(".", "p")
    return text


def beta_token(beta: float) -> str:
    return f"b{float(beta):.2f}".replace(".", "p")


def alpha_token(alpha: float) -> str:
    return f"a{float(alpha):.0f}" if float(alpha).is_integer() else f"a{float_token(alpha)}"


def cvar_lower_10(values: Iterable[float]) -> float:
    arr = np.sort(np.asarray(list(values), dtype=np.float64))
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.10 * arr.size)))
    return float(arr[:k].mean())


def source_indices(labels: np.ndarray, *, subject: int, repeat: int, per_class: int, seed: int) -> np.ndarray:
    if int(per_class) <= 0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1000003 * int(subject) + int(repeat))
    return balanced_subset(labels, int(per_class), rng)


def rank_weight(score: np.ndarray, *, gamma: float, floor: float) -> np.ndarray:
    rank = percentile_rank(np.asarray(score, dtype=np.float64))
    rank = np.clip(rank, float(floor), 1.0)
    return rank ** float(gamma)


def sigmoid_weight(score: np.ndarray, *, gamma: float, floor: float) -> np.ndarray:
    """Optional smooth alternative.

    gamma is used as inverse temperature.  This is not part of the default
    heavy run, but keeping it here makes E43-B a one-flag extension.
    """

    values = np.asarray(score, dtype=np.float64)
    center = float(np.nanmedian(values))
    scale = float(np.nanpercentile(values, 75) - np.nanpercentile(values, 25))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.ones_like(values)
    z = (values - center) / scale
    inv_temp = float(gamma)
    sig = 1.0 / (1.0 + np.exp(-np.clip(inv_temp * z, -40.0, 40.0)))
    return float(floor) + (1.0 - float(floor)) * sig


def make_weights(score: np.ndarray, spec: WeightSpec) -> np.ndarray:
    if spec.transform == "rank":
        return rank_weight(score, gamma=spec.gamma, floor=spec.floor)
    if spec.transform == "sigmoid":
        return sigmoid_weight(score, gamma=spec.gamma, floor=spec.floor)
    raise ValueError(f"unknown transform: {spec.transform}")


def fit_binary_lda_margins(
    xs: np.ndarray,
    ys: np.ndarray,
    xt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(xs, ys)
    classes = np.asarray(clf.classes_)
    if len(classes) != 2:
        raise ValueError(f"E43 expects binary labels, got {classes}")
    return (
        np.asarray(clf.decision_function(xs), dtype=np.float64),
        np.asarray(clf.decision_function(xt), dtype=np.float64),
        classes,
    )


def fit_binary_ridge_margins(
    xs_scaled: np.ndarray,
    ys: np.ndarray,
    xt_scaled: np.ndarray,
    *,
    weights: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xsw = xs_scaled * weights[None, :]
    xtw = xt_scaled * weights[None, :]
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(xsw, ys)
    classes = np.asarray(clf.classes_)
    if len(classes) != 2:
        raise ValueError(f"E43 expects binary labels, got {classes}")
    return (
        np.asarray(clf.decision_function(xsw), dtype=np.float64),
        np.asarray(clf.decision_function(xtw), dtype=np.float64),
        classes,
    )


def scale_by_source_margin(source_margin: np.ndarray, target_margin: np.ndarray) -> np.ndarray:
    scale = float(np.std(np.asarray(source_margin, dtype=np.float64)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.asarray(target_margin, dtype=np.float64) / scale


def accuracy_from_margin(margin: np.ndarray, classes: np.ndarray, labels: np.ndarray) -> float:
    pred = np.where(np.asarray(margin) >= 0.0, classes[1], classes[0])
    return float(np.mean(pred == labels) * 100.0)


def candidate_id(spec: WeightSpec, alpha: float, beta: float) -> str:
    return f"wr_{spec.id}_{alpha_token(alpha)}_{beta_token(beta)}"


def build_weight_specs(args: argparse.Namespace) -> list[WeightSpec]:
    specs: list[WeightSpec] = []
    for family in args.families:
        if family not in {"source", "longitudinal"}:
            raise ValueError(f"unknown family: {family}")
        for transform in args.transforms:
            for gamma in args.gammas:
                for floor in args.floors:
                    specs.append(
                        WeightSpec(
                            family=str(family),
                            transform=str(transform),
                            gamma=float(gamma),
                            floor=float(floor),
                        )
                    )
    return specs


def process_regime(
    config: Mapping[str, object],
    *,
    weight_specs: Sequence[WeightSpec],
    alphas: Sequence[float],
    betas: Sequence[float],
    seed: int,
    quiet: bool,
) -> pd.DataFrame:
    regime = str(config["regime_label"])
    cache_dir = Path(config["cache_dir"])
    subjects = parse_subjects(str(config["subjects"]))
    per_class = int(config["per_class"])
    repeats = int(config["repeats"]) if per_class > 0 else 1
    payloads = {subject: load_payload(cache_dir, subject) for subject in subjects}

    rows: list[dict[str, object]] = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        stats = aggregate_stats(payloads, held_subject)
        scores = {
            "source": source_score(stats),
            "longitudinal": longitudinal_score(stats),
        }
        weight_lookup = {
            spec: make_weights(scores[spec.family], spec).astype(np.float64)
            for spec in weight_specs
        }

        labels = np.asarray(payload["source_labels"], dtype=np.int64)
        source_features = np.asarray(payload["source_features"], dtype=np.float64)
        target_features = np.asarray(payload["target_features"], dtype=np.float64)
        target_labels = np.asarray(payload["target_labels"], dtype=np.int64)

        for repeat in range(repeats):
            idx = source_indices(labels, subject=held_subject, repeat=repeat, per_class=per_class, seed=seed)
            xs = source_features[idx]
            ys = labels[idx]
            xt = target_features
            yt = target_labels

            lda_source_margin, lda_target_margin, lda_classes = fit_binary_lda_margins(xs, ys, xt)
            lda_margin = scale_by_source_margin(lda_source_margin, lda_target_margin)
            lda_acc = accuracy_from_margin(lda_margin, lda_classes, yt)
            rows.append(
                {
                    "regime_label": regime,
                    "subject": int(held_subject),
                    "repeat": int(repeat),
                    "source_per_class": per_class,
                    "source_total": int(len(idx)),
                    "candidate_id": "lda_full",
                    "method_family": "lda",
                    "weight_family": "none",
                    "transform": "none",
                    "gamma": np.nan,
                    "floor": np.nan,
                    "alpha": np.nan,
                    "beta": 0.0,
                    "accuracy": lda_acc,
                }
            )

            scaler = StandardScaler()
            xs_scaled = scaler.fit_transform(xs)
            xt_scaled = scaler.transform(xt)
            for spec in weight_specs:
                weights = weight_lookup[spec]
                for alpha in alphas:
                    ridge_source_margin, ridge_target_margin, ridge_classes = fit_binary_ridge_margins(
                        xs_scaled,
                        ys,
                        xt_scaled,
                        weights=weights,
                        alpha=float(alpha),
                    )
                    if not np.array_equal(lda_classes, ridge_classes):
                        raise ValueError((regime, held_subject, repeat, lda_classes, ridge_classes))
                    ridge_margin = scale_by_source_margin(ridge_source_margin, ridge_target_margin)
                    for beta in betas:
                        beta = float(beta)
                        margin = (1.0 - beta) * lda_margin + beta * ridge_margin
                        rows.append(
                            {
                                "regime_label": regime,
                                "subject": int(held_subject),
                                "repeat": int(repeat),
                                "source_per_class": per_class,
                                "source_total": int(len(idx)),
                                "candidate_id": candidate_id(spec, float(alpha), beta),
                                "method_family": "weighted_ridge_interp",
                                "weight_family": spec.family,
                                "transform": spec.transform,
                                "gamma": float(spec.gamma),
                                "floor": float(spec.floor),
                                "alpha": float(alpha),
                                "beta": beta,
                                "accuracy": accuracy_from_margin(margin, lda_classes, yt),
                            }
                        )
        if not quiet:
            print(f"{regime} S{held_subject}: done", flush=True)

    return pd.DataFrame(rows)


def add_gain(records: pd.DataFrame) -> pd.DataFrame:
    baseline = records[records["candidate_id"] == "lda_full"].copy()
    baseline = baseline[["regime_label", "subject", "repeat", "accuracy"]].rename(
        columns={"accuracy": "lda_full_accuracy"}
    )
    merged = records.merge(baseline, on=["regime_label", "subject", "repeat"], how="left")
    merged["gain_vs_lda_full_pp"] = merged["accuracy"] - merged["lda_full_accuracy"]
    return merged


def summarize(records: pd.DataFrame, *, level: str, bootstrap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    group_cols = [
        "regime_label",
        "candidate_id",
        "method_family",
        "weight_family",
        "transform",
        "gamma",
        "floor",
        "alpha",
        "beta",
    ]
    for keys, sub in records.groupby(group_cols, sort=True, dropna=False):
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
        (
            regime,
            candidate,
            method_family,
            weight_family,
            transform,
            gamma,
            floor,
            alpha,
            beta,
        ) = keys
        rows.append(
            {
                "regime_label": regime,
                "summary_level": level,
                "candidate_id": candidate,
                "method_family": method_family,
                "weight_family": weight_family,
                "transform": transform,
                "gamma": gamma,
                "floor": floor,
                "alpha": alpha,
                "beta": beta,
                "n_units": int(n_units),
                "accuracy_mean": float(np.mean(acc)),
                "accuracy_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(np.mean(gain)),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "gain_vs_lda_full_cvar10_pp": cvar_lower_10(gain),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_vs_lda_full_lt_0": float(np.mean(gain < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def source_selected_records(
    records: pd.DataFrame,
    *,
    risk_constraint: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.Series] = []
    choice_rows: list[dict[str, object]] = []
    candidates = sorted(records["candidate_id"].unique().tolist())
    for regime, regime_df in records.groupby("regime_label", sort=True):
        for unit in regime_df[["subject", "repeat"]].drop_duplicates().itertuples(index=False):
            subject = int(unit.subject)
            repeat = int(unit.repeat)
            inner = regime_df[regime_df["subject"] != subject].copy()
            stats = (
                inner.groupby("candidate_id", sort=True)["gain_vs_lda_full_pp"]
                .agg(
                    inner_mean_gain_pp="mean",
                    inner_p_harm=lambda x: float(np.mean(np.asarray(x, dtype=np.float64) < HARM_THRESHOLD_PP)),
                    inner_p_neg=lambda x: float(np.mean(np.asarray(x, dtype=np.float64) < 0.0)),
                    inner_q05=lambda x: float(np.quantile(np.asarray(x, dtype=np.float64), 0.05)),
                    inner_n="size",
                )
                .reset_index()
            )
            # Ensure LDA is always available as a safe fallback.
            if "lda_full" not in set(stats["candidate_id"]):
                stats = pd.concat(
                    [
                        stats,
                        pd.DataFrame(
                            [
                                {
                                    "candidate_id": "lda_full",
                                    "inner_mean_gain_pp": 0.0,
                                    "inner_p_harm": 0.0,
                                    "inner_p_neg": 0.0,
                                    "inner_q05": 0.0,
                                    "inner_n": len(inner),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            eligible = stats[stats["inner_p_harm"] <= float(risk_constraint)].copy()
            if len(eligible) == 0:
                chosen = "lda_full"
                chosen_stats = {
                    "inner_mean_gain_pp": 0.0,
                    "inner_p_harm": 0.0,
                    "inner_p_neg": 0.0,
                    "inner_q05": 0.0,
                    "inner_n": len(inner),
                }
            else:
                chosen_row = eligible.sort_values(
                    ["inner_mean_gain_pp", "inner_q05", "candidate_id"],
                    ascending=[False, False, True],
                ).iloc[0]
                chosen = str(chosen_row["candidate_id"])
                chosen_stats = chosen_row.to_dict()
            outer = regime_df[
                (regime_df["subject"] == subject)
                & (regime_df["repeat"] == repeat)
                & (regime_df["candidate_id"] == chosen)
            ]
            if len(outer) != 1:
                raise ValueError((regime, subject, repeat, chosen, len(outer), len(candidates)))
            row = outer.iloc[0].copy()
            row["candidate_id_original"] = row["candidate_id"]
            row["method_family_original"] = row["method_family"]
            row["weight_family_original"] = row["weight_family"]
            row["transform_original"] = row["transform"]
            row["gamma_original"] = row["gamma"]
            row["floor_original"] = row["floor"]
            row["alpha_original"] = row["alpha"]
            row["beta_original"] = row["beta"]
            row["candidate_id"] = f"source_selected_risk{float_token(risk_constraint)}"
            row["method_family"] = "source_selected"
            row["weight_family"] = "mixed"
            row["transform"] = "nested_exclude_subject"
            row["gamma"] = np.nan
            row["floor"] = np.nan
            row["alpha"] = np.nan
            row["beta"] = np.nan
            selected_rows.append(row)
            choice_rows.append(
                {
                    "regime_label": regime,
                    "subject": subject,
                    "repeat": repeat,
                    "chosen_candidate_id": chosen,
                    "risk_constraint": float(risk_constraint),
                    "inner_mean_gain_pp": float(chosen_stats["inner_mean_gain_pp"]),
                    "inner_p_harm": float(chosen_stats["inner_p_harm"]),
                    "inner_p_neg": float(chosen_stats["inner_p_neg"]),
                    "inner_q05": float(chosen_stats["inner_q05"]),
                    "inner_n": int(chosen_stats["inner_n"]),
                }
            )
    selected = pd.DataFrame(selected_rows)
    choices = pd.DataFrame(choice_rows)
    return selected, choices


def top_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[
        (summary["summary_level"] == "unit")
        & (summary["method_family"] != "source_selected")
    ].copy()
    candidates["risk_utility"] = (
        candidates["gain_vs_lda_full_mean_pp"]
        - 10.0 * candidates["p_gain_vs_lda_full_lt_minus5"]
        - 2.0 * candidates["p_gain_vs_lda_full_lt_0"]
    )
    frames = []
    for regime, sub in candidates.groupby("regime_label", sort=True):
        for label, part in [
            ("best_mean", sub.sort_values("gain_vs_lda_full_mean_pp", ascending=False).head(20)),
            ("best_risk_utility", sub.sort_values("risk_utility", ascending=False).head(20)),
            (
                "best_mean_under_p_harm_08",
                sub[sub["p_gain_vs_lda_full_lt_minus5"] <= 0.08]
                .sort_values("gain_vs_lda_full_mean_pp", ascending=False)
                .head(20),
            ),
        ]:
            part = part.copy()
            part.insert(1, "ranking", label)
            frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)


def benchmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    current = {
        "lee2019_sensorimotor20": {"e42_gain": 2.500, "e42_p_harm": 4.0 / 54.0},
        "bnci2014_001_full_source": {"e42_gain": 0.000, "e42_p_harm": 0.000},
        "bnci2014_001_m8_per_class": {"e42_gain": 2.470, "e42_p_harm": 18.0 / 288.0},
    }
    rows: list[dict[str, object]] = []
    unit = summary[summary["summary_level"] == "unit"].copy()
    for regime, values in current.items():
        sub = unit[
            (unit["regime_label"] == regime)
            & (unit["method_family"] != "source_selected")
        ].copy()
        if len(sub) == 0:
            continue
        selected = unit[
            (unit["regime_label"] == regime)
            & (unit["method_family"] == "source_selected")
        ].copy()
        selected_row = selected.iloc[0] if len(selected) else None
        best_mean = sub.sort_values("gain_vs_lda_full_mean_pp", ascending=False).iloc[0]
        eligible = sub[sub["p_gain_vs_lda_full_lt_minus5"] <= values["e42_p_harm"] + 1e-12]
        best_under_risk = (
            eligible.sort_values("gain_vs_lda_full_mean_pp", ascending=False).iloc[0]
            if len(eligible)
            else None
        )
        rows.append(
            {
                "regime_label": regime,
                "e42_gain_pp": values["e42_gain"],
                "e42_p_harm": values["e42_p_harm"],
                "best_mean_candidate": best_mean["candidate_id"],
                "best_mean_gain_pp": float(best_mean["gain_vs_lda_full_mean_pp"]),
                "best_mean_p_harm": float(best_mean["p_gain_vs_lda_full_lt_minus5"]),
                "best_mean_delta_gain_vs_e42_pp": float(best_mean["gain_vs_lda_full_mean_pp"] - values["e42_gain"]),
                "best_under_e42_risk_candidate": None if best_under_risk is None else best_under_risk["candidate_id"],
                "best_under_e42_risk_gain_pp": np.nan
                if best_under_risk is None
                else float(best_under_risk["gain_vs_lda_full_mean_pp"]),
                "best_under_e42_risk_p_harm": np.nan
                if best_under_risk is None
                else float(best_under_risk["p_gain_vs_lda_full_lt_minus5"]),
                "best_under_e42_risk_delta_gain_vs_e42_pp": np.nan
                if best_under_risk is None
                else float(best_under_risk["gain_vs_lda_full_mean_pp"] - values["e42_gain"]),
                "source_selected_candidate": None if selected_row is None else selected_row["candidate_id"],
                "source_selected_gain_pp": np.nan
                if selected_row is None
                else float(selected_row["gain_vs_lda_full_mean_pp"]),
                "source_selected_p_harm": np.nan
                if selected_row is None
                else float(selected_row["p_gain_vs_lda_full_lt_minus5"]),
                "source_selected_delta_gain_vs_e42_pp": np.nan
                if selected_row is None
                else float(selected_row["gain_vs_lda_full_mean_pp"] - values["e42_gain"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weight_specs = build_weight_specs(args)
    alphas = sorted(dict.fromkeys(float(x) for x in args.alphas))
    betas = sorted(dict.fromkeys(float(x) for x in args.betas))

    frames = []
    for config in REGIMES:
        frames.append(
            process_regime(
                config,
                weight_specs=weight_specs,
                alphas=alphas,
                betas=betas,
                seed=int(args.seed),
                quiet=bool(args.quiet),
            )
        )
    records = add_gain(pd.concat(frames, ignore_index=True, sort=False))
    selected, choices = source_selected_records(records, risk_constraint=float(args.risk_constraint))
    combined_for_summary = pd.concat([records, selected], ignore_index=True, sort=False)
    summary = pd.concat(
        [
            summarize(combined_for_summary, level="subject", bootstrap=int(args.bootstrap), seed=int(args.seed)),
            summarize(combined_for_summary, level="unit", bootstrap=int(args.bootstrap), seed=int(args.seed)),
        ],
        ignore_index=True,
        sort=False,
    )
    top = top_candidates(summary)
    benchmark = benchmark_table(summary)
    choice_summary = (
        choices.groupby(["regime_label", "chosen_candidate_id"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["regime_label", "count"], ascending=[True, False])
    )

    records.to_csv(args.output_dir / "e43_grid_records.csv", index=False)
    selected.to_csv(args.output_dir / "e43_source_selected_records.csv", index=False)
    choices.to_csv(args.output_dir / "e43_source_selection_choices.csv", index=False)
    choice_summary.to_csv(args.output_dir / "e43_source_selection_choice_summary.csv", index=False)
    summary.to_csv(args.output_dir / "e43_summary.csv", index=False)
    top.to_csv(args.output_dir / "e43_top_candidates.csv", index=False)
    benchmark.to_csv(args.output_dir / "e43_benchmark_vs_e42.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "weight_specs": [spec.__dict__ for spec in weight_specs],
                "alphas": alphas,
                "betas": betas,
                "risk_constraint": float(args.risk_constraint),
                "regimes": REGIMES,
                "outputs": {
                    "records": str(args.output_dir / "e43_grid_records.csv"),
                    "selected": str(args.output_dir / "e43_source_selected_records.csv"),
                    "choices": str(args.output_dir / "e43_source_selection_choices.csv"),
                    "summary": str(args.output_dir / "e43_summary.csv"),
                    "top": str(args.output_dir / "e43_top_candidates.csv"),
                    "benchmark": str(args.output_dir / "e43_benchmark_vs_e42.csv"),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "n_grid_records": int(len(records)),
                "n_selected_records": int(len(selected)),
                "summary": str(args.output_dir / "e43_summary.csv"),
                "top": str(args.output_dir / "e43_top_candidates.csv"),
                "benchmark": str(args.output_dir / "e43_benchmark_vs_e42.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

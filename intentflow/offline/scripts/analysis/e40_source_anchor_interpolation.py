"""E40 source-anchored score interpolation.

E39 froze the practical regime-level family rule.  The remaining weakness is
that the decision is still mostly binary: use LDA or use the weighted Ridge
family.  E40 tests a continuous source-anchor:

    margin(beta) = (1 - beta) * z_margin_LDA + beta * z_margin_weighted_Ridge

where each margin is divided by its source-training standard deviation.  This
keeps beta=0 identical to LDA and beta=1 identical to the weighted Ridge
classifier, while making intermediate beta values scale-comparable without
using target labels.

Two variants are evaluated:

* global_interpolation: use the regime-level weighted family for every unit.
* guarded_interpolation: use the same family only when the E38 local LDA
  fallback guard passes; otherwise beta is forced to 0.
"""

from __future__ import annotations

import argparse
import json
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
    rank_weights,
    source_score,
)


LEE_CACHE = (
    RESULTS_DIR
    / "260629_lee2019_channel_dimension_ablation_e20"
    / "sensorimotor20"
    / "subject_cache"
)
BNCI_CACHE = RESULTS_DIR / "260628_bnci2014_001_fraction_sweep_e18" / "subject_cache"
DEFAULT_E38_SELECTIONS = (
    RESULTS_DIR / "260701_e38_regime_level_family_rule" / "e38_combined_selection_records.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "260701_e40_source_anchor_interpolation"

HARM_THRESHOLD_PP = -5.0


REGIMES = [
    {
        "regime_label": "lee2019_sensorimotor20",
        "cache_dir": LEE_CACHE,
        "subjects": "1-54",
        "per_class": 0,
        "repeats": 1,
        "family": "source",
    },
    {
        "regime_label": "bnci2014_001_full_source",
        "cache_dir": BNCI_CACHE,
        "subjects": "1-9",
        "per_class": 0,
        "repeats": 1,
        "family": "lda",
    },
    {
        "regime_label": "bnci2014_001_m8_per_class",
        "cache_dir": BNCI_CACHE,
        "subjects": "1-9",
        "per_class": 8,
        "repeats": 32,
        "family": "longitudinal",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--e38-selections", type=Path, default=DEFAULT_E38_SELECTIONS)
    parser.add_argument("--betas", nargs="*", type=float, default=[i / 10 for i in range(11)])
    parser.add_argument("--bootstrap", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def cvar_lower_10(values: np.ndarray) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(0.10 * arr.size)))
    return float(arr[:k].mean())


def beta_token(beta: float) -> str:
    return f"b{float(beta):.2f}".replace(".", "p")


def source_indices(labels: np.ndarray, *, subject: int, repeat: int, per_class: int, seed: int) -> np.ndarray:
    if int(per_class) <= 0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed) + 1000003 * int(subject) + int(repeat))
    return balanced_subset(labels, int(per_class), rng)


def fit_binary_lda_margins(
    xs: np.ndarray,
    ys: np.ndarray,
    xt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(xs, ys)
    classes = np.asarray(clf.classes_)
    if len(classes) != 2:
        raise ValueError(f"E40 currently expects binary labels, got {classes}")
    source_margin = np.asarray(clf.decision_function(xs), dtype=np.float64)
    target_margin = np.asarray(clf.decision_function(xt), dtype=np.float64)
    return source_margin, target_margin, classes


def fit_binary_ridge_margins(
    xs: np.ndarray,
    ys: np.ndarray,
    xt: np.ndarray,
    *,
    weights: np.ndarray,
    alpha: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    xs_scaled = scaler.fit_transform(xs)
    xt_scaled = scaler.transform(xt)
    xs_scaled = xs_scaled * weights[None, :]
    xt_scaled = xt_scaled * weights[None, :]
    clf = RidgeClassifier(alpha=float(alpha))
    clf.fit(xs_scaled, ys)
    classes = np.asarray(clf.classes_)
    if len(classes) != 2:
        raise ValueError(f"E40 currently expects binary labels, got {classes}")
    source_margin = np.asarray(clf.decision_function(xs_scaled), dtype=np.float64)
    target_margin = np.asarray(clf.decision_function(xt_scaled), dtype=np.float64)
    return source_margin, target_margin, classes


def scale_by_source_margin(source_margin: np.ndarray, target_margin: np.ndarray) -> np.ndarray:
    scale = float(np.std(np.asarray(source_margin, dtype=np.float64)))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return np.asarray(target_margin, dtype=np.float64) / scale


def predict_from_margin(margin: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(margin) >= 0.0, classes[1], classes[0])


def accuracy_from_margin(margin: np.ndarray, classes: np.ndarray, labels: np.ndarray) -> float:
    pred = predict_from_margin(margin, classes)
    return float(np.mean(pred == labels) * 100.0)


def guard_lookup(selection_path: Path) -> dict[tuple[str, int, int], bool]:
    if not selection_path.exists():
        return {}
    df = pd.read_csv(selection_path)
    df = df[df["mode"] == "repeat"].copy()
    lookup = {}
    for row in df.itertuples():
        lookup[(str(row.regime_label), int(row.subject), int(row.repeat))] = bool(
            row.pass_margin and row.pass_risk
        )
    return lookup


def weights_for_family(
    payloads: Mapping[int, Mapping[str, np.ndarray]],
    held_subject: int,
    family: str,
) -> np.ndarray | None:
    if family == "lda":
        return None
    stats = aggregate_stats(payloads, held_subject)
    if family == "source":
        return rank_weights(source_score(stats), gamma=2.0)
    if family == "longitudinal":
        return rank_weights(longitudinal_score(stats), gamma=2.0)
    raise ValueError(family)


def process_regime(
    config: Mapping[str, object],
    *,
    betas: Sequence[float],
    guard: Mapping[tuple[str, int, int], bool],
    seed: int,
    quiet: bool,
) -> pd.DataFrame:
    regime = str(config["regime_label"])
    cache_dir = Path(config["cache_dir"])
    subjects = parse_subjects(str(config["subjects"]))
    per_class = int(config["per_class"])
    repeats = int(config["repeats"]) if per_class > 0 else 1
    family = str(config["family"])
    payloads = {subject: load_payload(cache_dir, subject) for subject in subjects}

    rows = []
    for held_subject in subjects:
        payload = payloads[held_subject]
        weights = weights_for_family(payloads, held_subject, family)
        labels = np.asarray(payload["source_labels"], dtype=np.int64)
        xt = np.asarray(payload["target_features"], dtype=np.float64)
        yt = np.asarray(payload["target_labels"], dtype=np.int64)

        for repeat in range(repeats):
            idx = source_indices(labels, subject=held_subject, repeat=repeat, per_class=per_class, seed=seed)
            xs = np.asarray(payload["source_features"], dtype=np.float64)[idx]
            ys = np.asarray(payload["source_labels"], dtype=np.int64)[idx]

            lda_source_margin, lda_target_margin, lda_classes = fit_binary_lda_margins(xs, ys, xt)
            lda_margin = scale_by_source_margin(lda_source_margin, lda_target_margin)
            if weights is None:
                ridge_margin = lda_margin.copy()
                ridge_classes = lda_classes.copy()
            else:
                ridge_source_margin, ridge_target_margin, ridge_classes = fit_binary_ridge_margins(
                    xs,
                    ys,
                    xt,
                    weights=np.asarray(weights, dtype=np.float64),
                )
                ridge_margin = scale_by_source_margin(ridge_source_margin, ridge_target_margin)
            if not np.array_equal(lda_classes, ridge_classes):
                raise ValueError((regime, held_subject, repeat, lda_classes, ridge_classes))

            guard_pass = bool(guard.get((regime, int(held_subject), int(repeat)), True))
            for beta in betas:
                beta = float(beta)
                global_margin = (1.0 - beta) * lda_margin + beta * ridge_margin
                guarded_beta = beta if guard_pass else 0.0
                guarded_margin = (1.0 - guarded_beta) * lda_margin + guarded_beta * ridge_margin
                for variant, effective_beta, margin in [
                    ("global_interpolation", beta, global_margin),
                    ("guarded_interpolation", guarded_beta, guarded_margin),
                ]:
                    rows.append(
                        {
                            "regime_label": regime,
                            "subject": int(held_subject),
                            "repeat": int(repeat),
                            "source_per_class": per_class,
                            "source_total": int(len(idx)),
                            "family": family,
                            "variant": variant,
                            "beta": beta,
                            "effective_beta": float(effective_beta),
                            "guard_pass": guard_pass,
                            "method": f"e40_{variant}_{beta_token(beta)}",
                            "accuracy": accuracy_from_margin(margin, lda_classes, yt),
                        }
                    )
            # Explicit anchors for easy comparison.
            rows.append(
                {
                    "regime_label": regime,
                    "subject": int(held_subject),
                    "repeat": int(repeat),
                    "source_per_class": per_class,
                    "source_total": int(len(idx)),
                    "family": family,
                    "variant": "anchor",
                    "beta": 0.0,
                    "effective_beta": 0.0,
                    "guard_pass": guard_pass,
                    "method": "lda_full",
                    "accuracy": accuracy_from_margin(lda_margin, lda_classes, yt),
                }
            )
            rows.append(
                {
                    "regime_label": regime,
                    "subject": int(held_subject),
                    "repeat": int(repeat),
                    "source_per_class": per_class,
                    "source_total": int(len(idx)),
                    "family": family,
                    "variant": "anchor",
                    "beta": 1.0,
                    "effective_beta": 1.0 if weights is not None else 0.0,
                    "guard_pass": guard_pass,
                    "method": f"ridge_{family}_rank_g2_a100" if weights is not None else "lda_full_anchor_duplicate",
                    "accuracy": accuracy_from_margin(ridge_margin, lda_classes, yt),
                }
            )
        if not quiet:
            print(f"{regime} S{held_subject}: done", flush=True)
    return pd.DataFrame(rows)


def add_gain(records: pd.DataFrame) -> pd.DataFrame:
    baseline = records[records["method"] == "lda_full"].copy()
    baseline = baseline[["regime_label", "subject", "repeat", "accuracy"]].rename(
        columns={"accuracy": "lda_full_accuracy"}
    )
    merged = records.merge(baseline, on=["regime_label", "subject", "repeat"], how="left")
    merged["gain_vs_lda_full_pp"] = merged["accuracy"] - merged["lda_full_accuracy"]
    return merged


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
                "variant": str(first["variant"]),
                "family": str(first["family"]),
                "beta": float(first["beta"]),
                "mean_effective_beta": float(sub["effective_beta"].mean()),
                "guard_pass_rate": float(sub["guard_pass"].mean()),
                "n_units": int(n_units),
                "accuracy_mean": float(acc.mean()),
                "accuracy_95ci": bootstrap_ci(acc, rng, bootstrap),
                "gain_vs_lda_full_mean_pp": float(gain.mean()),
                "gain_vs_lda_full_95ci": bootstrap_ci(gain, rng, bootstrap),
                "gain_vs_lda_full_q05_pp": float(np.quantile(gain, 0.05)),
                "gain_vs_lda_full_cvar10_pp": cvar_lower_10(gain),
                "p_gain_vs_lda_full_lt_minus5": float(np.mean(gain < HARM_THRESHOLD_PP)),
                "p_gain_vs_lda_full_lt_0": float(np.mean(gain < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def paired_diff(
    records: pd.DataFrame,
    *,
    regime: str,
    level: str,
    method_a: str,
    method_b: str,
    bootstrap: int,
    seed: int,
) -> dict[str, object] | None:
    sub = records[records["regime_label"] == regime].copy()
    if level == "unit":
        pivot = sub.pivot_table(
            index=["subject", "repeat"], columns="method", values="gain_vs_lda_full_pp", aggfunc="mean"
        )
    elif level == "subject":
        pivot = (
            sub.groupby(["subject", "method"], sort=True)["gain_vs_lda_full_pp"]
            .mean()
            .reset_index()
            .pivot_table(index="subject", columns="method", values="gain_vs_lda_full_pp", aggfunc="mean")
        )
    else:
        raise ValueError(level)
    if method_a not in pivot.columns or method_b not in pivot.columns:
        return None
    diff = (pivot[method_a] - pivot[method_b]).dropna().to_numpy(dtype=np.float64)
    if diff.size == 0:
        return None
    rng = np.random.default_rng(seed)
    return {
        "regime_label": regime,
        "summary_level": level,
        "method_a": method_a,
        "method_b": method_b,
        "n_units": int(diff.size),
        "mean_diff_pp": float(diff.mean()),
        "diff_95ci": bootstrap_ci(diff, rng, bootstrap),
        "q05_diff_pp": float(np.quantile(diff, 0.05)),
        "cvar10_diff_pp": cvar_lower_10(diff),
        "p_diff_lt_minus5": float(np.mean(diff < HARM_THRESHOLD_PP)),
        "p_diff_lt_0": float(np.mean(diff < 0.0)),
        "p_diff_gt_0": float(np.mean(diff > 0.0)),
    }


def best_by_regime(summary: pd.DataFrame) -> pd.DataFrame:
    candidates = summary[
        (summary["summary_level"] == "unit")
        & (summary["variant"].isin(["global_interpolation", "guarded_interpolation"]))
    ].copy()
    # A compact risk-aware score used only to rank candidates for inspection.
    # Main claims should still use the separate mean/risk columns.
    candidates["risk_utility"] = (
        candidates["gain_vs_lda_full_mean_pp"]
        - 10.0 * candidates["p_gain_vs_lda_full_lt_minus5"]
        - 2.0 * candidates["p_gain_vs_lda_full_lt_0"]
    )
    return candidates.sort_values(["regime_label", "risk_utility"], ascending=[True, False]).groupby(
        "regime_label", sort=True
    ).head(5)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    betas = sorted(dict.fromkeys(float(beta) for beta in args.betas))
    guard = guard_lookup(args.e38_selections)

    frames = []
    for config in REGIMES:
        frames.append(
            process_regime(
                config,
                betas=betas,
                guard=guard,
                seed=int(args.seed),
                quiet=bool(args.quiet),
            )
        )
    records = add_gain(pd.concat(frames, ignore_index=True, sort=False))
    summary = pd.concat(
        [
            summarize(records, level="subject", bootstrap=int(args.bootstrap), seed=int(args.seed)),
            summarize(records, level="unit", bootstrap=int(args.bootstrap), seed=int(args.seed)),
        ],
        ignore_index=True,
    )

    contrast_rows = []
    for regime in sorted(records["regime_label"].unique()):
        for level in ["subject", "unit"]:
            for beta in betas:
                for variant in ["global_interpolation", "guarded_interpolation"]:
                    method = f"e40_{variant}_{beta_token(beta)}"
                    result = paired_diff(
                        records,
                        regime=regime,
                        level=level,
                        method_a=method,
                        method_b="lda_full",
                        bootstrap=int(args.bootstrap),
                        seed=int(args.seed),
                    )
                    if result is not None:
                        result["contrast"] = "vs_lda_full"
                        contrast_rows.append(result)
            # Compare the best coarse practical beta to beta=1 guarded/global.
            for variant in ["global_interpolation", "guarded_interpolation"]:
                candidates = summary[
                    (summary["regime_label"] == regime)
                    & (summary["summary_level"] == level)
                    & (summary["variant"] == variant)
                ].copy()
                if len(candidates) == 0:
                    continue
                candidates["risk_utility"] = (
                    candidates["gain_vs_lda_full_mean_pp"]
                    - 10.0 * candidates["p_gain_vs_lda_full_lt_minus5"]
                    - 2.0 * candidates["p_gain_vs_lda_full_lt_0"]
                )
                best = str(candidates.sort_values("risk_utility", ascending=False).iloc[0]["method"])
                beta1 = f"e40_{variant}_{beta_token(1.0)}"
                result = paired_diff(
                    records,
                    regime=regime,
                    level=level,
                    method_a=best,
                    method_b=beta1,
                    bootstrap=int(args.bootstrap),
                    seed=int(args.seed),
                )
                if result is not None:
                    result["contrast"] = f"risk_utility_best_{variant}_vs_beta1"
                    contrast_rows.append(result)
    contrasts = pd.DataFrame(contrast_rows)
    best = best_by_regime(summary)

    records.to_csv(args.output_dir / "e40_interpolation_records.csv", index=False)
    summary.to_csv(args.output_dir / "e40_interpolation_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "e40_interpolation_contrasts.csv", index=False)
    best.to_csv(args.output_dir / "e40_best_risk_utility_candidates.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "betas": betas,
                "regimes": REGIMES,
                "outputs": {
                    "records": str(args.output_dir / "e40_interpolation_records.csv"),
                    "summary": str(args.output_dir / "e40_interpolation_summary.csv"),
                    "contrasts": str(args.output_dir / "e40_interpolation_contrasts.csv"),
                    "best": str(args.output_dir / "e40_best_risk_utility_candidates.csv"),
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
                "records": str(args.output_dir / "e40_interpolation_records.csv"),
                "summary": str(args.output_dir / "e40_interpolation_summary.csv"),
                "best": str(args.output_dir / "e40_best_risk_utility_candidates.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

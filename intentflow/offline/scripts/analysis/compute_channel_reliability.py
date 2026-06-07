"""Compute per-channel reliability diagnostics for Reliability-Aware EA.

This script is intentionally diagnostic-first. It does not train a model and it
does not use labels. It compares the target/test session against the train
session after the repository's usual z-scaling, then estimates whether each
channel looks unreliable for alignment covariance estimation.

The output is meant to answer: "would a per-channel weighted EA lower the right
channels for S2/S5/S7/S8 before we spend GPU time on RAA training?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from utils.get_datamodule_cls import get_datamodule_cls  # noqa: E402


EPS = 1e-8


def subjects_from_arg(arg: str, all_subjects: Iterable[int]) -> list[int]:
    if arg == "all":
        return list(all_subjects)
    return [int(x) for x in arg.split(",") if x.strip()]


def dataset_x(dataset) -> np.ndarray:
    return dataset.tensors[0].detach().cpu().numpy()


def robust_z(values: np.ndarray, ref_values: np.ndarray) -> np.ndarray:
    mean = ref_values.mean(axis=0, keepdims=True)
    std = ref_values.std(axis=0, keepdims=True) + EPS
    return (values - mean) / std


def channel_log_variance(x: np.ndarray) -> np.ndarray:
    return np.log(np.var(x, axis=-1) + EPS)


def channel_abs_kurtosis(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=-1, keepdims=True)
    var = np.mean(centered * centered, axis=-1) + EPS
    fourth = np.mean(centered**4, axis=-1)
    excess = fourth / (var * var) - 3.0
    return np.abs(excess)


def channel_line_noise_ratio(x: np.ndarray, sfreq: float) -> np.ndarray:
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    total_mask = (freqs >= 1.0) & (freqs <= min(100.0, sfreq / 2.0 - 1.0))
    line_mask = ((freqs >= 48.0) & (freqs <= 52.0)) | (
        (freqs >= 58.0) & (freqs <= 62.0)
    )
    total = spec[..., total_mask].sum(axis=-1) + EPS
    line = spec[..., line_mask].sum(axis=-1) + EPS
    return np.log(line / total + EPS)


def channel_corr_with_reference(x: np.ndarray) -> np.ndarray:
    # Correlate each channel with the average of all other channels in the same
    # trial. This is cheap and catches channels that decorrelate from the montage.
    n, c, _ = x.shape
    out = np.zeros((n, c), dtype=np.float64)
    for ch in range(c):
        ref = (x.sum(axis=1) - x[:, ch]) / max(c - 1, 1)
        a = x[:, ch] - x[:, ch].mean(axis=-1, keepdims=True)
        b = ref - ref.mean(axis=-1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1)) + EPS
        out[:, ch] = (a * b).sum(axis=-1) / denom
    return out


def mean_cov(x: np.ndarray) -> np.ndarray:
    return np.matmul(x, np.swapaxes(x, 1, 2)).mean(axis=0) / x.shape[-1]


def condition_number(r: np.ndarray) -> float:
    evals = np.linalg.eigvalsh(r)
    evals = np.maximum(evals, 1e-12)
    return float(evals[-1] / evals[0])


def covariance_leverage(x: np.ndarray) -> np.ndarray:
    r = mean_cov(x)
    full = np.log(condition_number(r) + EPS)
    scores = []
    for ch in range(r.shape[0]):
        keep = [i for i in range(r.shape[0]) if i != ch]
        reduced = r[np.ix_(keep, keep)]
        reduced_cond = np.log(condition_number(reduced) + EPS)
        scores.append(max(0.0, full - reduced_cond))
    scores = np.asarray(scores, dtype=np.float64)
    if scores.max() > 0:
        scores = scores / scores.max() * 3.0
    return scores


def weights_from_badness(badness: np.ndarray, w_min: float, tau: float) -> np.ndarray:
    weights = np.exp(-badness / tau)
    return np.clip(weights, w_min, 1.0)


def relative_outlier(values: np.ndarray, margin: float) -> np.ndarray:
    # A global session shift should be handled by EA itself. Channel reliability
    # should mainly downweight channels that are worse than their peers.
    baseline = np.median(values)
    return np.clip(values - baseline - margin, 0.0, None)


def summarize_subject(x_train: np.ndarray, x_test: np.ndarray, sfreq: float, args) -> dict:
    train_log_var = channel_log_variance(x_train)
    test_log_var = channel_log_variance(x_test)
    train_abs_kurt = channel_abs_kurtosis(x_train)
    test_abs_kurt = channel_abs_kurtosis(x_test)
    train_line = channel_line_noise_ratio(x_train, sfreq)
    test_line = channel_line_noise_ratio(x_test, sfreq)
    train_corr = channel_corr_with_reference(x_train)
    test_corr = channel_corr_with_reference(x_test)

    var_z = robust_z(test_log_var, train_log_var)
    high_var = np.clip(var_z, 0.0, args.clip_z)
    flatline = np.clip(-var_z, 0.0, args.clip_z)
    kurt = np.clip(robust_z(test_abs_kurt, train_abs_kurt), 0.0, args.clip_z)
    line = np.clip(robust_z(test_line, train_line), 0.0, args.clip_z)
    corr = np.clip(np.abs(robust_z(test_corr, train_corr)), 0.0, args.clip_z)
    leverage = covariance_leverage(x_test)

    high_var_m = high_var.mean(axis=0)
    flatline_m = flatline.mean(axis=0)
    kurt_m = kurt.mean(axis=0)
    line_m = line.mean(axis=0)
    corr_m = corr.mean(axis=0)

    r_test = mean_cov(x_test)
    diag = np.diag(r_test)
    cov = {
        "condition": condition_number(r_test),
        "diag_cv": float(diag.std() / (diag.mean() + EPS)),
        "diag_ratio": float(diag.max() / (diag.min() + EPS)),
    }

    rel_high_var = relative_outlier(high_var_m, args.relative_margin)
    rel_flatline = relative_outlier(flatline_m, args.relative_margin)
    rel_kurt = relative_outlier(kurt_m, args.relative_margin)
    rel_line = relative_outlier(line_m, args.relative_margin)
    rel_corr = relative_outlier(corr_m, args.relative_margin)
    rel_leverage = relative_outlier(leverage, args.relative_margin)

    bad_var = rel_high_var + rel_flatline
    bad_artifact = bad_var + 0.5 * rel_kurt + 0.5 * rel_line
    bad_full = bad_artifact + 0.4 * rel_corr + 0.5 * rel_leverage

    weights = {
        "raa_var": weights_from_badness(bad_var, args.w_min, args.tau),
        "raa_artifact": weights_from_badness(bad_artifact, args.w_min, args.tau),
        "raa_full": weights_from_badness(bad_full, args.w_min, args.tau),
    }

    metrics = {
        "high_variance_z": high_var_m,
        "flatline_z": flatline_m,
        "kurtosis_z": kurt_m,
        "line_noise_z": line_m,
        "corr_abnormal_z": corr_m,
        "cov_leverage": leverage,
        "relative_high_variance": rel_high_var,
        "relative_flatline": rel_flatline,
        "relative_kurtosis": rel_kurt,
        "relative_line_noise": rel_line,
        "relative_corr_abnormal": rel_corr,
        "relative_cov_leverage": rel_leverage,
        "badness_var": bad_var,
        "badness_artifact": bad_artifact,
        "badness_full": bad_full,
    }

    top = {}
    for key, values in metrics.items():
        order = np.argsort(values)[::-1][: args.topk]
        top[key] = [
            {"channel": int(ch), "score": float(values[ch])}
            for ch in order
            if values[ch] > 0.0
        ]

    return {
        "covariance": cov,
        "metrics": {k: v.astype(float).round(6).tolist() for k, v in metrics.items()},
        "weights": {k: v.astype(float).round(6).tolist() for k, v in weights.items()},
        "weight_summary": {
            k: {
                "min": float(v.min()),
                "mean": float(v.mean()),
                "max": float(v.max()),
                "low_channels": [
                    int(ch) for ch in np.where(v <= args.low_weight_threshold)[0]
                ],
            }
            for k, v in weights.items()
        },
        "top": top,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="bcic2a")
    parser.add_argument("--subjects", default="2,5,7,8")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip_z", type=float, default=6.0)
    parser.add_argument("--w_min", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--relative_margin", type=float, default=0.25)
    parser.add_argument("--low_weight_threshold", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument(
        "--stress_mode",
        default=None,
        choices=[
            "none",
            "gaussian",
            "highvar",
            "flatline",
            "dropout",
            "line",
            "burst",
            "scale",
            "mixed",
        ],
    )
    parser.add_argument("--stress_channels", default=None)
    parser.add_argument("--stress_n_channels", type=int, default=None)
    parser.add_argument("--stress_level", type=float, default=None)
    parser.add_argument("--stress_seed", type=int, default=None)
    parser.add_argument("--stress_p_trials", type=float, default=None)
    parser.add_argument("--stress_line_freq", type=float, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    preprocessing_cfg = cfg["preprocessing"]
    if args.dataset in preprocessing_cfg:
        preprocessing = dict(preprocessing_cfg[args.dataset])
        preprocessing["z_scale"] = cfg.get("z_scale", preprocessing.get("z_scale", True))
        if args.dataset == "bcic2a":
            preprocessing["data_path"] = cfg.get("data_path", preprocessing.get("data_path"))
            preprocessing["eval_label_path"] = cfg.get(
                "data_path_2a_eval_labels", preprocessing.get("eval_label_path")
            )
    else:
        preprocessing = dict(preprocessing_cfg)
    preprocessing["ea"] = {"enabled": False}
    if args.stress_mode is not None:
        stress = {
            "enabled": args.stress_mode != "none",
            "mode": args.stress_mode,
        }
        if args.stress_channels is not None:
            stress["channels"] = args.stress_channels
        if args.stress_n_channels is not None:
            stress["n_channels"] = args.stress_n_channels
        if args.stress_level is not None:
            stress["level"] = args.stress_level
        if args.stress_seed is not None:
            stress["seed"] = args.stress_seed
        if args.stress_p_trials is not None:
            stress["p_trials"] = args.stress_p_trials
        if args.stress_line_freq is not None:
            stress["line_freq"] = args.stress_line_freq
        preprocessing["artifact_stress"] = stress
    sfreq = float(preprocessing.get("sfreq", 250))

    datamodule_cls = get_datamodule_cls(args.dataset)
    rows = []
    for sid in subjects_from_arg(args.subjects, datamodule_cls.all_subject_ids):
        dm = datamodule_cls(preprocessing, subject_id=sid)
        dm.setup("fit")
        x_train = dataset_x(dm.train_dataset)
        x_test = dataset_x(dm.test_dataset)
        row = summarize_subject(x_train, x_test, sfreq, args)
        row["subject"] = sid
        rows.append(row)
        full = row["weight_summary"]["raa_full"]
        cov = row["covariance"]
        print(
            f"S{sid}: full_w min={full['min']:.3f} mean={full['mean']:.3f} "
            f"low={full['low_channels']} cond={cov['condition']:.1f} "
            f"diag_cv={cov['diag_cv']:.3f}"
        )
        print(f"  top full badness: {row['top']['badness_full'][: args.topk]}")

    out = {
        "config": args.config,
        "dataset": args.dataset,
        "subjects": [row["subject"] for row in rows],
        "parameters": {
            "clip_z": args.clip_z,
            "w_min": args.w_min,
            "tau": args.tau,
            "relative_margin": args.relative_margin,
            "low_weight_threshold": args.low_weight_threshold,
        },
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

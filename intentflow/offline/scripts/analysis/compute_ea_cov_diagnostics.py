"""Compute label-free covariance diagnostics for EA/RAA selection.

The script loads train/test tensors with EA disabled but the usual z-scaling
enabled, then reports covariance statistics that are available at test time
without labels:

- train/test covariance condition number
- test channel variance CV and max/min ratio
- normalized train-test covariance distance

These features are intended to complement prediction-statistic based rejection
in Reliability-Aware Alignment.
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


def subjects_from_arg(arg: str, all_subjects: Iterable[int]) -> list[int]:
    if arg == "all":
        return list(all_subjects)
    return [int(x) for x in arg.split(",") if x.strip()]


def tensor_dataset_x(dataset) -> np.ndarray:
    x = dataset.tensors[0]
    return x.detach().cpu().numpy()


def mean_cov(x: np.ndarray) -> np.ndarray:
    return np.matmul(x, np.swapaxes(x, 1, 2)).mean(axis=0) / x.shape[-1]


def norm_cov(r: np.ndarray) -> np.ndarray:
    return r / (np.trace(r) / r.shape[0] + 1e-12)


def cov_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = norm_cov(a)
    b = norm_cov(b)
    return float(np.linalg.norm(a - b, "fro") / (np.linalg.norm(a, "fro") + 1e-12))


def eig_stats(r: np.ndarray) -> dict[str, float]:
    evals = np.linalg.eigvalsh(r)
    evals = np.maximum(evals, 1e-12)
    probs = evals / evals.sum()
    return {
        "condition": float(evals[-1] / evals[0]),
        "eig_entropy": float(-(probs * np.log(probs)).sum() / np.log(len(evals))),
        "eig_min": float(evals[0]),
        "eig_max": float(evals[-1]),
    }


def diag_stats(r: np.ndarray) -> dict[str, float]:
    diag = np.diag(r)
    return {
        "diag_cv": float(diag.std() / (diag.mean() + 1e-12)),
        "diag_ratio": float(diag.max() / (diag.min() + 1e-12)),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="bcic2a")
    parser.add_argument("--subjects", default="all")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    preprocessing = dict(cfg["preprocessing"])
    preprocessing["ea"] = {"enabled": False}

    datamodule_cls = get_datamodule_cls(args.dataset)
    rows = []
    for sid in subjects_from_arg(args.subjects, datamodule_cls.all_subject_ids):
        dm = datamodule_cls(preprocessing, subject_id=sid)
        dm.setup("fit")
        x_train = tensor_dataset_x(dm.train_dataset)
        x_test = tensor_dataset_x(dm.test_dataset)

        r_train = mean_cov(x_train)
        r_test = mean_cov(x_test)
        train_eig = eig_stats(r_train)
        test_eig = eig_stats(r_test)
        test_diag = diag_stats(r_test)
        row = {
            "subject": sid,
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "cov_distance": cov_distance(r_train, r_test),
            "train": train_eig,
            "test": {**test_eig, **test_diag},
        }
        rows.append(row)
        print(
            f"S{sid}: cov_dist={row['cov_distance']:.3f} "
            f"test_cond={test_eig['condition']:.1f} "
            f"diag_cv={test_diag['diag_cv']:.3f} "
            f"diag_ratio={test_diag['diag_ratio']:.2f}"
        )

    out = {
        "config": args.config,
        "dataset": args.dataset,
        "rows": rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

"""E11: compactness-frontier tables and figures from E9/E10 summaries.

This script turns the E9 fraction sweep into paper-facing artifacts:

* a tidy CSV with dataset/task/family/q/gain/risk values
* gain-vs-compactness figures for Stieger LR, Stieger UD, Stieger pooled, Lee2019
* risk-utility figures showing gain vs lower-tail loss

The goal is not to run another method.  It visualizes the central E8/E9/E10
claim: source-side tangent subspace selection works, and the useful compactness
regime differs by dataset/task.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parents[1] / "results" / "research_outputs"
DEFAULT_STIEGER = RESULTS_DIR / "260628_stieger_fraction_sweep_e9_full" / "summary.json"
DEFAULT_LEE = RESULTS_DIR / "260628_lee2019_fraction_sweep_e9_full" / "summary.json"
DEFAULT_E10 = RESULTS_DIR / "260628_e10_source_side_nested_selection_primary" / "nested_summary.json"
DEFAULT_OUTPUT = RESULTS_DIR / "260628_e11_compactness_frontier"

FAMILIES = ("source_only", "longitudinal", "sep_no_drift", "target_only")
FAMILY_LABELS = {
    "source_only": "source-only",
    "longitudinal": "longitudinal",
    "sep_no_drift": "sep-no-drift",
    "target_only": "target-only",
}
COLORS = {
    "source_only": "#1f77b4",
    "longitudinal": "#d62728",
    "sep_no_drift": "#2ca02c",
    "target_only": "#9467bd",
}
MARKERS = {
    "source_only": "o",
    "longitudinal": "s",
    "sep_no_drift": "^",
    "target_only": "D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stieger-summary", type=Path, default=DEFAULT_STIEGER)
    parser.add_argument("--lee-summary", type=Path, default=DEFAULT_LEE)
    parser.add_argument("--e10-summary", type=Path, default=DEFAULT_E10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_q(method: str) -> float | None:
    if "q1p00" in method:
        return 1.0
    match = re.search(r"q0p(\d+)", method)
    if not match:
        return None
    return int(match.group(1)) / 100.0


def parse_family(method: str) -> str | None:
    for family in FAMILIES:
        if method.startswith(family + "_"):
            return family
    return None


def rows_from_stieger(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    group_map = {
        "primary_pooled": ("Stieger", "pooled LR+UD"),
        "pure_lr": ("Stieger", "LR"),
        "pure_ud": ("Stieger", "UD"),
    }
    for group, (dataset, task) in group_map.items():
        summary = payload["summaries"][group]
        full_key = "full_q1p00" if "full_q1p00" in summary else "longitudinal_q1p00"
        full_acc = summary[full_key]["test_accuracy_subject_pooled_mean"]
        for method, item in summary.items():
            family = parse_family(method)
            q = parse_q(method)
            if family is None or q is None or q >= 1.0:
                continue
            ci = item["gain_vs_full_subject_bootstrap_95ci"]
            rows.append(
                {
                    "dataset": dataset,
                    "task": task,
                    "method": method,
                    "family": family,
                    "family_label": FAMILY_LABELS[family],
                    "q": q,
                    "accuracy": float(item["test_accuracy_subject_pooled_mean"]),
                    "full_accuracy": float(full_acc),
                    "gain_pp": float(item["gain_vs_full_subject_pooled_pp"]),
                    "gain_ci_low": float(ci[0]),
                    "gain_ci_high": float(ci[1]),
                    "r10_loss_pp": float(item["loss_r10_vs_full_pp"]),
                    "p_loss_gt5": float(item["p_gain_vs_full_lt_minus5"]),
                    "q05_gain_pp": float(item["gain_vs_full_q05_pp"]),
                    "n_subjects": int(item["n_subjects"]),
                    "n_units": int(item["n_sessions"]),
                }
            )
    return rows


def rows_from_lee(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    summary = payload["summary"]
    full_acc = summary["full_q1p00"]["accuracy_mean"]
    rows: list[dict[str, object]] = []
    for method, item in summary.items():
        family = parse_family(method)
        q = parse_q(method)
        if family is None or q is None or q >= 1.0:
            continue
        ci = item["gain_vs_full_subject_bootstrap_95ci"]
        rows.append(
            {
                "dataset": "Lee2019",
                "task": "LR",
                "method": method,
                "family": family,
                "family_label": FAMILY_LABELS[family],
                "q": q,
                "accuracy": float(item["accuracy_mean"]),
                "full_accuracy": float(full_acc),
                "gain_pp": float(item["gain_vs_full_mean_pp"]),
                "gain_ci_low": float(ci[0]),
                "gain_ci_high": float(ci[1]),
                "r10_loss_pp": float(item["loss_r10_vs_full_pp"]),
                "p_loss_gt5": float(item["p_gain_vs_full_lt_minus5"]),
                "q05_gain_pp": float(item["gain_vs_full_q05_pp"]),
                "n_subjects": int(item["n_subjects"]),
                "n_units": int(item["n_subjects"]),
            }
        )
    return rows


def e10_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    rows: list[dict[str, object]] = []
    for key, item in payload["summary"].items():
        dataset, policy = key.split("__", maxsplit=1)
        if policy not in {"global_risk", "condition_risk"}:
            continue
        ci = item["gain_vs_full_subject_bootstrap_95ci"]
        rows.append(
            {
                "dataset_policy": key,
                "dataset": dataset,
                "policy": policy,
                "accuracy": float(item["accuracy_subject_pooled_mean"]),
                "gain_pp": float(item["gain_vs_full_subject_pooled_pp"]),
                "gain_ci_low": float(ci[0]),
                "gain_ci_high": float(ci[1]),
                "r10_loss_pp": float(item["loss_r10_vs_full_unit_pp"]),
                "p_loss_gt5": float(item["p_gain_vs_full_lt_minus5_unit"]),
                "choices": json.dumps(item["chosen_candidate_counts"], sort_keys=True),
            }
        )
    return rows


def plot_gain_frontier(df: pd.DataFrame, output_dir: Path) -> None:
    panels = [
        ("Stieger", "LR"),
        ("Stieger", "UD"),
        ("Stieger", "pooled LR+UD"),
        ("Lee2019", "LR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    for ax, (dataset, task) in zip(axes.ravel(), panels):
        sub = df[(df["dataset"] == dataset) & (df["task"] == task)]
        for family in FAMILIES:
            g = sub[sub["family"] == family].sort_values("q")
            if g.empty:
                continue
            ax.plot(
                g["q"],
                g["gain_pp"],
                marker=MARKERS[family],
                linewidth=2,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
            ax.fill_between(
                g["q"].to_numpy(dtype=float),
                g["gain_ci_low"].to_numpy(dtype=float),
                g["gain_ci_high"].to_numpy(dtype=float),
                color=COLORS[family],
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{dataset} {task}")
        ax.set_xlabel("selected tangent-feature fraction q")
        ax.set_ylabel("gain vs full tangent LDA (pp)")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "compactness_gain_frontier.png", dpi=200)
    fig.savefig(output_dir / "compactness_gain_frontier.pdf")
    plt.close(fig)


def plot_risk_utility(df: pd.DataFrame, output_dir: Path) -> None:
    panels = [
        ("Stieger", "LR"),
        ("Stieger", "UD"),
        ("Stieger", "pooled LR+UD"),
        ("Lee2019", "LR"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=False, sharey=False)
    for ax, (dataset, task) in zip(axes.ravel(), panels):
        sub = df[(df["dataset"] == dataset) & (df["task"] == task)]
        for family in FAMILIES:
            g = sub[sub["family"] == family].sort_values("q")
            if g.empty:
                continue
            ax.plot(
                g["r10_loss_pp"],
                g["gain_pp"],
                marker=MARKERS[family],
                linewidth=1.8,
                color=COLORS[family],
                label=FAMILY_LABELS[family],
            )
            for _, row in g.iterrows():
                ax.text(
                    float(row["r10_loss_pp"]) + 0.03,
                    float(row["gain_pp"]) + 0.03,
                    f"{row['q']:.2f}",
                    fontsize=7,
                    color=COLORS[family],
                )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{dataset} {task}")
        ax.set_xlabel("lower-tail R10 loss vs full (pp, lower is safer)")
        ax.set_ylabel("gain vs full tangent LDA (pp)")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "compactness_risk_utility_frontier.png", dpi=200)
    fig.savefig(output_dir / "compactness_risk_utility_frontier.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = rows_from_stieger(args.stieger_summary) + rows_from_lee(args.lee_summary)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_dir / "compactness_frontier_table.csv", index=False)
    e10 = pd.DataFrame(e10_rows(args.e10_summary))
    e10.to_csv(args.output_dir / "e10_nested_policy_points.csv", index=False)

    best_rows = (
        df.sort_values(["dataset", "task", "gain_pp"], ascending=[True, True, False])
        .groupby(["dataset", "task"], as_index=False)
        .head(1)
    )
    best_rows.to_csv(args.output_dir / "compactness_best_by_panel.csv", index=False)
    plot_gain_frontier(df, args.output_dir)
    plot_risk_utility(df, args.output_dir)
    summary = {
        "n_rows": int(len(df)),
        "best_by_panel": best_rows[
            ["dataset", "task", "method", "accuracy", "gain_pp", "gain_ci_low", "gain_ci_high", "r10_loss_pp", "p_loss_gt5"]
        ].to_dict("records"),
        "e10_policy_points": e10.to_dict("records"),
        "outputs": [
            "compactness_frontier_table.csv",
            "compactness_best_by_panel.csv",
            "e10_nested_policy_points.csv",
            "compactness_gain_frontier.png",
            "compactness_gain_frontier.pdf",
            "compactness_risk_utility_frontier.png",
            "compactness_risk_utility_frontier.pdf",
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot matched-seed classification results and export paper-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "RNA-only",
    "ADT-only",
    "RNA+ADT Concatenation",
    "WNN",
    "totalVI",
    "MOFA+",
]
METHOD_LABELS = ["RNA-only", "ADT-only", "RNA+ADT", "WNN", "totalVI", "MOFA+"]
SERIES = [
    ("RF", "accuracy", "RF Acc", "#1f77b4"),
    ("RF", "macro_f1", "RF Macro-F1", "#aec7e8"),
    ("RF", "weighted_f1", "RF Weighted-F1", "#17becf"),
    ("XGB", "accuracy", "XGB Acc", "#2ca02c"),
    ("XGB", "macro_f1", "XGB Macro-F1", "#98df8a"),
    ("XGB", "weighted_f1", "XGB Weighted-F1", "#ff7f0e"),
]


def plot_dataset(metrics: pd.DataFrame, dataset: str, output: Path) -> None:
    data = metrics.loc[metrics["dataset"].eq(dataset)]
    fig, ax = plt.subplots(figsize=(11.9, 5.7))
    x = np.arange(len(METHOD_ORDER))
    width = 0.115
    for i, (model, metric, label, color) in enumerate(SERIES):
        means = []
        standard_deviations = []
        for method in METHOD_ORDER:
            values = data.loc[
                data["method"].eq(method) & data["model"].eq(model), metric
            ]
            if len(values) != 5:
                raise ValueError(
                    f"Expected five {model} {metric} scores for {dataset}/{method}; got {len(values)}"
                )
            means.append(values.mean() * 100)
            standard_deviations.append(values.std(ddof=1) * 100)

        ax.bar(
            x + (i - 2.5) * width,
            means,
            width,
            color=color,
            edgecolor="black",
            linewidth=0.4,
            yerr=standard_deviations,
            error_kw={"ecolor": "black", "elinewidth": 0.8, "capsize": 2},
            label=label,
            zorder=3,
        )

    ax.set_xticks(x, METHOD_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.grid(axis="y", color="#D9DEDE", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.99), fontsize=9)
    title = "PBMC classification performance (RF and XGBoost)" if dataset == "PBMC" else "CD8 classification performance (RF and XGBoost)"
    fig.suptitle(title, y=0.97, fontsize=12)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.20, top=0.89)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.18, facecolor="white")
    plt.close(fig)


def export_table(metrics: pd.DataFrame, dataset: str, output: Path) -> None:
    data = metrics.loc[metrics["dataset"].eq(dataset)]
    rows = []
    for method in METHOD_ORDER:
        row = {"Method": method}
        for model, prefix in [("RF", "RF"), ("XGB", "XGBoost")]:
            subset = data.loc[data["method"].eq(method) & data["model"].eq(model)]
            if len(subset) != 5:
                raise ValueError(f"Expected five rows for {dataset}/{method}/{model}; got {len(subset)}")
            for metric, label in [("macro_f1", "Macro-F1"), ("accuracy", "Accuracy")]:
                mean = subset[metric].mean()
                sd = subset[metric].std(ddof=1)
                row[f"{prefix} {label}"] = f"{mean:.3f} ± {sd:.3f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    required = {"dataset", "method", "model", "seed", "accuracy", "macro_f1", "weighted_f1"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    for dataset, stem in [("PBMC", "pbmc"), ("CD8", "cd8_enriched")]:
        plot_dataset(metrics, dataset, args.out / f"{stem}_method_comparison_mean_sd.png")
        export_table(metrics, dataset, args.out / f"{stem}_method_comparison_mean_sd.csv")
    print(f"Figures written to {args.out.resolve()}")


if __name__ == "__main__":
    main()

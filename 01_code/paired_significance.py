#!/usr/bin/env python3
"""Compare fusion methods with unimodal baselines on matched seed-level macro-F1."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon


FUSION_METHODS = ("RNA+ADT Concatenation", "WNN", "totalVI", "MOFA+")
BASELINES = ("RNA-only", "ADT-only")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    required = {"dataset", "method", "model", "seed", "macro_f1"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics file is missing columns: {sorted(missing)}")

    rows = []
    for (dataset, model), group in metrics.groupby(["dataset", "model"], sort=True):
        scores = group.pivot(index="seed", columns="method", values="macro_f1")
        if len(scores) != 5 or scores.isna().any().any():
            raise ValueError(f"Expected five complete paired splits for {dataset}/{model}")
        for method in FUSION_METHODS:
            for baseline in BASELINES:
                difference = scores[method] - scores[baseline]
                result = wilcoxon(difference, alternative="two-sided", method="exact")
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "baseline": baseline,
                    "n_splits": len(difference),
                    "mean_macro_f1_difference": difference.mean(),
                    "wilcoxon_statistic": result.statistic,
                    "p_value_two_sided": result.pvalue,
                })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Paired comparisons written to {args.out.resolve()}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create CD8 CellTypist confidence scores for the saved CD8 labels.

The earlier CD8 annotation file kept the labels but not the probability matrix.
This helper uses the saved CellTypist model directly, aligns the CD8 RNA matrix
to the model gene order, and adds the probability for the retained
majority-voting label.
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


DEFAULT_OUT = Path(__file__).resolve().parent / "label_outputs" / "cd8_celltypist_predictions.csv"


def read_gene_by_cell_csv(path: Path) -> pd.DataFrame:
    matrix = pd.read_csv(path, index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def predict_from_celltypist_bundle(gene_by_cell: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    classifier = bundle["Model"]
    scaler = bundle["Scaler_"]
    model_genes = pd.Index([str(gene) for gene in classifier.features])
    model_gene_to_position = pd.Series(np.arange(len(model_genes)), index=model_genes)

    input_genes = pd.Index(gene_by_cell.index.astype(str))
    common_genes = input_genes[input_genes.isin(model_genes)]
    if common_genes.empty:
        raise RuntimeError("No overlap between CD8 RNA genes and CellTypist model genes")

    lr_idx = model_gene_to_position.loc[common_genes].to_numpy()
    X = gene_by_cell.loc[common_genes].T.to_numpy(dtype=np.float32)
    means = scaler.mean_[lr_idx] if scaler.with_mean else 0
    scales = scaler.scale_[lr_idx]
    X = (X - means) / scales
    X[X > 10] = 10

    scores = X @ classifier.coef_[:, lr_idx].T + classifier.intercept_
    probs = expit(scores)
    pred_idx = scores.argmax(axis=1)

    cells = gene_by_cell.columns.astype(str)
    predicted = pd.DataFrame(
        {
            "barcode": cells,
            "predicted_labels": classifier.classes_[pred_idx],
        }
    )
    prob_df = pd.DataFrame(probs, index=cells, columns=classifier.classes_)
    return predicted, prob_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate CD8 CellTypist predictions with confidence scores.")
    parser.add_argument("--rna", type=Path, required=True, help="Gene-by-cell log-normalized CD8 RNA matrix")
    parser.add_argument("--model", type=Path, required=True, help="CellTypist model file")
    parser.add_argument("--old-labels", type=Path, required=True, help="Existing CD8 annotation table with majority_voting")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.rna.exists():
        raise FileNotFoundError(args.rna)
    if not args.model.exists():
        raise FileNotFoundError(args.model)

    gene_by_cell = read_gene_by_cell_csv(args.rna)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with args.model.open("rb") as handle:
            bundle = pickle.load(handle)

    labels, prob_df = predict_from_celltypist_bundle(gene_by_cell, bundle)
    labels["predicted_label_prob"] = [
        prob_df.loc[barcode, label]
        for barcode, label in zip(labels["barcode"], labels["predicted_labels"], strict=True)
    ]

    if args.old_labels.exists():
        old = pd.read_csv(args.old_labels, index_col=0)
        old.index = old.index.astype(str)
        if "majority_voting" in old.columns:
            labels["majority_voting"] = old["majority_voting"].reindex(labels["barcode"]).to_numpy()
        else:
            labels["majority_voting"] = labels["predicted_labels"]
    else:
        labels["majority_voting"] = labels["predicted_labels"]

    labels = labels.dropna(subset=["majority_voting"])
    labels["majority_voting_prob"] = [
        prob_df.loc[barcode, majority] if majority in prob_df.columns else prob_df.loc[barcode].max()
        for barcode, majority in zip(labels["barcode"], labels["majority_voting"], strict=True)
    ]
    labels["celltype_ref_prob"] = labels["majority_voting_prob"]
    labels = labels.dropna(subset=["predicted_label_prob", "majority_voting_prob"])
    keep_cols = [
        "barcode",
        "predicted_labels",
        "majority_voting",
        "predicted_label_prob",
        "majority_voting_prob",
        "celltype_ref_prob",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels[keep_cols].to_csv(args.out, index=False)

    n_confident = int((labels["celltype_ref_prob"] >= 0.5).sum())
    print(f"Wrote {len(labels)} CD8 predictions to {args.out}")
    print(f"Cells with probability >= 0.5: {n_confident}")


if __name__ == "__main__":
    main()

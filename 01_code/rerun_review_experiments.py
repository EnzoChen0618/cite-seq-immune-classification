#!/usr/bin/env python3
"""
Rerun the classification tables requested after review.

The script uses the representations that were already generated in the
project folder, then evaluates all methods with the same cells, labels,
train/test split, seeds, and classifier settings.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except Exception:  # keep RF reruns usable if xgboost is not installed yet
    XGBClassifier = None


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "01_code" else SCRIPT_DIR
PACKAGE_DATA = PACKAGE_DIR / "04_feature_inputs"
PACKAGE_LABELS = PACKAGE_DIR / "00_label_inputs"

PBMC_CELLTYPIST_PREDICTIONS = PACKAGE_LABELS / "pbmc_celltypist_predictions.csv"
CD8_CELLTYPIST_PREDICTIONS = PACKAGE_LABELS / "cd8_celltypist_predictions.csv"

PBMC_RNA = PACKAGE_DATA / "pbmc" / "pbmc_rna_preprocessed.csv"
PBMC_ADT = PACKAGE_DATA / "pbmc" / "pbmc_adt_clr_preprocessed_patched.h5ad"
PBMC_WNN = PACKAGE_DATA / "pbmc" / "pbmc_rna_with_adtclr_wnn.h5ad"
PBMC_TOTALVI = PACKAGE_DATA / "pbmc" / "pbmc_totalvi_fused.h5ad"
PBMC_MOFA = PACKAGE_DATA / "pbmc" / "pbmc_mofa_fused.h5ad"

CD8_RNA = PACKAGE_DATA / "cd8" / "rna_scaled_hvg.csv"
CD8_ADT = PACKAGE_DATA / "cd8" / "GSE100866_CD8_merged-ADT_umi_clr-transformed.csv"
CD8_WNN = PACKAGE_DATA / "cd8" / "CD8_wnn_fused.h5ad"
CD8_TOTALVI = PACKAGE_DATA / "cd8" / "CD8_totalVI_fused.h5ad"
CD8_MOFA = PACKAGE_DATA / "cd8" / "CD8_mofa_fused.h5ad"

SEEDS = [7, 21, 42, 77, 2024]
TEST_SIZE = 0.20
WNN_GRAPH_COMPONENTS = 20
CD8_CYTOTOXIC_T_LABELS = [
    "CD8a/a",
    "CD8a/b(entry)",
    "Tcm/Naive cytotoxic T cells",
    "Tem/Temra cytotoxic T cells",
    "Tem/Trm cytotoxic T cells",
    "Trm cytotoxic T cells",
]


@dataclass(frozen=True)
class FeatureBlock:
    dataset: str
    method: str
    features: pd.DataFrame


def read_h5ad_obsm(path: Path, key: str) -> pd.DataFrame:
    obj = ad.read_h5ad(path)
    if key not in obj.obsm:
        raise KeyError(f"{path.name} does not contain obsm['{key}']")
    arr = np.asarray(obj.obsm[key])
    cols = [f"{key}_{i + 1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, index=obj.obs_names.astype(str), columns=cols)


def read_h5ad_obsp_svd(path: Path, key: str, n_components: int) -> pd.DataFrame:
    obj = ad.read_h5ad(path)
    if key not in obj.obsp:
        raise KeyError(f"{path.name} does not contain obsp['{key}']")
    graph = obj.obsp[key]
    n = min(n_components, max(2, min(graph.shape) - 1))
    emb = TruncatedSVD(n_components=n, random_state=0).fit_transform(graph)
    cols = [f"X_wnn_graph_{i + 1}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=obj.obs_names.astype(str), columns=cols)


def read_h5ad_obs_labels(path: Path, column: str) -> pd.Series:
    obj = ad.read_h5ad(path, backed="r")
    if column not in obj.obs.columns:
        obj.file.close()
        raise KeyError(f"{path.name} does not contain obs['{column}']")
    labels = obj.obs[column].astype(str).copy()
    labels.index = obj.obs_names.astype(str)
    obj.file.close()
    return labels


def read_csv_features(path: Path, index_col: str | int = 0) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=index_col)
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def read_transposed_marker_csv(path: Path, index_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).set_index(index_name).T
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


def clean_labels(labels: pd.Series) -> pd.Series:
    labels = labels.astype("string").str.strip()
    labels = labels.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "Unknown": pd.NA})
    return labels.dropna().astype(str)


def read_confident_labels(
    path: Path,
    cell_col: str,
    label_col: str,
    prob_col: str,
    min_probability: float,
    allowed_labels: list[str] | None = None,
) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    missing = [col for col in [cell_col, label_col, prob_col] if col not in df.columns]
    if missing:
        raise KeyError(f"{path.name} is missing columns: {missing}")

    df[cell_col] = df[cell_col].astype(str)
    df[label_col] = df[label_col].astype("string").str.strip()
    df[prob_col] = pd.to_numeric(df[prob_col], errors="coerce")
    df = df.dropna(subset=[cell_col, label_col, prob_col])
    df = df[df[prob_col] >= min_probability]
    if allowed_labels is not None:
        df = df[df[label_col].isin(allowed_labels)]

    labels = pd.Series(df[label_col].astype(str).to_numpy(), index=df[cell_col].astype(str), name=label_col)
    labels = labels[~labels.index.duplicated(keep="first")]
    return clean_labels(labels)


def make_pbmc_blocks(label_prob_threshold: float, label_path: Path) -> tuple[list[FeatureBlock], pd.Series]:
    labels = read_confident_labels(
        label_path,
        cell_col="barcode",
        label_col="celltype_ref",
        prob_col="celltype_ref_prob",
        min_probability=label_prob_threshold,
    )

    rna = read_csv_features(PBMC_RNA)
    adt = read_h5ad_obsm(PBMC_ADT, "X_pca_adt")
    concat = join_feature_tables(rna, adt, "rna", "adt")

    blocks = [
        FeatureBlock("PBMC", "RNA-only", rna),
        FeatureBlock("PBMC", "ADT-only", adt),
        FeatureBlock("PBMC", "RNA+ADT Concatenation", concat),
        FeatureBlock("PBMC", "WNN", read_h5ad_obsp_svd(PBMC_WNN, "connectivities", WNN_GRAPH_COMPONENTS)),
        FeatureBlock("PBMC", "totalVI", read_h5ad_obsm(PBMC_TOTALVI, "X_totalvi")),
        FeatureBlock("PBMC", "MOFA+", read_h5ad_obsm(PBMC_MOFA, "X_mofa")),
    ]
    return blocks, labels


def make_cd8_blocks(
    label_prob_threshold: float,
    label_path: Path,
    label_col: str = "majority_voting",
    prob_col: str = "celltype_ref_prob",
    allowed_labels: list[str] | None = None,
) -> tuple[list[FeatureBlock], pd.Series]:
    labels = read_confident_labels(
        label_path,
        cell_col="barcode",
        label_col=label_col,
        prob_col=prob_col,
        min_probability=label_prob_threshold,
        allowed_labels=allowed_labels,
    )

    rna = read_transposed_marker_csv(CD8_RNA, "gene")
    adt = read_transposed_marker_csv(CD8_ADT, "marker")
    concat = join_feature_tables(rna, adt, "rna", "adt")

    blocks = [
        FeatureBlock("CD8", "RNA-only", rna),
        FeatureBlock("CD8", "ADT-only", adt),
        FeatureBlock("CD8", "RNA+ADT Concatenation", concat),
        FeatureBlock("CD8", "WNN", read_h5ad_obsp_svd(CD8_WNN, "connectivities", WNN_GRAPH_COMPONENTS)),
        FeatureBlock("CD8", "totalVI", read_h5ad_obsm(CD8_TOTALVI, "X_totalVI")),
        FeatureBlock("CD8", "MOFA+", read_h5ad_obsm(CD8_MOFA, "X_mofa")),
    ]
    return blocks, labels


def join_feature_tables(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> pd.DataFrame:
    cells = left.index.intersection(right.index)
    left = left.loc[cells].add_prefix(f"{left_name}_")
    right = right.loc[cells].add_prefix(f"{right_name}_")
    return pd.concat([left, right], axis=1)


def common_labeled_cells(blocks: list[FeatureBlock], labels: pd.Series, min_class_size: int) -> pd.Index:
    cells = labels.index
    for block in blocks:
        cells = cells.intersection(block.features.index)

    y = labels.loc[cells]
    counts = y.value_counts()
    keep_classes = counts[counts >= min_class_size].index
    return y[y.isin(keep_classes)].index


def as_matrix(features: pd.DataFrame, cells: pd.Index) -> np.ndarray:
    X = features.loc[cells].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def split_indices(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=seed,
        stratify=y,
    )
    return train_idx, test_idx


def rf_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def xgb_model(seed: int, n_classes: int):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed in this Python environment")
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def score_predictions(dataset: str, method: str, model_name: str, seed: int, y_true, y_pred) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "model": model_name,
        "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_block(
    block: FeatureBlock,
    labels: pd.Series,
    cells: pd.Index,
    models: list[str],
    out_dir: Path,
) -> tuple[list[dict], list[dict]]:
    X = as_matrix(block.features, cells)
    y_text = labels.loc[cells].astype(str).to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)
    rows = []
    confusion_jobs = []

    for seed in SEEDS:
        train_idx, test_idx = split_indices(y, seed)

        if "rf" in models:
            clf = rf_model(seed)
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            rows.append(score_predictions(block.dataset, block.method, "RF", seed, y[test_idx], pred))
            if seed == 42:
                confusion_jobs.append(
                    {
                        "dataset": block.dataset,
                        "method": block.method,
                        "model": "RF",
                        "y_true": y[test_idx],
                        "y_pred": pred,
                        "labels": encoder.classes_,
                    }
                )

        if "xgb" in models:
            clf = xgb_model(seed, len(encoder.classes_))
            clf.fit(X[train_idx], y[train_idx])
            pred = clf.predict(X[test_idx])
            rows.append(score_predictions(block.dataset, block.method, "XGB", seed, y[test_idx], pred))
            if seed == 42:
                confusion_jobs.append(
                    {
                        "dataset": block.dataset,
                        "method": block.method,
                        "model": "XGB",
                        "y_true": y[test_idx],
                        "y_pred": pred,
                        "labels": encoder.classes_,
                    }
                )

    meta = {
        "dataset": block.dataset,
        "method": block.method,
        "n_cells": int(len(cells)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
    }
    return rows, [dict(meta, **job) for job in confusion_jobs]


def summarize_scores(scores: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]
    grouped = scores.groupby(["dataset", "method", "model"], sort=False)[metrics]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std(ddof=1).add_suffix("_sd")
    return pd.concat([means, stds], axis=1).reset_index()


def format_mean_sd(mean: float, sd: float) -> str:
    if pd.isna(sd):
        sd = 0.0
    return f"{mean:.3f} +/- {sd:.3f}"


def reviewer_table(summary: pd.DataFrame, dataset: str) -> pd.DataFrame:
    wanted_methods = ["RNA-only", "ADT-only", "RNA+ADT Concatenation", "WNN", "totalVI", "MOFA+"]
    rows = []
    for method in wanted_methods:
        row = {"Method": method}
        subset = summary[(summary["dataset"] == dataset) & (summary["method"] == method)]
        for model in ["RF", "XGB"]:
            hit = subset[subset["model"] == model]
            for metric, label in [
                ("accuracy", "Acc"),
                ("macro_f1", "Macro-F1"),
                ("weighted_f1", "Weighted-F1"),
            ]:
                col = f"{model} {label}"
                if hit.empty:
                    row[col] = "not run"
                else:
                    row[col] = format_mean_sd(
                        float(hit[f"{metric}_mean"].iloc[0]),
                        float(hit[f"{metric}_sd"].iloc[0]),
                    )
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion_matrix(job: dict, out_dir: Path) -> None:
    labels = job["labels"]
    cm = confusion_matrix(job["y_true"], job["y_pred"], labels=np.arange(len(labels)))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    stem = f"{job['dataset'].lower()}_{safe_name(job['method'])}_{job['model'].lower()}"
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"confusion_{stem}.csv")

    width = max(7, 0.5 * len(labels) + 3)
    height = max(6, 0.45 * len(labels) + 2)
    plt.figure(figsize=(width, height))
    sns.heatmap(cm_norm, cmap="Blues", xticklabels=labels, yticklabels=labels, vmin=0, vmax=1)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{job['dataset']} {job['method']} {job['model']} confusion matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{stem}.png", dpi=220)
    plt.close()


def safe_name(text: str) -> str:
    return (
        text.lower()
        .replace("+", "plus")
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def display_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PACKAGE_DIR.resolve()))
    except ValueError:
        return str(path)


def write_protocol(
    out_dir: Path,
    models: list[str],
    min_class_size: int,
    label_prob_threshold: float,
    pbmc_label_path: Path,
    cd8_label_path: Path,
    cd8_label_column: str,
    cd8_probability_column: str,
    cd8_label_set: str,
) -> None:
    hyper_rows = [
        {"component": "Label filtering", "parameter": "PBMC label source", "value": display_path(pbmc_label_path)},
        {"component": "Label filtering", "parameter": "CD8 label source", "value": display_path(cd8_label_path)},
        {"component": "Label filtering", "parameter": "Minimum CellTypist probability", "value": str(label_prob_threshold)},
        {"component": "Label filtering", "parameter": "CD8 label column", "value": cd8_label_column},
        {"component": "Label filtering", "parameter": "CD8 probability column", "value": cd8_probability_column},
        {"component": "Label filtering", "parameter": "CD8 label set", "value": cd8_label_set},
        {"component": "RNA preprocessing", "parameter": "Highly variable genes", "value": "2000"},
        {"component": "RNA preprocessing", "parameter": "PCA components", "value": "50"},
        {"component": "ADT preprocessing", "parameter": "Normalization", "value": "CLR"},
        {"component": "WNN", "parameter": "Neighbors", "value": "15 for PBMC, 30 in the saved CD8 WNN run"},
        {"component": "WNN", "parameter": "Leiden resolution", "value": "0.8 for PBMC, 0.6 in the saved CD8 WNN run"},
        {"component": "WNN", "parameter": "Classifier representation", "value": f"{WNN_GRAPH_COMPONENTS}-component SVD embedding of the saved WNN connectivities graph"},
        {"component": "totalVI", "parameter": "Latent representation", "value": "Saved X_totalvi/X_totalVI embedding"},
        {"component": "MOFA+", "parameter": "Factors", "value": "Saved X_mofa latent factors"},
        {"component": "Evaluation split", "parameter": "Train/test", "value": "Stratified 80/20"},
        {"component": "Evaluation split", "parameter": "Seeds", "value": ", ".join(map(str, SEEDS))},
        {"component": "Random Forest", "parameter": "n_estimators", "value": "200"},
        {"component": "Random Forest", "parameter": "max_depth", "value": "15"},
        {"component": "Random Forest", "parameter": "min_samples_split", "value": "5"},
        {"component": "Random Forest", "parameter": "class_weight", "value": "balanced"},
        {"component": "XGBoost", "parameter": "n_estimators", "value": "300"},
        {"component": "XGBoost", "parameter": "learning_rate", "value": "0.05"},
        {"component": "XGBoost", "parameter": "max_depth", "value": "5"},
        {"component": "XGBoost", "parameter": "subsample", "value": "0.8"},
        {"component": "XGBoost", "parameter": "colsample_bytree", "value": "0.8"},
        {"component": "XGBoost", "parameter": "tree_method", "value": "hist"},
    ]
    pd.DataFrame(hyper_rows).to_csv(out_dir / "hyperparameters.csv", index=False)

    protocol = {
        "label_filtering": [
            f"Low-confidence CellTypist labels are filtered first using probability >= {label_prob_threshold}.",
            "Labels are selected once per dataset before evaluation.",
            "The same retained cells are used for RNA-only, ADT-only, RNA+ADT concatenation, WNN, totalVI, and MOFA+.",
            f"Classes with fewer than {min_class_size} cells after alignment are removed before stratified splitting.",
        ],
        "split": {
            "type": "stratified train/test split",
            "train_fraction": 0.80,
            "test_fraction": 0.20,
            "seeds": SEEDS,
        },
        "models_run": models,
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_split": 5,
            "class_weight": "balanced",
        },
        "xgboost": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
        },
        "feature_notes": {
            "RNA": f"PBMC RNA: {display_path(PBMC_RNA)}; CD8 RNA: {display_path(CD8_RNA)}.",
            "ADT": f"PBMC ADT: {display_path(PBMC_ADT)}; CD8 ADT: {display_path(CD8_ADT)}.",
            "WNN": f"Uses a {WNN_GRAPH_COMPONENTS}-component SVD embedding of the saved WNN connectivities graph, not Leiden clusters or cluster IDs.",
            "totalVI": f"Uses saved totalVI latent representations: {display_path(PBMC_TOTALVI)}; {display_path(CD8_TOTALVI)}.",
            "MOFA+": f"Uses saved MOFA latent factors: {display_path(PBMC_MOFA)}; {display_path(CD8_MOFA)}.",
        },
        "label_sources": {
            "PBMC": display_path(pbmc_label_path),
            "CD8": display_path(cd8_label_path),
        },
        "cd8_label_settings": {
            "label_column": cd8_label_column,
            "probability_column": cd8_probability_column,
            "label_set": cd8_label_set,
            "allowed_labels": CD8_CYTOTOXIC_T_LABELS if cd8_label_set == "cytotoxic_t" else "all",
        },
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def run_dataset(
    name: str,
    loader: Callable[[float], tuple[list[FeatureBlock], pd.Series]],
    models: list[str],
    out_dir: Path,
    min_class_size: int,
    label_prob_threshold: float,
) -> tuple[pd.DataFrame, list[dict], dict]:
    blocks, labels = loader(label_prob_threshold)
    cells = common_labeled_cells(blocks, labels, min_class_size)
    if len(cells) == 0:
        raise RuntimeError(f"{name}: no common labeled cells left after filtering")

    y = labels.loc[cells]
    dataset_meta = {
        "dataset": name,
        "n_cells": int(len(cells)),
        "n_classes": int(y.nunique()),
        "label_probability_threshold": float(label_prob_threshold),
        "class_counts": y.value_counts().to_dict(),
    }
    all_rows = []
    all_cms = []
    for block in blocks:
        rows, cms = evaluate_block(block, labels, cells, models, out_dir)
        all_rows.extend(rows)
        all_cms.extend(cms)
    return pd.DataFrame(all_rows), all_cms, dataset_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun reviewer-requested classification baselines.")
    parser.add_argument("--out", default="rerun_results", help="Output directory")
    parser.add_argument("--datasets", default="PBMC,CD8", help="Comma-separated: PBMC,CD8")
    parser.add_argument("--models", default="rf,xgb", help="Comma-separated: rf,xgb")
    parser.add_argument("--min-class-size", type=int, default=5)
    parser.add_argument("--label-prob-threshold", type=float, default=0.5)
    parser.add_argument("--pbmc-labels", type=Path, default=PBMC_CELLTYPIST_PREDICTIONS)
    parser.add_argument("--cd8-labels", type=Path, default=CD8_CELLTYPIST_PREDICTIONS)
    parser.add_argument("--cd8-label-column", default="majority_voting")
    parser.add_argument("--cd8-prob-column", default="celltype_ref_prob")
    parser.add_argument("--cd8-label-set", choices=["all", "cytotoxic_t"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    if "xgb" in models and XGBClassifier is None:
        raise RuntimeError(
            "xgboost is not installed. Install it or rerun with --models rf."
        )

    cd8_allowed_labels = CD8_CYTOTOXIC_T_LABELS if args.cd8_label_set == "cytotoxic_t" else None
    loaders = {
        "PBMC": lambda threshold: make_pbmc_blocks(threshold, args.pbmc_labels),
        "CD8": lambda threshold: make_cd8_blocks(
            threshold,
            args.cd8_labels,
            label_col=args.cd8_label_column,
            prob_col=args.cd8_prob_column,
            allowed_labels=cd8_allowed_labels,
        ),
    }

    score_frames = []
    confusion_jobs = []
    meta = []
    for dataset in datasets:
        if dataset not in loaders:
            raise ValueError(f"Unknown dataset: {dataset}")
        scores, cms, dataset_meta = run_dataset(
            dataset,
            loaders[dataset],
            models,
            out_dir,
            args.min_class_size,
            args.label_prob_threshold,
        )
        score_frames.append(scores)
        confusion_jobs.extend(cms)
        meta.append(dataset_meta)

    scores = pd.concat(score_frames, ignore_index=True)
    scores.to_csv(out_dir / "all_seed_metrics.csv", index=False)

    summary = summarize_scores(scores)
    summary.to_csv(out_dir / "summary_metrics_mean_sd.csv", index=False)

    for dataset in datasets:
        reviewer_table(summary, dataset).to_csv(out_dir / f"{dataset.lower()}_reviewer_table.csv", index=False)

    preferred = "XGB" if "xgb" in models else "RF"
    for job in confusion_jobs:
        if job["model"] == preferred:
            save_confusion_matrix(job, out_dir)

    write_protocol(
        out_dir,
        models,
        args.min_class_size,
        args.label_prob_threshold,
        args.pbmc_labels,
        args.cd8_labels,
        args.cd8_label_column,
        args.cd8_prob_column,
        args.cd8_label_set,
    )
    (out_dir / "dataset_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Finished. Results written to {out_dir}")


if __name__ == "__main__":
    main()

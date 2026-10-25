#!/usr/bin/env python3
"""Redraw descriptive CITE-seq panels for the shared classification cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap


ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "04_feature_inputs"
LABELS = ROOT / "00_label_inputs"


def confident_cells(dataset: str) -> tuple[pd.Index, pd.Series]:
    if dataset == "CD8":
        labels = pd.read_csv(LABELS / "cd8_celltypist_predictions.csv")
        name = "majority_voting"
        sources = [
            pd.read_csv(INPUTS / "cd8/rna_scaled_hvg.csv", nrows=0).columns[1:],
            pd.read_csv(INPUTS / "cd8/GSE100866_CD8_merged-ADT_umi_clr-transformed.csv", nrows=0).columns[1:],
        ]
        objects = ["CD8_wnn_fused.h5ad", "CD8_totalVI_fused.h5ad", "CD8_mofa_fused.h5ad"]
        folder = INPUTS / "cd8"
    else:
        labels = pd.read_csv(LABELS / "pbmc_celltypist_predictions.csv")
        name = "celltype_ref"
        sources = [
            pd.read_csv(INPUTS / "pbmc/pbmc_rna_preprocessed.csv", usecols=[0]).iloc[:, 0],
        ]
        objects = [
            "pbmc_adt_clr_preprocessed_patched.h5ad",
            "pbmc_rna_with_adtclr_wnn.h5ad",
            "pbmc_totalvi_fused.h5ad",
            "pbmc_mofa_fused.h5ad",
        ]
        folder = INPUTS / "pbmc"

    labels = labels.dropna(subset=["barcode", name, "celltype_ref_prob"])
    labels = labels.loc[labels["celltype_ref_prob"] >= 0.5]
    labels[name] = labels[name].astype(str).str.strip()
    labels = labels.loc[~labels[name].isin(["", "nan", "None", "Unknown"])]
    labels = labels.drop_duplicates("barcode", keep="first").set_index("barcode")[name]
    cells = labels.index
    for source in sources:
        cells = cells.intersection(pd.Index(source.astype(str)), sort=False)
    for filename in objects:
        obj = ad.read_h5ad(folder / filename, backed="r")
        cells = cells.intersection(pd.Index(obj.obs_names.astype(str)), sort=False)
        obj.file.close()
    counts = labels.loc[cells].value_counts()
    cells = labels.loc[cells][labels.loc[cells].isin(counts[counts >= 5].index)].index
    expected = 387 if dataset == "CD8" else 1621
    if len(cells) != expected:
        raise ValueError(f"{dataset}: expected {expected} retained cells, found {len(cells)}")
    return cells, labels.loc[cells]


def positions(obj: ad.AnnData, cells: pd.Index, key: str) -> np.ndarray:
    loc = obj.obs_names.get_indexer(cells)
    if (loc < 0).any():
        raise ValueError(f"Missing cells in {key}")
    return np.asarray(obj.obsm[key])[loc]


def categories(ax: plt.Axes, xy: np.ndarray, values: pd.Series, title: str) -> None:
    names = sorted(values.unique())
    colors = plt.get_cmap("tab20", len(names))
    for i, name in enumerate(names):
        mask = values.to_numpy() == name
        ax.scatter(xy[mask, 0], xy[mask, 1], s=7, color=colors(i), label=name, linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])


def markers(ax: plt.Axes, xy: np.ndarray, values: np.ndarray, title: str) -> None:
    order = np.argsort(values)
    sc = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], s=5, cmap="viridis", linewidths=0, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cd8_figures(out: Path) -> None:
    cells, labels = confident_cells("CD8")
    wnn = ad.read_h5ad(INPUTS / "cd8/CD8_wnn_fused.h5ad", backed="r")
    mofa = ad.read_h5ad(INPUTS / "cd8/CD8_mofa_fused.h5ad", backed="r")
    xy = positions(wnn, cells, "X_umap")
    factors = positions(mofa, cells, "X_mofa")
    clusters = wnn.obs.loc[cells, "leiden"].astype(str)
    marker_path = INPUTS / "cd8/GSE100866_CD8_merged-ADT_umi_clr-transformed.csv"
    adt = pd.read_csv(marker_path, index_col=0).loc[["CD8", "CD3", "CD56", "CD57", "CD2"], cells]

    fig = plt.figure(figsize=(8.0, 9.2), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=[2.5, 1.3, 1.3])
    top = fig.add_subplot(grid[0, :])
    categories(top, xy, clusters, "WNN Leiden clusters")
    top.set_xlabel("UMAP 1")
    top.set_ylabel("UMAP 2")
    top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Cluster", frameon=False, fontsize=8)
    for i, marker in enumerate(adt.index):
        ax = fig.add_subplot(grid[1 + i // 3, (i % 3) * 2 : (i % 3 + 1) * 2])
        markers(ax, xy, adt.loc[marker].to_numpy(float), marker)
    save(fig, out / "fig1_cd8_wnn_filtered.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    categories(axes[0], xy, clusters, "Reference WNN Leiden")
    categories(axes[1], factors, clusters, "MOFA+ factors")
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[1].set_xlabel("Factor 1")
    axes[1].set_ylabel("Factor 2")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Cluster", frameon=False, fontsize=8)
    save(fig, out / "fig2_cd8_mofa_filtered.png")
    wnn.file.close()
    mofa.file.close()
    pd.DataFrame({"barcode": cells, "label": labels.to_numpy()}).to_csv(out / "cd8_figure_cells.csv", index=False)


def pbmc_figures(out: Path) -> None:
    cells, labels = confident_cells("PBMC")
    wnn = ad.read_h5ad(INPUTS / "pbmc/pbmc_rna_with_adtclr_wnn.h5ad", backed="r")
    xy_wnn = positions(wnn, cells, "X_umap")
    rna_pca = positions(wnn, cells, "X_rna")
    xy_rna = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42).fit_transform(rna_pca)
    markers_to_show = ["CD3", "CD4", "CD8", "CD14", "CD19", "CD16"]
    adt_obj = ad.read_h5ad(INPUTS / "pbmc/pbmc_adt_clr_preprocessed_patched.h5ad", backed="r")
    adt = positions(adt_obj, cells, "protein_clr")
    protein_names = list(adt_obj.var_names)

    fig = plt.figure(figsize=(9, 9.4), layout="constrained")
    grid = fig.add_gridspec(3, 6, height_ratios=[2.4, 1.2, 1.2])
    top = fig.add_subplot(grid[0, :])
    categories(top, xy_rna, labels, "RNA-based cell types")
    top.set_xlabel("UMAP 1")
    top.set_ylabel("UMAP 2")
    top.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=7)
    for i, marker in enumerate(markers_to_show):
        ax = fig.add_subplot(grid[1 + i // 3, (i % 3) * 2 : (i % 3 + 1) * 2])
        markers(ax, xy_rna, adt[:, protein_names.index(marker)], marker)
    save(fig, out / "fig4_pbmc_rna_adt_filtered.png")

    fig, ax = plt.subplots(figsize=(7.2, 5.2), layout="constrained")
    clusters = wnn.obs.loc[cells, "leiden_wnn"].astype(str)
    categories(ax, xy_wnn, clusters, "WNN Leiden clusters")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title="Cluster", frameon=False, fontsize=7, ncol=2)
    save(fig, out / "fig5_pbmc_wnn_filtered.png")
    wnn.file.close()
    adt_obj.file.close()
    pd.DataFrame({"barcode": cells, "label": labels.to_numpy()}).to_csv(out / "pbmc_figure_cells.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cd8_figures(args.out)
    pbmc_figures(args.out)
    print(f"Saved filtered descriptive figures and barcode lists to {args.out}")


if __name__ == "__main__":
    main()

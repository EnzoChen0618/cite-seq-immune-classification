# CITE-seq immune-cell classification

Code and aggregate results for the PBMC and CD8-enriched classification comparisons in the accompanying paper. The analysis compares RNA-only, ADT-only, RNA+ADT concatenation, WNN, totalVI, and MOFA+ representations with Random Forest and XGBoost.

## Scope and data

The paired RNA and ADT source data are available from GEO under accession [GSE100866](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866). This public release does **not** contain per-cell CellTypist labels, processed feature matrices, saved WNN graphs, or totalVI/MOFA+ embeddings. Those inputs must be obtained separately and placed under `00_label_inputs/` and `04_feature_inputs/` with the filenames specified in `01_code/rerun_review_experiments.py` and `results/protocol.json`.

The code reproduces the downstream classification comparison **from saved representations**. It does not train WNN, totalVI, or MOFA+ from raw sequencing counts. Without the withheld analysis-ready inputs, the aggregate results can be inspected and the performance figures regenerated, but the classification experiment itself cannot be rerun end to end.

The CD8-enriched label set includes non-CD8 immune populations and should not be interpreted as a pure CD8 T-cell-state task.

## Evaluation

Low-confidence CellTypist labels (probability below 0.5) are removed before comparing methods. Classes with fewer than five aligned cells are also removed before splitting. Each method then uses the same retained cells and labels on stratified 80/20 train/test splits with seeds 7, 21, 42, 77, and 2024. Classifier settings are recorded in `results/protocol.json` and `results/hyperparameters.csv`.

WNN classification uses a 20-component SVD embedding of the saved connectivity graph, not Leiden cluster IDs or UMAP coordinates. The multimodal representations were learned before supervised splitting, so the evaluation is transductive. CellTypist labels are RNA-derived and are not independent of the RNA modality.

## Run

Use Python 3.10 or newer. Install the listed dependencies:

```bash
python -m pip install -r 01_code/requirements.txt
```

To redraw the two performance figures and tables from the included per-seed aggregate metrics:

```bash
python 01_code/make_performance_figures.py \
  --metrics results/all_seed_metrics.csv \
  --out rerun_figures
```

The plots use the six accuracy, macro-F1, and weighted-F1 series shown in the paper, with five-split standard deviations as error bars. Recalculate the exploratory two-sided paired Wilcoxon comparisons from the same metrics:

```bash
python 01_code/paired_significance.py \
  --metrics results/all_seed_metrics.csv \
  --out rerun_results/paired_wilcoxon_macro_f1.csv
```

After supplying the omitted per-cell labels and saved feature inputs, rerun the classification comparison:

```bash
python 01_code/rerun_review_experiments.py \
  --models rf,xgb \
  --label-prob-threshold 0.5 \
  --out rerun_results
```

`make_filtered_exploratory_figures.py` also requires the omitted per-cell inputs. `make_cd8_celltypist_predictions.py` additionally requires a compatible CellTypist model and source RNA matrix; it reads a Python pickle file, so only use a model from a trusted source.

## Included results

`results/all_seed_metrics.csv` gives per-seed classification metrics, `results/summary_metrics_mean_sd.csv` gives mean and standard deviation, and the remaining CSV/JSON files record the manuscript tables, cohort counts, settings, and protocol. No cell-level records are included.

## Contributors

- Enzo Chen
- Yutong Zheng
- Aike Liu
- Fengyang Weng

"""
OLS residualization of scGPT zero-shot attention against gene-level features.

For each cell type:
  1. Recompute the same signed PD-vs-control attention score used by the
     existing attention task.
  2. Compute gene-level univariate features from the original h5ad X:
       - mean expression
       - expression variance
       - dropout rate (fraction of cells with X == 0)
  3. Fit, across genes, the OLS model
       Attention_g = beta_0 + beta_1*expression_g
                              + beta_2*variance_g
                              + beta_3*dropout_g + epsilon_g
  4. Save the fitted value and residual epsilon_g for every gene.

The residual is the part of the signed attention score not explained by the
three supplied univariate gene features.

This is intentionally OLS-only for the first implementation. It uses numpy
rather than adding statsmodels as a dependency.
"""

from pathlib import Path
from typing import Dict, Tuple

import csv
import json

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import issparse

from pipeline import config as C
from pipeline.steps import _step7_common as common
from pipeline.utils.scgpt_model import load_model_configs, load_vocab


DATA_DIR = C.DATA_DIR
TRAIN_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_train.h5ad"
VAL_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_val.h5ad"


def _load_full_dataset() -> ad.AnnData:
    train = sc.read_h5ad(TRAIN_H5AD)
    val = sc.read_h5ad(VAL_H5AD)
    return ad.concat([train, val])


def _cell_type_subsets(adata: ad.AnnData):
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    for ct in sorted(values.unique()):
        yield ct, common._slug_cell_type(ct), adata[(values == ct).to_numpy()].copy()


def _gene_features(adata: ad.AnnData) -> Dict[str, Dict[str, float]]:
    """Calculate mean expression, variance and dropout directly from adata.X."""
    X = adata.X
    n_cells = X.shape[0]
    if n_cells == 0:
        raise ValueError("Cannot calculate gene features for an empty cell type.")

    if issparse(X):
        X = X.tocsr()
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        variance = np.maximum(mean_sq - mean * mean, 0.0)

        # Number of non-zero entries per gene.
        nonzero = np.asarray(X.getnnz(axis=0)).ravel()
        dropout = 1.0 - nonzero / float(n_cells)
    else:
        X = np.asarray(X)
        mean = np.nanmean(X, axis=0)
        variance = np.nanvar(X, axis=0)
        dropout = np.mean(X == 0, axis=0)

    genes = [str(g) for g in adata.var.index]
    return {
        gene: {
            "expression_mean": float(mean[i]),
            "expression_variance": float(variance[i]),
            "dropout_rate": float(dropout[i]),
        }
        for i, gene in enumerate(genes)
    }


def _ols_residualize(
    genes,
    attention_scores: Dict[str, float],
    features: Dict[str, Dict[str, float]],
) -> Tuple[list, dict]:
    """Fit OLS across genes and return all-gene rows plus model metadata."""
    rows = []
    for gene in genes:
        gene = str(gene)
        rows.append(
            {
                "gene": gene,
                "attention_score": float(attention_scores.get(gene, 0.0)),
                **features[gene],
            }
        )

    y = np.asarray([r["attention_score"] for r in rows], dtype=float)
    feature_names = [
        "expression_mean",
        "expression_variance",
        "dropout_rate",
    ]
    X_raw = np.asarray(
        [[r[name] for name in feature_names] for r in rows],
        dtype=float,
    )

    finite = np.isfinite(y) & np.isfinite(X_raw).all(axis=1)
    if finite.sum() <= len(feature_names) + 1:
        raise ValueError(
            f"Not enough finite genes for OLS: {int(finite.sum())} available."
        )

    X_fit_raw = X_raw[finite]
    y_fit = y[finite]

    # Standardization is only for numerical conditioning. With an intercept
    # and no regularization, it does not change fitted values or residuals.
    means = X_fit_raw.mean(axis=0)
    stds = X_fit_raw.std(axis=0)
    active = stds > 0

    Z = np.zeros_like(X_fit_raw)
    Z[:, active] = (X_fit_raw[:, active] - means[active]) / stds[active]

    design = np.column_stack([np.ones(Z.shape[0]), Z])
    beta, *_ = np.linalg.lstsq(design, y_fit, rcond=None)

    predicted = np.full_like(y, np.nan, dtype=float)
    Z_all = np.zeros_like(X_raw)
    Z_all[:, active] = (X_raw[:, active] - means[active]) / stds[active]
    predicted[finite] = np.column_stack(
        [np.ones(int(finite.sum())), Z_all[finite]]
    ) @ beta

    residual = y - predicted

    ss_res = float(np.sum((y_fit - design @ beta) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    for i, row in enumerate(rows):
        row["predicted_attention"] = float(predicted[i])
        row["residual_attention"] = float(residual[i])

    valid_residuals = [
        r["residual_attention"]
        for r in rows
        if np.isfinite(r["residual_attention"])
    ]
    resid_mean = float(np.mean(valid_residuals))
    resid_std = float(np.std(valid_residuals))

    for row in rows:
        if not np.isfinite(row["residual_attention"]):
            row["residual_zscore"] = float("nan")
        elif resid_std > 0:
            row["residual_zscore"] = (
                row["residual_attention"] - resid_mean
            ) / resid_std
        else:
            row["residual_zscore"] = 0.0

    # Positive residual first: genes whose attention is higher than expected
    # from expression/variance/dropout.
    rows.sort(
        key=lambda r: (
            -r["residual_attention"]
            if np.isfinite(r["residual_attention"])
            else float("inf")
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    metadata = {
        "n_genes": len(rows),
        "n_genes_used": int(finite.sum()),
        "predictors": feature_names,
        "standardized_predictors": True,
        "intercept": float(beta[0]),
        "coefficients_standardized": {
            name: float(beta[i + 1]) if active[i] else 0.0
            for i, name in enumerate(feature_names)
        },
        "r_squared": r2,
    }
    return rows, metadata


def _write_csv(rows: list, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "gene",
        "attention_score",
        "predicted_attention",
        "residual_attention",
        "residual_zscore",
        "expression_mean",
        "expression_variance",
        "dropout_rate",
    ]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def task_attention_residualized_ols(
    results_dir: Path,
    device: torch.device,
) -> None:
    """Run OLS residualization for every cell type and save all genes."""
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset()
    model = common._load_zero_shot_model(vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets(adata):
        print(f"\n# attention_residualized_ols | {ct} #")

        # This deliberately uses the exact same attention computation as the
        # existing attention_zero_shot task.
        pt = common._tokenise(sub, vocab)
        means, gate = common._compute_attn_per_gene_per_condition(
            model,
            pt,
            vocab,
            model_configs["nheads"],
            device,
            n_cls=2,
            use_predictions=False,
        )
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue

        attention_scores = common._gene_scores(
            means,
            lambda mean_pd, mean_ctrl: mean_pd - mean_ctrl,
        )

        # Same gene universe as the attention task: genes present in the
        # scGPT vocabulary and in this cell-type AnnData.
        in_vocab = np.array([g in vocab for g in sub.var.index])
        genes = [str(g) for g in sub.var.index[in_vocab]]
        sub_for_features = sub[:, in_vocab].copy()

        features = _gene_features(sub_for_features)
        rows, metadata = _ols_residualize(
            genes,
            attention_scores,
            features,
        )

        rows.sort(
            key=lambda r: (
                -r["predicted attention"]
                if np.isfinite(r["predicted_attention"])
                else float("inf")
            )
        )

        out_dir = results_dir / "attention_residualized_ols" / slug
        out_dir.mkdir(parents=True, exist_ok=True)

        out_csv = out_dir / "residualized_genes.csv"
        _write_csv(rows, out_csv)

        # Useful audit information; the requested all-gene result remains the
        # CSV above.
        with open(out_dir / "ols_summary.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(
            f"  wrote {out_csv} ({len(rows)} genes; "
            f"R^2={metadata['r_squared']:.4f})"
        )

    del model
    torch.cuda.empty_cache()

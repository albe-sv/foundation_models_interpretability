#!/usr/bin/env python
"""
Interpretability outputs for the scGPT model:

  1. Zero-shot CLS-attention signed score (PD vs Control) -> top-30 genes.
  2. Integrated Gradients on the LoRA fine-tuned scGPT model -> top-30 genes.

Runs on the full dataset (train + val combined)

Run separately for each cell type.
"""

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import anndata as ad
import numpy as np
import scanpy as sc
import torch

from pipeline import config as C
from pipeline.steps import _step7_common as common
from pipeline.steps import step7_explainability_ig as ig
from pipeline.utils.scgpt_model import load_model_configs, load_vocab

DATA_DIR = _HERE / "data"
TRAIN_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_train.h5ad"
VAL_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_val.h5ad"
PHASE2_FULL_MODEL = DATA_DIR / "step6_phase2" / "best_full_model.pt"

TOP_N = 30

# scGPT's binning spreads tied expression values with the global numpy RNG
# (scgpt.preprocess._digitize), so without this the top-30 genes change on
# every run. Record it alongside the results: it fixes one draw, it does not
# make the ranking independent of the draw.
SEED = 42

# This function is pretty simple, but is because previously I had another type of score
def _signed_score(mean_pd: float, mean_ctrl: float) -> float:
    return mean_pd - mean_ctrl

# Write the csv with the top N genes 
def _write_top_n(score_map: dict, out_csv: Path, n: int = TOP_N) -> None:
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked[:n], start=1):
            w.writerow([r, g, s])

def _load_full_dataset() -> ad.AnnData:
    train = sc.read_h5ad(TRAIN_H5AD)
    val = sc.read_h5ad(VAL_H5AD)
    return ad.concat([train, val])

# This function is just for getting the cell-type subsets
# Return the name of the cell type (ct), a slug for the cell type (slug, corrected name)
# and the subset of the adata for that cell type (sub)
def _cell_type_subsets(adata):
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    for ct in sorted(values.unique()):
        yield ct, common._slug_cell_type(ct), adata[(values == ct).to_numpy()].copy()

def task_attention(results_dir: Path, device: torch.device) -> None:
    # See scgpt_model.py for this functions description
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset()
    model = common._load_zero_shot_model(vocab, model_configs, device)
    for ct, slug, sub in _cell_type_subsets(adata):
        print(f"\n# attention | {ct} #")
        # Tokenise of the vocav of our data for the current cell type
        pt = common._tokenise(sub, vocab)
        # Attention per gene per condition
        means, gate = common._compute_attn_per_gene_per_condition(
            model, pt, vocab, model_configs["nheads"], device, n_cls=2,
            use_predictions=False,
        )
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue

        # Average of attention per gene per condition
        scores = common._gene_scores(means, _signed_score)

        # Write the csv with the top N genes
        _write_top_n(scores, results_dir / "attention_zero_shot" / slug / "top30_genes.csv")
    del model
    torch.cuda.empty_cache()

def task_ig_lora(results_dir: Path, device: torch.device) -> None:
    # See scgpt_model.py for this functions description
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset()
    model = common._load_lora_phase2_model(PHASE2_FULL_MODEL, vocab, model_configs, device)
    for ct, slug, sub in _cell_type_subsets(adata):
        print(f"\n=== ig_lora | {ct} ===")
        # Tokenise of the vocav of our data for the current cell type
        pt = common._tokenise(sub, vocab)
        # IG per gene per condition
        mean_abs, gate = ig._compute_ig_importance(model, pt, vocab, device)
        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}")
            continue
        
        # Write the csv with the top N genes
        _write_top_n(mean_abs, results_dir / "ig_lora" / slug / "top30_genes.csv")
    del model
    torch.cuda.empty_cache()

TASKS = {"attention": task_attention, "ig_lora": task_ig_lora}

def main():
    # This seed is for the binning of tied expression values in scGPT's preprocess
    np.random.seed(SEED)

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    args = p.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for name in (list(TASKS) if args.task == "all" else [args.task]):
        print(f"\n# {name} #")
        TASKS[name](args.results_dir, device)

if __name__ == "__main__":
    main()

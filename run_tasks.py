#!/usr/bin/env python
"""
Interpretability outputs for the scGPT model:

  1. Zero-shot CLS-attention signed score (PD vs Control) -> top-30 genes.
  2. Zero-shot CLS-attention CSSI-max score -> all genes.
  3. Integrated Gradients on the LoRA fine-tuned scGPT model -> top-30 genes.

Runs on the full dataset (train + val combined), separately for each cell type.

For GPU-bound model tasks, --gpus can be used to run independent cell-type
workers on multiple GPUs. Each worker owns one copy of the model and writes
only the cell types assigned to it, so no DataParallel/Captum interaction is
needed.
"""

import argparse
import csv
import sys
from multiprocessing import get_context
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import anndata as ad
import numpy as np
import scanpy as sc
import torch

from pipeline import config as C
from pipeline.postprocess.enrichment import task_enrichment
from pipeline.postprocess.enrichment_summary import task_enrichment_summary
from pipeline.postprocess.gene_matrix import task_gene_matrix
from pipeline.postprocess.gwas_similarity import task_gwas_similarity
from pipeline.steps import _step7_common as common
from pipeline.steps import attention_cssi_leiden as cssi
from pipeline.steps import step7_explainability_ig as ig
from pipeline.utils.scgpt_model import load_model_configs, load_vocab


DATA_DIR = _HERE / "data"
TRAIN_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_train.h5ad"
VAL_H5AD = DATA_DIR / "mixed_split" / "Mixed_HVG_1000_pc_noXY_noMT_mixed_val.h5ad"
PHASE2_FULL_MODEL = DATA_DIR / "step6_phase2" / "best_full_model.pt"

TOP_N = 30
SEED = 42
CSSI_N_NEIGHBORS = 15
CSSI_TARGET_K = 6
CSSI_MIN_CELLS_PER_CONDITION = 20


def _signed_score(mean_pd: float, mean_ctrl: float) -> float:
    return mean_pd - mean_ctrl


def _write_top_n(score_map: dict, out_csv: Path, n: int = TOP_N) -> None:
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked[:n], start=1):
            w.writerow([r, g, s])

def _write_all_scores(score_map: dict, out_csv: Path) -> None:
    """Write all genes ranked by signed score (used for CSSI output)."""
    ranked = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "gene", "score"])
        for r, (g, s) in enumerate(ranked, start=1):
            w.writerow([r, g, s])


def _load_full_dataset() -> ad.AnnData:
    train = sc.read_h5ad(TRAIN_H5AD)
    val = sc.read_h5ad(VAL_H5AD)
    return ad.concat([train, val])


def _cell_type_subsets(adata, allowed_cell_types=None):
    """Yield cell-type subsets, optionally restricted to assigned cell types."""
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    allowed = None if allowed_cell_types is None else set(allowed_cell_types)

    for ct in sorted(values.unique()):
        if allowed is not None and ct not in allowed:
            continue
        mask = (values == ct).to_numpy()
        yield ct, common._slug_cell_type(ct), adata[mask].copy()


def _all_cell_types_with_counts(adata):
    """Return cell types sorted by descending cell count."""
    values = adata.obs[C.CELL_TYPE_COL].astype(str)
    counts = values.value_counts().to_dict()
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))


def _split_cell_types(adata, n_workers):
    """
    Greedily balance cell types by number of cells.

    A whole cell type stays on one GPU. This avoids merging partial results and
    guarantees that each output CSV is produced by exactly one worker.
    """
    workers = [[] for _ in range(n_workers)]
    loads = [0] * n_workers

    for ct, count in _all_cell_types_with_counts(adata):
        worker = min(range(n_workers), key=loads.__getitem__)
        workers[worker].append(ct)
        loads[worker] += count

    return workers, loads


def task_attention(results_dir: Path, device: torch.device, cell_types=None) -> None:
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset()
    model = common._load_zero_shot_model(vocab, model_configs, device)

    for ct, slug, sub in _cell_type_subsets(adata, cell_types):
        print(f"\n# attention | {ct} | {device} #", flush=True)
        pt = common._tokenise(sub, vocab)

        means, cssi_scores, gate, embeddings, strata, meta = cssi.compute_attention_cssi(
            model,
            pt,
            vocab,
            model_configs["nheads"],
            device,
            n_cls=2,
            n_neighbors=CSSI_N_NEIGHBORS,
            target_k=CSSI_TARGET_K,
            min_cells_per_condition=CSSI_MIN_CELLS_PER_CONDITION,
            random_state=SEED,
        )
        print(
            f"  CSSI: {meta.get('n_clusters', 1)} Leiden strata "
            f"(resolution={meta.get('resolution', 0):.4g}), "
            f"{meta.get('n_embedding_dims', embeddings.shape[1])}-D embeddings",
            flush=True,
        )

        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}", flush=True)
            continue

        pooled_scores = common._gene_scores(means, _signed_score)
        out_dir = results_dir / "attention_zero_shot" / slug
        _write_top_n(pooled_scores, out_dir / "top30_genes.csv")
        _write_all_scores(cssi_scores, out_dir / "cssi_genes.csv")

        np.save(out_dir / "cell_embeddings.npy", embeddings)
        import pandas as pd
        pd.DataFrame({"cell_index": np.arange(len(strata)), "leiden": strata}).to_csv(
            out_dir / "leiden_clusters.csv", index=False
        )
        cssi.save_metadata(out_dir / "cssi_metadata.json", meta)

        print(f"  wrote {out_dir / 'top30_genes.csv'}", flush=True)
        print(f"  wrote {out_dir / 'cssi_genes.csv'} ({len(cssi_scores)} genes)", flush=True)

    del model
    torch.cuda.empty_cache()

def task_ig_lora(results_dir: Path, device: torch.device, cell_types=None) -> None:
    vocab, model_configs = load_vocab(), load_model_configs()
    adata = _load_full_dataset()

    print(f"[{device}] loading LoRA model...", flush=True)
    model = common._load_lora_phase2_model(
        PHASE2_FULL_MODEL, vocab, model_configs, device
    )

    for ct, slug, sub in _cell_type_subsets(adata, cell_types):
        print(f"\n=== ig_lora | {ct} | {device} ===", flush=True)

        pt = common._tokenise(sub, vocab)

        mean_abs, gate = ig._compute_ig_importance(
            model, pt, vocab, device
        )

        if not gate["ok"]:
            print(f"  skipped: {gate['reason']}", flush=True)
            continue

        _write_top_n(
            mean_abs,
            results_dir / "ig_lora" / slug / "top30_genes.csv",
        )

    del model
    torch.cuda.empty_cache()


def _gpu_worker(task_name, gpu_id, cell_types, results_dir):
    """
    One independent process per GPU.

    This is intentionally process-based rather than torch.nn.DataParallel.
    Captum IntegratedGradients performs backward passes with respect to `vals`,
    so independent model replicas are safer and simpler here.
    """
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Prevent each worker from consuming all CPU threads.
    torch.set_num_threads(1)
    np.random.seed(SEED)

    print(
        f"[worker GPU {gpu_id}] starting {task_name}; "
        f"{len(cell_types)} cell types",
        flush=True,
    )

    if task_name == "ig_lora":
        task_ig_lora(results_dir, device, cell_types)
    elif task_name == "attention":
        task_attention(results_dir, device, cell_types)
    else:
        raise ValueError(f"Multi-GPU worker does not support task: {task_name}")

    print(f"[worker GPU {gpu_id}] finished", flush=True)


MODEL_TASKS = {"attention": task_attention, "ig_lora": task_ig_lora}

POSTPROCESS_TASKS = {
    "enrichment": task_enrichment,
    "enrichment_summary": task_enrichment_summary,
    "gwas_similarity": task_gwas_similarity,
    "gene_matrix": task_gene_matrix,
}

TASKS = {**MODEL_TASKS, **POSTPROCESS_TASKS}


def _parse_gpus(value):
    try:
        gpus = [int(x.strip()) for x in value.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--gpus must be a comma-separated list, e.g. 0,1"
        ) from exc

    if not gpus:
        raise argparse.ArgumentTypeError("--gpus cannot be empty")
    if len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError("--gpus contains duplicates")
    if any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError("GPU ids must be >= 0")

    return gpus


def _run_multi_gpu(task_name, gpus, results_dir):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "--gpus requires CUDA, but torch.cuda.is_available() is False"
        )

    n_available = torch.cuda.device_count()
    invalid = [g for g in gpus if g >= n_available]
    if invalid:
        raise RuntimeError(
            f"Requested GPU(s) {invalid}, but PyTorch only sees "
            f"{n_available} GPU(s)."
        )

    # Used only to calculate a balanced cell-type assignment.
    print("Loading dataset to balance cell types...", flush=True)
    adata = _load_full_dataset()

    assignments, loads = _split_cell_types(adata, len(gpus))

    print("\nMulti-GPU assignment:", flush=True)
    for gpu_id, cell_types, load in zip(gpus, assignments, loads):
        print(
            f"  GPU {gpu_id}: {len(cell_types)} cell types, "
            f"{load:,} cells",
            flush=True,
        )
        for ct in cell_types:
            print(f"    - {ct}", flush=True)

    # Important for CUDA: do not fork a process after CUDA has been initialized.
    ctx = get_context("spawn")
    processes = []

    for gpu_id, cell_types in zip(gpus, assignments):
        if not cell_types:
            print(f"WARNING: GPU {gpu_id} received no cell types.", flush=True)
            continue

        p = ctx.Process(
            target=_gpu_worker,
            args=(task_name, gpu_id, cell_types, results_dir),
        )
        p.start()
        processes.append((gpu_id, p))

    failed = []
    for gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((gpu_id, p.exitcode))

    if failed:
        details = ", ".join(
            f"GPU {gpu}: exit code {code}" for gpu, code in failed
        )
        raise RuntimeError(f"One or more GPU workers failed: {details}")


def main():
    np.random.seed(SEED)

    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS) + ["all"], default="all")

    # Original single-GPU option kept for backwards compatibility.
    p.add_argument("--device", default="cuda:0")

    # New multi-GPU option.
    p.add_argument(
        "--gpus",
        type=_parse_gpus,
        default=None,
        help="Comma-separated GPU ids, e.g. 0,1",
    )

    p.add_argument("--results-dir", type=Path, default=_HERE / "results")
    args = p.parse_args()

    if args.gpus is not None:
        if args.task == "all":
            # Run model-bound tasks one after another; each task is internally
            # distributed across the requested GPUs. Postprocess tasks are still
            # intentionally separate because they depend on completed outputs.
            for model_task in MODEL_TASKS:
                print(f"\n# {model_task} | multi-GPU #", flush=True)
                _run_multi_gpu(model_task, args.gpus, args.results_dir)
            return

        if args.task not in MODEL_TASKS:
            raise SystemExit(
                "--gpus is supported with --task attention, --task ig_lora, or --task all"
            )

        _run_multi_gpu(args.task, args.gpus, args.results_dir)
        return

    # Original single-GPU behavior.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for name in (list(MODEL_TASKS) if args.task == "all" else [args.task]):
        print(f"\n# {name} #")
        if name in MODEL_TASKS:
            TASKS[name](args.results_dir, device)
        else:
            TASKS[name](args.results_dir)


if __name__ == "__main__":
    main()

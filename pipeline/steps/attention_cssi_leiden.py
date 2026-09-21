"""Paper-faithful CSSI for the gene-level zero-shot attention pipeline.

The paper's CSSI stratification is:
  1. model cell embeddings from the residual stream
  2. cosine k-NN graph (k=15 default)
  3. Leiden community detection, targeting ~5-7 strata
  4. score attention within each stratum
  5. CSSI-max aggregation across strata

Your existing output is gene-level (PD-control attention), not TF-target edges,
so the scoring step is the gene-level analogue of the paper's per-edge scoring.
"""

from collections import defaultdict
import json
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from einops import rearrange

from . import _step7_common as common
from .. import config as C

DEFAULT_N_NEIGHBORS = 15
DEFAULT_TARGET_K = 6
DEFAULT_RESOLUTIONS = tuple(np.geomspace(0.1, 2.5, 25))


def _forward_cls_attention_with_embedding(
    model: torch.nn.Module,
    batch_pt: Dict[str, torch.Tensor],
    vocab,
    n_head: int,
    device: torch.device,
    n_cls: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CLS attention, predictions, and residual-stream CLS embedding."""
    pad_id = vocab[C.PAD_TOKEN]
    gids = batch_pt["gene_ids"].to(device)
    vals = batch_pt["values"].to(device)
    mask = gids.eq(pad_id)

    capture = common._LayerInputCapture(
        model.transformer_encoder.layers[common.NUM_ATTN_LAYERS]
    )
    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            output = model(
                gids,
                vals,
                src_key_padding_mask=mask,
                batch_labels=None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
            )
    finally:
        capture.remove()

    embs = capture.captured.float()
    self_attn = model.transformer_encoder.layers[common.NUM_ATTN_LAYERS].self_attn
    qkv = rearrange(
        self_attn.Wqkv(embs),
        "b s (three h d) -> b s three h d",
        three=3,
        h=n_head,
    )
    q = qkv[:, :, 0, :, :]
    k = qkv[:, :, 1, :, :]
    q_cls = q[:, 0:1, :, :].permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 3, 1)
    cls_scores = (q_cls @ k_t).squeeze(2)
    cls_attn = common._cls_attn_rank_norm_head_avg(cls_scores, mask)

    cls_output = output["cls_output"].float()
    if n_cls == 1:
        logits = cls_output.squeeze(1)
        preds = (logits > 0).long()
    else:
        preds = cls_output.argmax(dim=1)

    # CLS token entering the final attention layer: residual-stream cell embedding.
    cell_embeddings = embs[:, 0, :]
    return cls_attn, preds, cell_embeddings


def _leiden_from_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    target_k: int = DEFAULT_TARGET_K,
    resolutions=DEFAULT_RESOLUTIONS,
    random_state: int = 42,
):
    """Build cosine k-NN graph and choose Leiden resolution closest to target_k."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3:
        return np.zeros(embeddings.shape[0], dtype=int), {
            "n_neighbors": min(n_neighbors, max(1, embeddings.shape[0] - 1)),
            "resolution": 0.0,
            "n_clusters": 1,
        }

    graph_adata = ad.AnnData(X=embeddings)
    nn = min(n_neighbors, embeddings.shape[0] - 1)
    sc.pp.neighbors(
        graph_adata,
        n_neighbors=nn,
        use_rep="X",
        metric="cosine",
        random_state=random_state,
    )

    candidates = []
    for resolution in resolutions:
        sc.tl.leiden(
            graph_adata,
            resolution=float(resolution),
            random_state=random_state,
            key_added="_leiden",
            directed=False,
        )
        labels = graph_adata.obs["_leiden"].astype(int).to_numpy()
        n_clusters = int(np.unique(labels).size)
        candidates.append((abs(n_clusters - target_k), -n_clusters, float(resolution), labels.copy()))

    _, _, best_resolution, best_labels = min(candidates, key=lambda x: (x[0], x[1], x[2]))
    meta = {
        "n_neighbors": nn,
        "resolution": best_resolution,
        "n_clusters": int(np.unique(best_labels).size),
        "target_k": target_k,
        "metric": "cosine",
        "embedding_source": "scGPT residual-stream CLS embedding entering transformer layer 11",
    }
    return best_labels.astype(int), meta


def _finalize_means(sum_by_gid, cnt_by_gid, vocab):
    means = {0: {}, 1: {}}
    for cond in (0, 1):
        for gid, total in sum_by_gid[cond].items():
            name = vocab.lookup_tokens([gid])[0]
            if name in C.SPECIAL_TOKENS:
                continue
            count = cnt_by_gid[cond][gid]
            means[cond][name] = total / count if count else 0.0
    return means


def compute_attention_cssi(
    model,
    pt: Dict[str, torch.Tensor],
    vocab,
    n_head: int,
    device: torch.device,
    n_cls: int = 2,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    target_k: int = DEFAULT_TARGET_K,
    min_cells_per_condition: int = 20,
    random_state: int = 42,
    batch_size: int = 16,
):
    """Compute pooled and CSSI-max gene scores plus embeddings/Leiden labels."""
    n_cells = pt["gene_ids"].shape[0]
    pad_id = vocab[C.PAD_TOKEN]
    special_ids = {vocab[t] for t in C.SPECIAL_TOKENS if t in vocab}

    # Cache per-cell attention/genes and residual embeddings so clustering only
    # requires one model pass.
    attn_batches = []
    gid_batches = []
    embeddings_batches = []
    labels_all = []
    n_per_cond = {0: 0, 1: 0}

    for start in range(0, n_cells, batch_size):
        end = min(start + batch_size, n_cells)
        batch = {k: v[start:end] for k, v in pt.items()}
        cls_attn, _, embeddings = _forward_cls_attention_with_embedding(
            model, batch, vocab, n_head, device, n_cls
        )
        attn_batches.append(cls_attn.cpu().numpy())
        gid_batches.append(batch["gene_ids"].cpu().numpy())
        embeddings_batches.append(embeddings.cpu().numpy())
        labels = batch["condition_labels"].cpu().numpy()
        labels_all.append(labels)
        for label in labels:
            n_per_cond[int(label)] += 1

    all_attn = np.concatenate(attn_batches, axis=0)
    all_gids = np.concatenate(gid_batches, axis=0)
    embeddings = np.concatenate(embeddings_batches, axis=0).astype(np.float32)
    labels = np.concatenate(labels_all, axis=0).astype(int)

    if n_per_cond[0] == 0 or n_per_cond[1] == 0:
        gate = {
            "ok": False,
            "reason": f"ground-truth counts: control={n_per_cond[0]}, pd={n_per_cond[1]}",
            "mode": "labels",
        }
        return {}, {}, gate, embeddings, np.zeros(n_cells, dtype=int), {}

    strata, cluster_meta = _leiden_from_embeddings(
        embeddings,
        n_neighbors=n_neighbors,
        target_k=target_k,
        random_state=random_state,
    )

    sum_by_gid = {0: {}, 1: {}}
    cnt_by_gid = {0: {}, 1: {}}
    sum_by_stratum = defaultdict(lambda: {0: {}, 1: {}})
    cnt_by_stratum = defaultdict(lambda: {0: {}, 1: {}})

    for i in range(n_cells):
        cond = int(labels[i])
        stratum = int(strata[i])
        row = all_attn[i]
        cell_gids = all_gids[i]

        for pos in range(1, len(cell_gids)):
            gid = int(cell_gids[pos])
            if gid == pad_id or gid in special_ids:
                continue
            value = float(row[pos])

            sum_by_gid[cond][gid] = sum_by_gid[cond].get(gid, 0.0) + value
            cnt_by_gid[cond][gid] = cnt_by_gid[cond].get(gid, 0) + 1
            sum_by_stratum[stratum][cond][gid] = (
                sum_by_stratum[stratum][cond].get(gid, 0.0) + value
            )
            cnt_by_stratum[stratum][cond][gid] = (
                cnt_by_stratum[stratum][cond].get(gid, 0) + 1
            )

    pooled_means = _finalize_means(sum_by_gid, cnt_by_gid, vocab)
    cssi_scores = {}
    valid_strata = 0

    for stratum in sorted(sum_by_stratum):
        cells_control = int(np.sum((strata == stratum) & (labels == 0)))
        cells_pd = int(np.sum((strata == stratum) & (labels == 1)))
        if min(cells_control, cells_pd) < min_cells_per_condition:
            continue
        valid_strata += 1

        stratum_means = _finalize_means(
            sum_by_stratum[stratum], cnt_by_stratum[stratum], vocab
        )
        scores = common._gene_scores(
            stratum_means,
            lambda mean_pd, mean_ctrl: mean_pd - mean_ctrl,
        )
        for gene, score in scores.items():
            # CSSI-max is the maximum signed stratum-specific contrast.
            if gene not in cssi_scores or score > cssi_scores[gene]:
                cssi_scores[gene] = float(score)

    if valid_strata == 0:
        # No valid PD/control-balanced stratum: do not invent a contrast.
        cssi_scores = common._gene_scores(
            pooled_means,
            lambda mean_pd, mean_ctrl: mean_pd - mean_ctrl,
        )

    cluster_meta["valid_strata"] = valid_strata
    cluster_meta["n_cells"] = int(n_cells)
    cluster_meta["n_embedding_dims"] = int(embeddings.shape[1])
    return pooled_means, cssi_scores, {
        "ok": True,
        "reason": "",
        "mode": "labels",
    }, embeddings, strata, cluster_meta


def save_metadata(path, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(metadata, handle, indent=2)

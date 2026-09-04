# Foundation Models Interpretability

Run `run_tasks.py` for running the interpretability tasks:

- **attention**: forward-passes cells through the zero-shot pretrained scGPT model and captures the CLS-token attention to each gene. Genes are scored by `mean_attention_pd - mean_attention_control`, so the sign shows which condition attends to that gene more.
- **ig_lora**: runs Integrated Gradients (Captum library) on a classification model after the foundation model, scoring each gene by its mean absolute attribution to the PD-classification logit.

Run `--task attention` or `--task ig_lora` to run only one of them. `--task all` (default) runs both.

## Post-hoc gene-set analyses

These read the top-30 CSVs produced above; run them individually after `attention`/`ig_lora`, in this order:

| Task | What it does | Output |
|---|---|---|
| `enrichment` | g:Profiler GO/KEGG/REAC enrichment | `results/enrichment/` |
| `enrichment_summary` | term-count/IC stats + pairwise Mann-Whitney tests over the enrichment above | `results/enrichment_summary/` |
| `gwas_similarity` | GO3 gene-level similarity (lin/SimRel/wang or any other distance) vs GWAS Catalog PD/AD/CAD gene sets | `results/gwas_similarity/` |
| `gene_matrix` | gene x gene GO3 similarity matrices vs the Parkinson GWAS gene set | `results/gene_matrix/` |

```
python run_tasks.py --task <name>
```

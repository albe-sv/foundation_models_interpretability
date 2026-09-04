# Foundation Models Interpretability

Run `run_tasks.py` for running the two interpretability tasks:

- **attention**: zero-shot CLS-attention signed score (PD vs control) -> top-30 genes per cell type.
- **ig_lora**: Integrated Gradients on the LoRA fine-tuned model -> top-30 genes per cell type.

Run `--task attention` or `--task ig_lora` to run only one of them.

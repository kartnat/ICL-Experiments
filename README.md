# ICL-Experiments

Reference implementations for experiments related to in-context learning (ICL) multiclass classification.

## Getting Started

### Environment Setup

```bash
conda env create -f environment.yml
conda activate icl-experiments
```

### Running Training

```bash
python train_onehot.py --config in-context_onehot.yaml
```

Config files live in `conf/`. `in-context_onehot.yaml` inherits from `tiny_onehot.yaml` (model architecture defaults) and adds training hyperparameters and experiment settings.

### Configuration Options

**`loss`** — selects the loss function:
- `mse` — mean squared error
- `softmax` — cross entropy (applied to softmax outputs)
- `multiclass` — multiclass hinge loss

**`run_altpaper_adam`** — when `true`, switches the optimizer to Adam and uses the alternative input sampling from `sample_xsys_onehot_adam` (see `samplers.py` for how that input is constructed).

## File Overview

| File | Description |
|------|-------------|
| `models.py` | Wrappers for GPT-style transformers used across both classes of experiments |
| `samplers.py` | Input/output sampling functions |
| `tasks.py` | Loss functions: MSE, cross entropy, multiclass hinge loss |
| `modeling_gpt2_onehot.py` | Standalone GPT-2-style transformer backbone |
| `train_onehot.py` | Main training script |
| `schema.py` | Configuration schema (parsed via Quinine) |

### Experiment Classes

Files with the `_onehot` suffix correspond to experiments where the input is formatted in the style described in ["How do nonlinear transformers learn and generalize in in-context learning?"](https://arxiv.org/abs/2402.15449) — each basis element is one-hot encoded before being fed to the model.

Files without the `_onehot` suffix correspond to experiments with input constructed from raw sign pairs as the basis vectors.

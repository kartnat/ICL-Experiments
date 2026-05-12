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
python train.py --config in-context.yaml
```

Runs the experiment described in Section 1 of the paper, along with its variants detailed in the first part of Section 2 (with some modifications to the linear layers in `modeling_gpt2.py`).

```bash
python train_onehot.py --config in-context_onehot.yaml
```

Runs the experiments detailed in the latter part of Section 2.

Config files live in `conf/` and are composable via inheritance — for example, `in-context_onehot.yaml` inherits from `tiny_onehot.yaml` (model architecture defaults) and layers on training hyperparameters and experiment settings.

### Configuration Options

**`loss`** — selects the loss function:
- `mse` — mean squared error
- `softmax` — cross entropy (applied to softmax outputs)
- `multiclass` — multiclass hinge loss (onehot experiments only)

**`use_fullbatch`** — when `true`, builds the complete dataset once before training using `build_all_combinations` (all 3^10 label sequences across all four sign-pair tasks) and reuses it every step as a fixed full-batch gradient descent. When `false` (default), draws a fresh minibatch each step via `sample_xsys_multi_signpair`.

**`run_altpaper_adam`** — when `true`, switches the optimizer to Adam and uses the alternative input sampling from `sample_xsys_onehot_adam` (see `samplers.py` for how that input is constructed).

## File Overview

| File | Description |
|------|-------------|
| `models.py` | Wrappers for GPT-style transformers used across both classes of experiments |
| `samplers.py` | Input/output sampling functions |
| `tasks.py` | Loss functions: MSE, cross entropy, multiclass hinge loss |
| `modeling_gpt2_onehot.py` / `modeling_gpt2.py` | Standalone GPT-2-style transformer backbones |
| `train_onehot.py` / `train.py` | Main training scripts |
| `schema.py` | Configuration schema (parsed via Quinine) |

### Experiment Classes

Files with the `_onehot` suffix correspond to experiments where the input is formatted in a more traditional ICL style, similar to that described in ["How do nonlinear transformers learn and generalize in in-context learning?"](https://arxiv.org/abs/2402.15607) — here it is simplified so that each basis element is one-hot encoded before being fed to the model, as are labels.

Files without the `_onehot` suffix correspond to experiments with input constructed from raw sign pairs as the basis vectors, in a format similar to ["What Can Transformers Learn In-Context? A Case Study of Simple Function Classes"](https://arxiv.org/abs/2208.01066).

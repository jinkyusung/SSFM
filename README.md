# SSFM for CIFAR-10

A self-contained Strong Stochastic Flow Map implementation for CIFAR-10.

## Environment

Create a conda environment and install all Python dependencies with pip:

```bash
./setup_ssfm_cifar_env.sh auto
conda activate ssfm-cifar
```

Use `cuda12` or `cpu` instead of `auto` to force a backend. To install the
additional PyTorch dependencies needed for FID evaluation:

```bash
./setup_ssfm_cifar_env.sh auto ssfm-cifar metrics
```

## Training

```bash
python ssfm_cifar.py
```

CIFAR-10 is downloaded to `data/`, and checkpoints are written to
`checkpoints/`. Set `WANDB_MODE=offline` to train without logging into W&B.

## Evaluation

```bash
python eval_fid.py
python bm_consistency.py
```

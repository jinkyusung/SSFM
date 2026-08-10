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

Submit it through Slurm with the default training entry point:

```bash
mkdir -p slurm  # Slurm opens the output path before run.sh starts
sbatch run.sh
```

Or select any Python entry point and pass its remaining arguments through:

```bash
sbatch run.sh eval_fid.py
sbatch run.sh some_file.py --some-option value
```

`run.sh` uses the `ssfm-cifar` conda environment created by the setup script.
If the setup script was given a different environment name, export the same
name when submitting:

```bash
./setup_ssfm_cifar_env.sh cuda12 my-ssfm-env
sbatch --export=ALL,SSFM_CONDA_ENV=my-ssfm-env run.sh ssfm_cifar.py
```

CIFAR-10 is downloaded to `data/`, and checkpoints are written to
`checkpoints/`. Set `WANDB_MODE=offline` to train without logging into W&B.

## Evaluation

```bash
python eval_fid.py
python bm_consistency.py
```

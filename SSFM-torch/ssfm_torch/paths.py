"""Canonical repository and per-implementation filesystem locations."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPOSITORY_DIR = PROJECT_DIR.parent

# Dataset assets are shared by all implementations.
DATA_DIR = REPOSITORY_DIR / "data"

# Runtime outputs are isolated inside this implementation.
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
MODEL_DIR = PROJECT_DIR / "models"
SLURM_DIR = PROJECT_DIR / "slurm"
WANDB_DIR = PROJECT_DIR / "wandb"


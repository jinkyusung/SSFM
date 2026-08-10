#!/usr/bin/env python3
"""Visualize shared-path versus independently sampled Brownian trajectories."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from ssfm_torch.brownian import sample_levy
from ssfm_torch.checkpoint import load_checkpoint
from ssfm_torch.data import cifar10
from ssfm_torch.losses import UncertaintyMLP
from ssfm_torch.model import build_model
from ssfm_torch.paths import DATA_DIR, PROJECT_DIR
from ssfm_torch.sampling import integrate_shared_fine_path


def to_image(image: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> np.ndarray:
    return np.asarray((image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--output", default=str(PROJECT_DIR / "bm_consistency.png"))
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    shape = (3, 32, 32)
    _, mean, std = cifar10(args.data_dir)
    mean, std = mean.reshape(shape), std.reshape(shape)

    model = build_model().to(device)
    ema_model = build_model().to(device)
    uncertainty = UncertaintyMLP(128).to(device)
    load_checkpoint(args.checkpoint, model, ema_model, uncertainty, map_location=device)
    ema_model.eval()
    y_initial = torch.randn(shape, device=device, generator=generator)
    step_counts = (2, 4, 8, 16, 32)
    n_fine = max(step_counts)
    fine = [
        sample_levy(
            (1e-5 - 1.0) / n_fine,
            (1, *shape),
            device=device,
            generator=generator,
        )
        for _ in range(n_fine)
    ]
    shared = [
        integrate_shared_fine_path(
            ema_model,
            y_initial,
            1.0,
            1e-5,
            steps,
            n_fine,
            fine_increments=fine,
        )
        for steps in step_counts
    ]
    independent = [
        integrate_shared_fine_path(
            ema_model, y_initial, 1.0, 1e-5, n_fine, n_fine, generator=generator
        )
        for _ in step_counts
    ]

    figure, axes = plt.subplots(2, len(step_counts), figsize=(10, 4))
    for column, steps in enumerate(step_counts):
        axes[0, column].imshow(to_image(shared[column], mean, std))
        axes[0, column].set_title(f"n={steps}")
        axes[1, column].imshow(to_image(independent[column], mean, std))
        for row in range(2):
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Same W")
    axes[1, 0].set_ylabel("Different W")
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

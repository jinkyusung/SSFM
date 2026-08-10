#!/usr/bin/env python3
"""Evaluate generated CIFAR-10 samples with FID."""

from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from ssfm_torch.checkpoint import load_checkpoint
from ssfm_torch.data import cifar10
from ssfm_torch.losses import UncertaintyMLP
from ssfm_torch.metrics import FIDStats, InceptionFeatureExtractor, compute_fid, compute_statistics, preprocess_images
from ssfm_torch.model import build_model
from ssfm_torch.paths import DATA_DIR
from ssfm_torch.sampling import sample_flow_map_batched


def real_statistics(dataset: torch.Tensor, cache_path: str, device: torch.device) -> FIDStats:
    if os.path.exists(cache_path):
        return FIDStats.load(cache_path)
    extractor = InceptionFeatureExtractor(device)
    features = [extractor(batch) for batch in preprocess_images(dataset)]
    stats = compute_statistics(np.concatenate(features))
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    stats.save(cache_path)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--samples", type=int, default=50_000)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    dataset, data_mean, data_std = cifar10(args.data_dir)
    shape = (3, 32, 32)
    dataset = dataset.reshape(-1, *shape)
    data_mean, data_std = data_mean.reshape(shape), data_std.reshape(shape)
    real_images = ((dataset * data_std + data_mean).clamp(0, 1) * 2 - 1)
    stats = real_statistics(
        real_images, os.path.join(args.data_dir, "cifar10_inception_stats.npz"), device
    )

    model = build_model().to(device)
    ema_model = build_model().to(device)
    uncertainty = UncertaintyMLP(128).to(device)
    load_checkpoint(args.checkpoint, model, ema_model, uncertainty, map_location=device)
    ema_model.eval()
    for step_size in (1 / 16, 1 / 8, 1 / 4, 1 / 2):
        n_steps = max(1, math.ceil((1.0 - 1e-5) / step_size))
        samples = sample_flow_map_batched(
            ema_model,
            1e-5,
            args.samples,
            shape,
            step_size,
            n_steps,
            args.sample_batch_size,
            device=device,
        )
        samples = ((samples * data_std + data_mean).clamp(0, 1) * 2 - 1)
        fid = compute_fid(samples, stats, device=device)
        print(f"FID (h={step_size}, steps={n_steps}): {fid:.2f}")


if __name__ == "__main__":
    main()

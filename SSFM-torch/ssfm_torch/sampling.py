"""SSFM sampling and image-grid helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

from .brownian import LevyArea, combine_levy, sample_levy
from .model import EulerMaruyamaFlowMap


@torch.inference_mode()
def sample_flow_map(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    n_samples: int,
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    del step_size  # retained for API compatibility; n_steps defines the fixed grid.
    if n_steps < 1 or n_samples < 1:
        raise ValueError("n_steps and n_samples must be positive")
    if device is None:
        device = next(flow_map.parameters()).device
    dtype = next(flow_map.parameters()).dtype
    was_training = flow_map.training
    flow_map.eval()
    y = torch.randn(
        (n_samples, *data_shape), device=device, dtype=dtype, generator=generator
    )
    time_grid = torch.linspace(1.0, t_eps, n_steps + 1, device=device, dtype=dtype)
    for index in range(n_steps):
        s, t = time_grid[index], time_grid[index + 1]
        levy = sample_levy(
            t - s,
            tuple(y.shape),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        y = flow_map(y, s, t, levy.W, levy.H, levy.K)
    flow_map.train(was_training)
    return y


@torch.inference_mode()
def sample_flow_map_batched(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    n_samples: int,
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
    batch_size: int = 256,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
    output_device: torch.device | str = "cpu",
) -> Tensor:
    batches = []
    for start in range(0, n_samples, batch_size):
        size = min(batch_size, n_samples - start)
        batch = sample_flow_map(
            flow_map,
            t_eps,
            size,
            data_shape,
            step_size,
            n_steps,
            device=device,
            generator=generator,
        )
        batches.append(batch.to(output_device))
    return torch.cat(batches, dim=0)


def _merge_group(increments: Sequence[LevyArea]) -> LevyArea:
    if not increments:
        raise ValueError("cannot merge an empty increment group")
    result = increments[0]
    for increment in increments[1:]:
        result = combine_levy(result, increment)
    return result


@torch.inference_mode()
def integrate_shared_fine_path(
    flow_map: EulerMaruyamaFlowMap,
    y_initial: Tensor,
    s: float,
    t: float,
    n_steps: int,
    n_fine_steps: int,
    *,
    generator: torch.Generator | None = None,
    fine_increments: Sequence[LevyArea] | None = None,
) -> Tensor:
    """Integrate using coarsenings of one shared fine Brownian path."""
    if n_fine_steps % n_steps:
        raise ValueError("n_fine_steps must be divisible by n_steps")
    y, unbatched = (y_initial.unsqueeze(0), True) if y_initial.ndim == 3 else (y_initial, False)
    dtype, device = y.dtype, y.device
    fine_dt = (t - s) / n_fine_steps
    if fine_increments is None:
        fine = [
            sample_levy(
                fine_dt, tuple(y.shape), device=device, dtype=dtype, generator=generator
            )
            for _ in range(n_fine_steps)
        ]
    else:
        if len(fine_increments) != n_fine_steps:
            raise ValueError("fine_increments length must equal n_fine_steps")
        fine = list(fine_increments)
    group_size = n_fine_steps // n_steps
    time_grid = torch.linspace(s, t, n_steps + 1, device=device, dtype=dtype)
    for index in range(n_steps):
        levy = _merge_group(fine[index * group_size : (index + 1) * group_size])
        y = flow_map(y, time_grid[index], time_grid[index + 1], levy.W, levy.H, levy.K)
    return y.squeeze(0) if unbatched else y


def make_sample_grid(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    data_shape: tuple[int, ...],
    step_sizes: Sequence[float],
    data_mean: Tensor,
    data_std: Tensor,
    n_cols: int = 8,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(step_sizes), n_cols, figsize=(2 * n_cols, 2 * len(step_sizes)), squeeze=False
    )
    for row, size in enumerate(step_sizes):
        n_steps = max(1, math.ceil((1.0 - t_eps) / size))
        samples = sample_flow_map(
            flow_map,
            t_eps,
            n_cols,
            data_shape,
            size,
            n_steps,
            device=device,
            generator=generator,
        ).cpu()
        samples = (samples * data_std[None] + data_mean[None]).clamp(0, 1)
        for column in range(n_cols):
            axes[row, column].imshow(np.asarray(samples[column].permute(1, 2, 0)))
            axes[row, column].axis("off")
            if column == 0:
                axes[row, column].set_ylabel(f"h={size}", fontsize=10)
    figure.tight_layout()
    return figure

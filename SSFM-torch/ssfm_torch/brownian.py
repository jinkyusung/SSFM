"""Torch-native Brownian increments and space-time-time Levy areas.

For an interval of length ``dt``, W, H, and K are independent centred
Gaussians with variances ``|dt|``, ``|dt| / 12``, and ``|dt| / 720``.
``combine_levy`` applies Chen's relation, allowing two adjacent increments to
be reused as one increment without resampling. This is the consistency needed
by SSFM distillation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class LevyArea:
    dt: Tensor
    W: Tensor
    H: Tensor
    K: Tensor

    def to(self, *args, **kwargs) -> "LevyArea":
        return LevyArea(
            self.dt.to(*args, **kwargs),
            self.W.to(*args, **kwargs),
            self.H.to(*args, **kwargs),
            self.K.to(*args, **kwargs),
        )


def _expand_dt(dt: Tensor, target_ndim: int) -> Tensor:
    while dt.ndim < target_ndim:
        dt = dt.unsqueeze(-1)
    return dt


def sample_levy(
    dt: Tensor | float,
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> LevyArea:
    """Sample W, H, K with the same interval law used by the reference model.

    ``dt`` may be scalar or have leading batch dimensions. ``shape`` is the
    complete output shape, so a batched ``dt`` must broadcast to it.
    """
    dt = torch.as_tensor(dt, device=device, dtype=dtype)
    scale = _expand_dt(dt.abs().sqrt(), len(shape))
    randn = lambda: torch.randn(shape, device=device, dtype=dtype, generator=generator)
    return LevyArea(
        dt=dt,
        W=randn() * scale,
        H=randn() * (scale / math.sqrt(12.0)),
        K=randn() * (scale / math.sqrt(720.0)),
    )


def combine_levy(first: LevyArea, second: LevyArea) -> LevyArea:
    """Combine adjacent Levy increments using the exact Chen relation."""
    a = _expand_dt(first.dt, first.W.ndim)
    b = _expand_dt(second.dt, second.W.ndim)
    total = a + b
    if torch.any(total == 0):
        raise ValueError("Combined Levy interval must have non-zero duration")

    cross = b * first.W - a * second.W
    bar_h = a * first.H + b * second.H + 0.5 * cross
    bar_k = (
        a.square() * first.K
        + b.square() * second.K
        + 0.5 * a * b * (first.H - second.H)
        + ((b - a) / 12.0) * cross
    )
    return LevyArea(
        dt=(first.dt + second.dt),
        W=first.W + second.W,
        H=bar_h / total,
        K=bar_k / total.square(),
    )


def sample_split_levy(
    dt: Tensor | float,
    shape: tuple[int, ...],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> tuple[LevyArea, LevyArea, LevyArea]:
    """Sample two half intervals and return both halves plus their exact merge."""
    half_dt = torch.as_tensor(dt, device=device, dtype=dtype) / 2
    first = sample_levy(
        half_dt, shape, device=device, dtype=dtype, generator=generator
    )
    second = sample_levy(
        half_dt, shape, device=device, dtype=dtype, generator=generator
    )
    return first, second, combine_levy(first, second)


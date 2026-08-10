"""Variance-preserving diffusion used by SSFM."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class VPDiffusion:
    reverse_eta: float
    beta_min: float
    beta_max: float

    def beta(self, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t)
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def log_mean_coeff(self, t: Tensor | float) -> Tensor:
        t = torch.as_tensor(t)
        return -0.5 * (
            self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t.square()
        )

    def forward_drift(self, t: Tensor | float, y: Tensor) -> Tensor:
        t = torch.as_tensor(t, device=y.device, dtype=y.dtype)
        while t.ndim < y.ndim:
            t = t.unsqueeze(-1)
        return -0.5 * self.beta(t) * y

    def forward_diffusion(self, t: Tensor | float, *, like: Tensor | None = None) -> Tensor:
        if like is None:
            t = torch.as_tensor(t)
        else:
            t = torch.as_tensor(t, device=like.device, dtype=like.dtype)
        return self.beta(t).sqrt()

    def forward_sample(
        self,
        t: Tensor | float,
        y0: Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        t = torch.as_tensor(t, device=y0.device, dtype=y0.dtype)
        while t.ndim < y0.ndim:
            t = t.unsqueeze(-1)
        alpha = self.log_mean_coeff(t).exp()
        sigma = (1.0 - alpha.square()).sqrt()
        if noise is None:
            noise = torch.randn(
                y0.shape, device=y0.device, dtype=y0.dtype, generator=generator
            )
        yt = alpha * y0 + sigma * noise
        score = -noise / sigma
        return yt, score

    def reverse_drift(self, t: Tensor | float, y: Tensor, score: Tensor) -> Tensor:
        g = self.forward_diffusion(t, like=y)
        while g.ndim < y.ndim:
            g = g.unsqueeze(-1)
        return self.forward_drift(t, y) - 0.5 * (1 + self.reverse_eta**2) * g.square() * score

    def reverse_diffusion(self, t: Tensor | float, *, like: Tensor | None = None) -> Tensor:
        return self.reverse_eta * self.forward_diffusion(t, like=like)

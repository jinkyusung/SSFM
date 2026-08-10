"""Uncertainty-weighted SSFM training objectives."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .brownian import sample_levy, sample_split_levy
from .diffusion import VPDiffusion
from .model import EulerMaruyamaFlowMap


class UncertaintyMLP(nn.Module):
    def __init__(self, fourier_dim: int):
        super().__init__()
        self.register_buffer("freqs_s", torch.randn(fourier_dim))
        self.register_buffer("phases_s", torch.rand(fourier_dim))
        self.register_buffer("freqs_t", torch.randn(fourier_dim))
        self.register_buffer("phases_t", torch.rand(fourier_dim))
        self.linear = nn.Linear(2 * fourier_dim, 1)

    def forward(self, s: Tensor, t: Tensor) -> Tensor:
        s = s.reshape(-1, 1)
        t = t.reshape(-1, 1)
        feat_s = torch.cos(2 * torch.pi * (self.freqs_s * s + self.phases_s))
        feat_t = torch.cos(2 * torch.pi * (self.freqs_t * t + self.phases_t))
        return self.linear(torch.cat((feat_s, feat_t), dim=-1)).squeeze(-1)


class UncertaintyScoreLoss(nn.Module):
    def __init__(self, diffusion: VPDiffusion, dt: float, t_eps: float = 1e-5):
        super().__init__()
        self.diffusion = diffusion
        self.dt = dt
        self.t_eps = t_eps

    def forward(
        self,
        flow_map: EulerMaruyamaFlowMap,
        y0_batch: Tensor,
        u_mlp: UncertaintyMLP,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        batch_size = y0_batch.shape[0]
        device, dtype = y0_batch.device, y0_batch.dtype
        h = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        h = self.dt / 2 + h * (self.dt / 2)
        s = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        s = self.t_eps + h + s * (1.0 - self.t_eps - h)

        ys, score = self.diffusion.forward_sample(s, y0_batch, generator=generator)
        s = s.clamp(self.t_eps, 1.0 - self.t_eps)
        t = (s - h).clamp(self.t_eps, 1.0)
        drift = self.diffusion.reverse_drift(s, ys, score)
        diffusion_scale = self.diffusion.reverse_diffusion(s, like=ys)
        while diffusion_scale.ndim < ys.ndim:
            diffusion_scale = diffusion_scale.unsqueeze(-1)

        levy = sample_levy(
            t - s,
            tuple(y0_batch.shape),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        target = ys + drift * (t - s)[:, None, None, None] + diffusion_scale * levy.W
        prediction = flow_map(ys, s, t, levy.W, levy.H, levy.K)
        raw_loss = (prediction - target.detach()).square().flatten(1).sum(1) / h
        uncertainty = u_mlp(s, t)
        return (torch.exp(-uncertainty) * raw_loss + uncertainty).mean()


class UncertaintyDistillationLoss(nn.Module):
    def __init__(
        self, diffusion: VPDiffusion, dt: float, h_max: float, t_eps: float = 1e-5
    ):
        super().__init__()
        self.diffusion = diffusion
        self.dt = dt
        self.h_max = h_max
        self.t_eps = t_eps

    def forward(
        self,
        flow_map: EulerMaruyamaFlowMap,
        ema_flow_map: EulerMaruyamaFlowMap,
        y0_batch: Tensor,
        u_mlp: UncertaintyMLP,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        batch_size = y0_batch.shape[0]
        device, dtype = y0_batch.device, y0_batch.dtype
        h = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        h = self.dt + h * (self.h_max - self.dt)
        s = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        s = self.t_eps + h + s * (1.0 - self.t_eps - h)
        ys, _ = self.diffusion.forward_sample(s, y0_batch, generator=generator)
        s = s.clamp(self.t_eps, 1.0 - self.t_eps)
        t = (s - h).clamp(self.t_eps, 1.0)
        middle = (s - h / 2).clamp(self.t_eps, 1.0 - self.t_eps)

        first, second, whole = sample_split_levy(
            t - s,
            tuple(y0_batch.shape),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        with torch.no_grad():
            y_middle = ema_flow_map(ys, s, middle, first.W, first.H, first.K)
            target = ema_flow_map(
                y_middle, middle, t, second.W, second.H, second.K
            )
        prediction = flow_map(ys, s, t, whole.W, whole.H, whole.K)
        raw_loss = (prediction - target).square().flatten(1).sum(1) / h
        uncertainty = u_mlp(s, t)
        return (torch.exp(-uncertainty) * raw_loss + uncertainty).mean()


class UncertaintyJointLoss(nn.Module):
    def __init__(
        self,
        score_loss: UncertaintyScoreLoss,
        distill_loss: UncertaintyDistillationLoss,
        u_mlp: UncertaintyMLP,
        eta: float = 0.75,
    ):
        super().__init__()
        if not 0 < eta < 1:
            raise ValueError("eta must be strictly between zero and one")
        self.score_loss = score_loss
        self.distill_loss = distill_loss
        self.u_mlp = u_mlp
        self.eta = eta

    def forward(
        self,
        flow_map: EulerMaruyamaFlowMap,
        ema_flow_map: EulerMaruyamaFlowMap,
        y0_batch: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        n_score = int(y0_batch.shape[0] * self.eta)
        if n_score == 0 or n_score == y0_batch.shape[0]:
            raise ValueError("batch size must allocate at least one sample to each loss")
        score = self.score_loss(
            flow_map, y0_batch[:n_score], self.u_mlp, generator=generator
        )
        distillation = self.distill_loss(
            flow_map,
            ema_flow_map,
            y0_batch[n_score:],
            self.u_mlp,
            generator=generator,
        )
        return self.eta * score + (1 - self.eta) * distillation


@torch.no_grad()
def ema_update(model: nn.Module, ema_model: nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        source = model_state[name]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(source, alpha=1 - decay)
        else:
            ema_value.copy_(source)

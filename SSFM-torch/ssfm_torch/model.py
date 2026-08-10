"""EDM2-style networks and Euler--Maruyama flow map."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_batch(x: Tensor) -> tuple[Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim != 4:
        raise ValueError(f"expected CHW or NCHW tensor, got shape {tuple(x.shape)}")
    return x, False


def _batch_time(time: Tensor | float, x: Tensor) -> Tensor:
    time = torch.as_tensor(time, device=x.device, dtype=x.dtype)
    if time.ndim == 0:
        return time.expand(x.shape[0])
    if time.ndim == 1 and time.shape[0] == x.shape[0]:
        return time
    raise ValueError(f"time must be scalar or shape ({x.shape[0]},), got {tuple(time.shape)}")


def downsample(x: Tensor) -> Tensor:
    return F.avg_pool2d(x, kernel_size=2, stride=2)


def upsample(x: Tensor) -> Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups: int, eps: float = 1e-4):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        if c % self.num_groups:
            raise ValueError(f"{c} channels are not divisible by {self.num_groups} groups")
        grouped = x.reshape(n, self.num_groups, c // self.num_groups, h, w)
        rms = grouped.square().mean(dim=(2, 3, 4), keepdim=True).sqrt() + self.eps
        return (grouped / rms).reshape(n, c, h, w)


def pixel_norm(x: Tensor, eps: float = 1e-4) -> Tensor:
    return x / (x.square().mean(dim=-1, keepdim=True).sqrt() + eps)


class FourierTimeEmbedding(nn.Module):
    def __init__(self, fourier_dim: int, time_dim: int):
        super().__init__()
        self.register_buffer("freqs", torch.randn(fourier_dim))
        self.register_buffer("phases", torch.rand(fourier_dim))
        self.linear1 = nn.Linear(fourier_dim, time_dim, bias=False)
        self.linear2 = nn.Linear(time_dim, time_dim, bias=False)

    def forward(self, time: Tensor) -> Tensor:
        x = torch.cos(2 * torch.pi * (self.freqs * time[:, None] + self.phases))
        return self.linear2(F.silu(self.linear1(x)))


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        num_groups: int = 8,
        resample: str = "none",
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if resample not in {"none", "down", "up"}:
            raise ValueError(f"unknown resampling mode: {resample}")
        self.norm1 = RMSGroupNorm(min(num_groups, in_channels))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.time_proj = nn.Linear(time_dim, out_channels, bias=False)
        self.norm2 = RMSGroupNorm(min(num_groups, out_channels))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.skip_conv = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else None
        )
        self.resample = resample
        self.dropout = nn.Dropout(dropout_rate)

    def _resample(self, x: Tensor) -> Tensor:
        if self.resample == "down":
            return downsample(x)
        if self.resample == "up":
            return upsample(x)
        return x

    def forward(self, x: Tensor, time_embedding: Tensor) -> Tensor:
        h = self._resample(F.silu(self.norm1(x)))
        h = self.conv1(h)
        scale = self.time_proj(time_embedding)[:, :, None, None]
        h = self.norm2(h) * (1 + scale)
        h = self.conv2(self.dropout(F.silu(h)))
        skip = self._resample(x)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)
        return skip + h


class CosineAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if num_heads < 1 or channels % num_heads:
            raise ValueError("channels must be divisible by a positive num_heads")
        self.qkv_proj = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.num_heads = num_heads

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        head_dim = c // self.num_heads
        qkv = self.qkv_proj(x).reshape(n, 3, self.num_heads, head_dim, h * w)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)
        q, k = pixel_norm(q), pixel_norm(k)
        attention = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        out = torch.matmul(attention.softmax(dim=-1), v)
        out = out.permute(0, 1, 3, 2).reshape(n, c, h, w)
        return x + self.out_proj(out)


class EDM2UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        resolution: int = 28,
        base_channels: int = 64,
        channel_mult: Sequence[int] = (1, 2, 4),
        num_groups: int = 8,
        dropout_rate: float = 0.0,
        attn_resolutions: Sequence[int] | None = None,
        head_dim: int = 64,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        if not channel_mult:
            raise ValueError("channel_mult must not be empty")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.input_res = 2 ** math.ceil(math.log2(max(resolution, 2)))
        self.pad = (self.input_res - resolution) // 2

        time_dim = 4 * base_channels
        channels = [base_channels * multiplier for multiplier in channel_mult]
        self.s_embed = FourierTimeEmbedding(base_channels, time_dim)
        self.t_embed = FourierTimeEmbedding(base_channels, time_dim)
        self.time_fuse1 = nn.Linear(2 * time_dim, time_dim, bias=False)
        self.time_fuse2 = nn.Linear(time_dim, time_dim, bias=False)
        self.init_conv = nn.Conv2d(in_channels + 1, channels[0], 3, padding=1, bias=False)

        attention_resolutions = (
            {self.input_res // (2**i) for i in range(len(channels))}
            if attn_resolutions is None
            else set(attn_resolutions)
        )

        self.enc_blocks = nn.ModuleList()
        self.enc_attns = nn.ModuleList()
        previous_channels = channels[0]
        current_resolution = self.input_res
        for level, out_ch in enumerate(channels):
            resample = "down" if level > 0 else "none"
            if level > 0:
                current_resolution //= 2
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for block_index in range(num_res_blocks):
                in_ch = previous_channels if block_index == 0 else out_ch
                mode = resample if block_index == 0 else "none"
                blocks.append(
                    ResBlock(in_ch, out_ch, time_dim, num_groups, mode, dropout_rate)
                )
                attentions.append(
                    CosineAttention(out_ch, out_ch // head_dim)
                    if current_resolution in attention_resolutions
                    else nn.Identity()
                )
            self.enc_blocks.append(blocks)
            self.enc_attns.append(attentions)
            previous_channels = out_ch

        middle_resolution = self.input_res // (2 ** (len(channels) - 1))
        self.mid_block = ResBlock(
            channels[-1], channels[-1], time_dim, num_groups, dropout_rate=dropout_rate
        )
        self.mid_attn = (
            CosineAttention(channels[-1], channels[-1] // head_dim)
            if middle_resolution in attention_resolutions
            else nn.Identity()
        )

        self.dec_blocks = nn.ModuleList()
        self.dec_attns = nn.ModuleList()
        reverse_channels = list(reversed(channels))
        decoder_resolution = middle_resolution
        for level, in_level_channels in enumerate(reverse_channels):
            out_level_channels = (
                reverse_channels[level + 1]
                if level + 1 < len(reverse_channels)
                else in_level_channels
            )
            has_upsample = level < len(reverse_channels) - 1
            final_mode = "up" if has_upsample else "none"
            output_resolution = decoder_resolution * 2 if has_upsample else decoder_resolution
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for block_index in range(num_res_blocks):
                if block_index == 0:
                    in_ch, out_ch, mode = 2 * in_level_channels, in_level_channels, "none"
                elif block_index == num_res_blocks - 1:
                    in_ch, out_ch, mode = in_level_channels, out_level_channels, final_mode
                else:
                    in_ch, out_ch, mode = in_level_channels, in_level_channels, "none"
                blocks.append(
                    ResBlock(in_ch, out_ch, time_dim, num_groups, mode, dropout_rate)
                )
                attention_resolution = (
                    output_resolution if block_index == num_res_blocks - 1 else decoder_resolution
                )
                attentions.append(
                    CosineAttention(out_ch, out_ch // head_dim)
                    if attention_resolution in attention_resolutions
                    else nn.Identity()
                )
            self.dec_blocks.append(blocks)
            self.dec_attns.append(attentions)
            decoder_resolution = output_resolution

        self.final_norm = RMSGroupNorm(min(num_groups, channels[0]))
        self.final_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1, bias=False)

    def forward(self, x: Tensor, s: Tensor | float, t: Tensor | float) -> Tensor:
        x, unbatched = _as_batch(x)
        if x.shape[1:] != (self.in_channels, self.resolution, self.resolution):
            raise ValueError(
                f"expected (*, {self.in_channels}, {self.resolution}, {self.resolution}), "
                f"got {tuple(x.shape)}"
            )
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, self.pad, self.pad))
        ones = torch.ones(
            (x.shape[0], 1, self.input_res, self.input_res), device=x.device, dtype=x.dtype
        )
        x = torch.cat((x, ones), dim=1)

        s_embedding = self.s_embed(_batch_time(s, x))
        t_embedding = self.t_embed(_batch_time(t, x))
        time_embedding = self.time_fuse2(
            F.silu(self.time_fuse1(torch.cat((s_embedding, t_embedding), dim=-1)))
        )
        h = self.init_conv(x)
        skips: list[Tensor] = []
        for blocks, attentions in zip(self.enc_blocks, self.enc_attns):
            for block, attention in zip(blocks, attentions):
                h = attention(block(h, time_embedding))
            skips.append(h)
        h = self.mid_attn(self.mid_block(h, time_embedding))
        for blocks, attentions in zip(self.dec_blocks, self.dec_attns):
            h = torch.cat((h, skips.pop()), dim=1)
            for block, attention in zip(blocks, attentions):
                h = attention(block(h, time_embedding))
        h = self.final_conv(F.silu(self.final_norm(h)))
        if self.pad > 0:
            h = h[:, :, self.pad : self.pad + self.resolution, self.pad : self.pad + self.resolution]
        return h.squeeze(0) if unbatched else h


class EDM2EMStepModel(nn.Module):
    def __init__(self, drift_net: EDM2UNet, diffusion_net: EDM2UNet):
        super().__init__()
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net

    def drift(
        self, ys: Tensor, s: Tensor | float, t: Tensor | float, W: Tensor, H: Tensor, K: Tensor
    ) -> Tensor:
        dim = 0 if ys.ndim == 3 else 1
        return self.drift_net(torch.cat((ys, W, H, K), dim=dim), s, t)

    def diffusion(
        self, s: Tensor | float, t: Tensor | float, W: Tensor, H: Tensor, K: Tensor
    ) -> Tensor:
        dim = 0 if W.ndim == 3 else 1
        return self.diffusion_net(torch.cat((W, H, K), dim=dim), s, t)


class EulerMaruyamaFlowMap(nn.Module):
    def __init__(self, step_model: EDM2EMStepModel):
        super().__init__()
        self.step_model = step_model

    def forward(
        self,
        ys: Tensor,
        s: Tensor | float,
        t: Tensor | float,
        W: Tensor,
        H: Tensor,
        K: Tensor,
    ) -> Tensor:
        delta = torch.as_tensor(t, device=ys.device, dtype=ys.dtype) - torch.as_tensor(
            s, device=ys.device, dtype=ys.dtype
        )
        while delta.ndim < ys.ndim:
            delta = delta.unsqueeze(-1)
        return ys + delta * self.step_model.drift(ys, s, t, W, H, K) + W * self.step_model.diffusion(s, t, W, H, K)


def build_model(
    *,
    in_channels: int = 3,
    resolution: int = 32,
    base_channels: int = 128,
    channel_mult: Sequence[int] = (2, 2, 2),
    num_groups: int = 8,
    dropout_rate: float = 0.13,
    attn_resolutions: Sequence[int] = (16,),
    head_dim: int = 64,
    num_res_blocks: int = 4,
    diff_base_channels: int = 64,
    diff_channel_mult: Sequence[int] = (2, 2, 2),
    diff_attn_resolutions: Sequence[int] = (16,),
) -> EulerMaruyamaFlowMap:
    drift_net = EDM2UNet(
        4 * in_channels,
        in_channels,
        resolution,
        base_channels,
        channel_mult,
        num_groups,
        dropout_rate,
        attn_resolutions,
        head_dim,
        num_res_blocks,
    )
    diffusion_net = EDM2UNet(
        3 * in_channels,
        in_channels,
        resolution,
        diff_base_channels,
        diff_channel_mult,
        num_groups,
        0.0,
        diff_attn_resolutions,
        head_dim,
        num_res_blocks,
    )
    return EulerMaruyamaFlowMap(EDM2EMStepModel(drift_net, diffusion_net))


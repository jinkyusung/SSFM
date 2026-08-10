"""Self-contained Strong Stochastic Flow Map training for CIFAR-10.

This is the single source of truth for the CIFAR-10 model and training code.
Run it from the repository root with::

    python ssfm_cifar.py
"""

from abc import abstractmethod
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "1.0")
import math
import pickle
import tarfile
import urllib.request

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

Y = Float[Array, " *d"]
BatchY = Float[Array, "batch *d"]


class AbstractDiffusion(eqx.Module):
    reverse_eta: float

    @abstractmethod
    def forward_drift(self, t: Float[Array, ""], y: Y) -> Y:
        pass

    @abstractmethod
    def forward_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        pass

    @abstractmethod
    def forward_sample(
        self, t: Float[Array, ""], y0: Y, key: PRNGKeyArray
    ) -> tuple[Y, Y]:
        pass

    def reverse_drift(
        self,
        t: Float[Array, ""],
        y: Y,
        score: Y,
    ) -> Y:
        g = self.forward_diffusion(t)
        return self.forward_drift(t, y) - 0.5 * (1 + self.reverse_eta**2) * g**2 * score

    def reverse_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return self.reverse_eta * self.forward_diffusion(t)


class VPDiffusion(AbstractDiffusion):
    beta_min: float
    beta_max: float

    def _beta(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _log_mean_coeff(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return -0.5 * (self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2)

    def forward_drift(self, t: Float[Array, ""], y: Y) -> Y:
        return -0.5 * self._beta(t) * y

    def forward_diffusion(self, t: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sqrt(self._beta(t))

    def forward_sample(
        self, t: Float[Array, ""], y0: Y, key: PRNGKeyArray
    ) -> tuple[Y, Y]:
        alpha = jnp.exp(self._log_mean_coeff(t))
        sigma = jnp.sqrt(1.0 - alpha**2)
        noise = jax.random.normal(key, y0.shape)
        yt = alpha * y0 + sigma * noise
        score = -noise / sigma
        return yt, score

class AbstractEMStepModel(eqx.Module):
    @abstractmethod
    def drift(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass

    @abstractmethod
    def diffusion(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass


class AbstractFlowMap(eqx.Module):
    @abstractmethod
    def __call__(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        pass


class EulerMaruyamaFlowMap(AbstractFlowMap):
    step_model: AbstractEMStepModel

    def __call__(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key: PRNGKeyArray | None = None,
    ) -> Y:
        if key is not None:
            key_drift, key_diff = jax.random.split(key)
        else:
            key_drift = key_diff = None
        return (
            ys
            + (t - s) * self.step_model.drift(ys, s, t, W, H, K, key=key_drift)
            + W * self.step_model.diffusion(s, t, W, H, K, key=key_diff)
        )

def _downsample(x):
    c, h, w = x.shape
    x = x.reshape(c, h // 2, 2, w // 2, 2)
    return x.mean(axis=(2, 4))


def _upsample(x):
    x = jnp.repeat(x, 2, axis=1)
    x = jnp.repeat(x, 2, axis=2)
    return x


class RMSGroupNorm(eqx.Module):
    num_groups: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __init__(self, num_groups, eps=1e-4):
        self.num_groups = num_groups
        self.eps = eps

    def __call__(self, x):
        c, h, w = x.shape
        g = self.num_groups
        x = x.reshape(g, c // g, h, w)
        rms = jnp.sqrt(jnp.mean(x**2, axis=(1, 2, 3), keepdims=True)) + self.eps
        x = x / rms
        return x.reshape(c, h, w)


def pixel_norm(x, eps=1e-4):
    return x / (jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True)) + eps)


class FourierTimeEmbedding(eqx.Module):
    freqs: jax.Array
    phases: jax.Array
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, fourier_dim, time_dim, *, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.freqs = jax.random.normal(k1, (fourier_dim,))
        self.phases = jax.random.uniform(k2, (fourier_dim,))
        self.linear1 = eqx.nn.Linear(fourier_dim, time_dim, use_bias=False, key=k3)
        self.linear2 = eqx.nn.Linear(time_dim, time_dim, use_bias=False, key=k4)

    def __call__(self, time):
        freqs = jax.lax.stop_gradient(self.freqs)
        phases = jax.lax.stop_gradient(self.phases)
        x = jnp.cos(2 * jnp.pi * (freqs * time + phases))
        x = self.linear1(x)
        x = jax.nn.silu(x)
        x = self.linear2(x)
        return x


class ResBlock(eqx.Module):
    norm1: RMSGroupNorm
    conv1: eqx.nn.Conv2d
    time_proj: eqx.nn.Linear
    norm2: RMSGroupNorm
    conv2: eqx.nn.Conv2d
    skip_conv: eqx.nn.Conv2d | None
    resample: str
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        in_ch,
        out_ch,
        time_dim,
        num_groups=8,
        resample="none",
        dropout_rate=0.0,
        *,
        key,
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = RMSGroupNorm(min(num_groups, in_ch))
        self.conv1 = eqx.nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k1
        )
        self.time_proj = eqx.nn.Linear(time_dim, out_ch, use_bias=False, key=k2)
        self.norm2 = RMSGroupNorm(min(num_groups, out_ch))
        self.conv2 = eqx.nn.Conv2d(
            out_ch, out_ch, kernel_size=3, padding=1, use_bias=False, key=k3
        )
        self.skip_conv = (
            eqx.nn.Conv2d(in_ch, out_ch, kernel_size=1, use_bias=False, key=k4)
            if in_ch != out_ch
            else None
        )
        self.resample = resample
        self.dropout = eqx.nn.Dropout(p=dropout_rate)

    def __call__(self, x, t_emb, *, key=None):
        h = self.norm1(x)
        h = jax.nn.silu(h)
        if self.resample == "down":
            h = _downsample(h)
        elif self.resample == "up":
            h = _upsample(h)
        h = self.conv1(h)

        scale = self.time_proj(t_emb)
        h = self.norm2(h)
        h = h * (1 + scale[:, None, None])
        h = jax.nn.silu(h)
        h = self.dropout(h, key=key)
        h = self.conv2(h)

        skip = x
        if self.resample == "down":
            skip = _downsample(skip)
        elif self.resample == "up":
            skip = _upsample(skip)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        return skip + h


class CosineAttention(eqx.Module):
    qkv_proj: eqx.nn.Conv2d
    out_proj: eqx.nn.Conv2d
    num_heads: int = eqx.field(static=True)

    def __init__(self, channels, num_heads, *, key):
        k1, k2 = jax.random.split(key)
        self.qkv_proj = eqx.nn.Conv2d(
            channels, 3 * channels, kernel_size=1, use_bias=False, key=k1
        )
        self.out_proj = eqx.nn.Conv2d(
            channels, channels, kernel_size=1, use_bias=False, key=k2
        )
        self.num_heads = num_heads

    def __call__(self, x):
        c, h, w = x.shape
        head_dim = c // self.num_heads
        seq_len = h * w

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(3, self.num_heads, head_dim, seq_len)
        qkv = jnp.transpose(qkv, (0, 1, 3, 2))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = pixel_norm(q)
        k = pixel_norm(k)

        attn = jnp.matmul(q, jnp.transpose(k, (0, 2, 1))) / jnp.sqrt(head_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.matmul(attn, v)

        out = jnp.transpose(out, (0, 2, 1))
        out = out.reshape(c, h, w)
        out = self.out_proj(out)

        return x + out


class EDM2_UNet(eqx.Module):
    in_channels: int = eqx.field(static=True)
    out_channels: int = eqx.field(static=True)
    resolution: int = eqx.field(static=True)
    input_res: int = eqx.field(static=True)
    pad: int = eqx.field(static=True)
    s_embed: FourierTimeEmbedding
    t_embed: FourierTimeEmbedding
    time_fuse1: eqx.nn.Linear
    time_fuse2: eqx.nn.Linear
    init_conv: eqx.nn.Conv2d
    enc_blocks: list
    enc_attns: list
    mid_block: ResBlock
    mid_attn: CosineAttention | None
    dec_blocks: list
    dec_attns: list
    final_norm: RMSGroupNorm
    final_conv: eqx.nn.Conv2d

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        resolution=28,
        base_channels=64,
        channel_mult=(1, 2, 4),
        num_groups=8,
        dropout_rate=0.0,
        attn_resolutions=None,
        head_dim=64,
        num_res_blocks=4,
        *,
        key,
    ):
        n_keys = 3 * num_res_blocks * len(channel_mult) + 20
        keys = jax.random.split(key, n_keys)
        ki = iter(keys)

        self.in_channels = in_channels
        time_dim = 4 * base_channels
        channels = [base_channels * m for m in channel_mult]

        self.s_embed = FourierTimeEmbedding(base_channels, time_dim, key=next(ki))
        self.t_embed = FourierTimeEmbedding(base_channels, time_dim, key=next(ki))
        self.time_fuse1 = eqx.nn.Linear(
            2 * time_dim, time_dim, use_bias=False, key=next(ki)
        )
        self.time_fuse2 = eqx.nn.Linear(
            time_dim, time_dim, use_bias=False, key=next(ki)
        )
        self.init_conv = eqx.nn.Conv2d(
            in_channels + 1,
            channels[0],
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=next(ki),
        )

        input_res = 2 ** math.ceil(math.log2(max(resolution, 2)))
        pad = (input_res - resolution) // 2
        self.resolution = resolution
        self.out_channels = out_channels
        self.input_res = input_res
        self.pad = pad

        if attn_resolutions is None:
            attn_set = {input_res // (2**i) for i in range(len(channels))}
        else:
            attn_set = set(attn_resolutions)

        enc_blocks = []
        enc_attns = []
        prev_ch = channels[0]
        res = input_res
        for i, ch_out in enumerate(channels):
            resample = "down" if i > 0 else "none"
            if i > 0:
                res = res // 2
            level_blocks = []
            level_attns = []
            for j in range(num_res_blocks):
                in_ch = prev_ch if j == 0 else ch_out
                rs = resample if j == 0 else "none"
                level_blocks.append(
                    ResBlock(
                        in_ch,
                        ch_out,
                        time_dim,
                        num_groups,
                        resample=rs,
                        dropout_rate=dropout_rate,
                        key=next(ki),
                    )
                )
                level_attns.append(
                    CosineAttention(ch_out, ch_out // head_dim, key=next(ki))
                    if res in attn_set
                    else None
                )
            enc_blocks.append(level_blocks)
            enc_attns.append(level_attns)
            prev_ch = ch_out
        self.enc_blocks = enc_blocks
        self.enc_attns = enc_attns

        mid_res = input_res // (2 ** (len(channels) - 1))
        self.mid_block = ResBlock(
            channels[-1],
            channels[-1],
            time_dim,
            num_groups,
            dropout_rate=dropout_rate,
            key=next(ki),
        )
        self.mid_attn = (
            CosineAttention(channels[-1], channels[-1] // head_dim, key=next(ki))
            if mid_res in attn_set
            else None
        )

        dec_blocks = []
        dec_attns = []
        rev_channels = list(reversed(channels))
        dec_res = mid_res
        for i in range(len(rev_channels)):
            ch_in = rev_channels[i]
            ch_out = (
                rev_channels[i + 1] if i + 1 < len(rev_channels) else rev_channels[i]
            )
            has_up = i < len(rev_channels) - 1
            resample_last = "up" if has_up else "none"
            out_res = dec_res * 2 if has_up else dec_res
            level_blocks = []
            level_attns = []
            for j in range(num_res_blocks):
                if j == 0:
                    in_ch, out_ch, rs = 2 * ch_in, ch_in, "none"
                elif j == num_res_blocks - 1:
                    in_ch, out_ch, rs = ch_in, ch_out, resample_last
                else:
                    in_ch, out_ch, rs = ch_in, ch_in, "none"
                level_blocks.append(
                    ResBlock(
                        in_ch,
                        out_ch,
                        time_dim,
                        num_groups,
                        resample=rs,
                        dropout_rate=dropout_rate,
                        key=next(ki),
                    )
                )
                attn_ch = out_ch
                attn_res = out_res if j == num_res_blocks - 1 else dec_res
                level_attns.append(
                    CosineAttention(attn_ch, attn_ch // head_dim, key=next(ki))
                    if attn_res in attn_set
                    else None
                )
            dec_blocks.append(level_blocks)
            dec_attns.append(level_attns)
            dec_res = out_res
        self.dec_blocks = dec_blocks
        self.dec_attns = dec_attns

        self.final_norm = RMSGroupNorm(min(num_groups, channels[0]))
        self.final_conv = eqx.nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
            use_bias=False,
            key=next(ki),
        )

    def __call__(self, x, s, t, *, key=None):
        x = x.reshape(self.in_channels, self.resolution, self.resolution)
        if self.pad > 0:
            x = jnp.pad(x, ((0, 0), (self.pad, self.pad), (self.pad, self.pad)))
        x = jnp.concatenate([x, jnp.ones((1, self.input_res, self.input_res))], axis=0)

        s_emb = self.s_embed(s)
        t_emb = self.t_embed(t)
        t_emb = self.time_fuse2(
            jax.nn.silu(self.time_fuse1(jnp.concatenate([s_emb, t_emb])))
        )
        h = self.init_conv(x)

        def _maybe_split_key(key):
            if key is None:
                return None, None
            return jax.random.split(key)

        skips = []
        for level_blocks, level_attns in zip(self.enc_blocks, self.enc_attns):
            for block, attn in zip(level_blocks, level_attns):
                key, subkey = _maybe_split_key(key)
                h = block(h, t_emb, key=subkey)
                if attn is not None:
                    h = attn(h)
            skips.append(h)

        key, subkey = _maybe_split_key(key)
        h = self.mid_block(h, t_emb, key=subkey)
        if self.mid_attn is not None:
            h = self.mid_attn(h)

        for level_blocks, level_attns in zip(self.dec_blocks, self.dec_attns):
            h = jnp.concatenate([h, skips.pop()], axis=0)
            for block, attn in zip(level_blocks, level_attns):
                key, subkey = _maybe_split_key(key)
                h = block(h, t_emb, key=subkey)
                if attn is not None:
                    h = attn(h)

        h = self.final_norm(h)
        h = jax.nn.silu(h)
        h = self.final_conv(h)
        if self.pad > 0:
            h = h[
                :,
                self.pad : self.pad + self.resolution,
                self.pad : self.pad + self.resolution,
            ]
        return h


class EDM2EMStepModel(AbstractEMStepModel):
    drift_net: EDM2_UNet
    diffusion_net: EDM2_UNet

    def drift(
        self,
        ys: Y,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key=None,
    ) -> Y:
        x = jnp.concatenate([ys, W, H, K], axis=0)
        return self.drift_net(x, s, t, key=key)

    def diffusion(
        self,
        s: Float[Array, ""],
        t: Float[Array, ""],
        W: Y,
        H: Y,
        K: Y,
        *,
        key=None,
    ) -> Y:
        x = jnp.concatenate([W, H, K], axis=0)
        return self.diffusion_net(x, s, t, key=key)

class UncertaintyMLP(eqx.Module):
    freqs_s: jax.Array
    phases_s: jax.Array
    freqs_t: jax.Array
    phases_t: jax.Array
    linear: eqx.nn.Linear

    def __init__(self, fourier_dim: int, *, key: PRNGKeyArray):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.freqs_s = jax.random.normal(k1, (fourier_dim,))
        self.phases_s = jax.random.uniform(k2, (fourier_dim,))
        self.freqs_t = jax.random.normal(k3, (fourier_dim,))
        self.phases_t = jax.random.uniform(k4, (fourier_dim,))
        self.linear = eqx.nn.Linear(2 * fourier_dim, 1, key=k5)

    def __call__(self, s: Float[Array, ""], t: Float[Array, ""]) -> Float[Array, ""]:
        feat_s = jnp.cos(
            2
            * jnp.pi
            * (
                jax.lax.stop_gradient(self.freqs_s) * s
                + jax.lax.stop_gradient(self.phases_s)
            )
        )
        feat_t = jnp.cos(
            2
            * jnp.pi
            * (
                jax.lax.stop_gradient(self.freqs_t) * t
                + jax.lax.stop_gradient(self.phases_t)
            )
        )
        return self.linear(jnp.concatenate([feat_s, feat_t])).squeeze()


class UncertaintyScoreLoss(eqx.Module):
    diffusion: AbstractDiffusion
    dt: float
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: AbstractFlowMap,
        y0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        sample_key, vbt_key, dropout_key = jax.random.split(key, 3)

        ys, score = self.diffusion.forward_sample(s, y0, sample_key)

        s = jnp.clip(s, self.t_eps, 1.0 - self.t_eps)
        t = jnp.clip(s - h, self.t_eps, 1.0)

        drift = self.diffusion.reverse_drift(s, ys, score)
        g = self.diffusion.reverse_diffusion(s)

        vbt = diffrax.VirtualBrownianTree(
            t0=self.t_eps,
            t1=1.0,
            tol=h / 4,
            shape=y0.shape,
            key=vbt_key,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )
        levy = vbt.evaluate(s, t, use_levy=True)

        yt_target = ys + drift * (t - s) + g * levy.W
        yt_pred = flow_map(ys, s, t, levy.W, levy.H, levy.K, key=dropout_key)  # pyright: ignore

        raw_loss = jnp.sum((yt_pred - jax.lax.stop_gradient(yt_target)) ** 2) / h
        u = u_mlp(s, t)
        return jnp.exp(-u) * raw_loss + u

    def value(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0_batch: BatchY,
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        batch_size = y0_batch.shape[0]
        key_h, key_s, key_vbt = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt / 2, maxval=self.dt
        )
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps + h_vals, maxval=1.0
        )
        vbt_keys = jax.random.split(key_vbt, batch_size)

        losses = eqx.filter_vmap(
            lambda y0, s, h, k: self._single(flow_map, y0, s, h, k, u_mlp)
        )(y0_batch, s_vals, h_vals, vbt_keys)

        return jnp.mean(losses)


class UncertaintyDistillationLoss(eqx.Module):
    diffusion: AbstractDiffusion
    dt: float
    h_max: float
    t_eps: float = 1e-5

    def _single(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0: Y,
        s: Float[Array, ""],
        h: Float[Array, ""],
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        sample_key, vbt_key, dropout_key = jax.random.split(key, 3)
        ema_flow_map = eqx.nn.inference_mode(ema_flow_map)

        ys, _ = self.diffusion.forward_sample(s, y0, sample_key)

        s = jnp.clip(s, self.t_eps, 1.0 - self.t_eps)
        t = jnp.clip(s - h, self.t_eps, 1.0)
        mid = jnp.clip(s - h / 2, self.t_eps, 1.0 - self.t_eps)

        vbt = diffrax.VirtualBrownianTree(
            t0=self.t_eps,
            t1=1.0,
            tol=h / 4,
            shape=y0.shape,
            key=vbt_key,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        levy_1 = vbt.evaluate(s, mid, use_levy=True)
        y_mid = ema_flow_map(ys, s, mid, levy_1.W, levy_1.H, levy_1.K)  # pyright: ignore
        levy_2 = vbt.evaluate(mid, t, use_levy=True)
        yt_target = ema_flow_map(y_mid, mid, t, levy_2.W, levy_2.H, levy_2.K)  # pyright: ignore
        yt_target = jax.lax.stop_gradient(yt_target)

        levy = vbt.evaluate(s, t, use_levy=True)
        yt_pred = flow_map(ys, s, t, levy.W, levy.H, levy.K, key=dropout_key)  # pyright: ignore

        raw_loss = jnp.sum((yt_pred - yt_target) ** 2) / h
        u = u_mlp(s, t)
        return jnp.exp(-u) * raw_loss + u

    def value(
        self,
        flow_map: AbstractFlowMap,
        ema_flow_map: AbstractFlowMap,
        y0_batch: BatchY,
        key: PRNGKeyArray,
        u_mlp: UncertaintyMLP,
    ) -> Float[Array, ""]:
        batch_size = y0_batch.shape[0]
        key_h, key_s, key_vbt = jax.random.split(key, 3)

        h_vals = jax.random.uniform(
            key_h, (batch_size,), minval=self.dt, maxval=self.h_max
        )
        s_vals = jax.random.uniform(
            key_s, (batch_size,), minval=self.t_eps + h_vals, maxval=1.0
        )
        vbt_keys = jax.random.split(key_vbt, batch_size)

        losses = eqx.filter_vmap(
            lambda y0, s, h, k: self._single(flow_map, ema_flow_map, y0, s, h, k, u_mlp)
        )(y0_batch, s_vals, h_vals, vbt_keys)

        return jnp.mean(losses)


class UncertaintyJointLoss(eqx.Module):
    score_loss: UncertaintyScoreLoss
    distill_loss: UncertaintyDistillationLoss
    u_mlp: UncertaintyMLP
    eta: float = 0.75

    def value_and_grad(self, flow_map, ema_flow_map, y0_batch, key):
        @eqx.filter_value_and_grad
        def _loss(diff_args):
            fm, u = diff_args
            key_score, key_distill = jax.random.split(key)
            n_score = int(y0_batch.shape[0] * self.eta)
            l_score = self.score_loss.value(
                fm, ema_flow_map, y0_batch[:n_score], key_score, u
            )
            l_distill = self.distill_loss.value(
                fm, ema_flow_map, y0_batch[n_score:], key_distill, u
            )
            return self.eta * l_score + (1 - self.eta) * l_distill

        loss, (fm_grads, u_grads) = _loss((flow_map, self.u_mlp))
        return loss, fm_grads, u_grads


def cifar10(path="data"):
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(path, "cifar-10-python.tar.gz")

    os.makedirs(path, exist_ok=True)
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, tar_path)

    all_images = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for i in range(1, 6):
            member = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            batch = pickle.load(member, encoding="bytes")  # pyright: ignore
            all_images.append(batch[b"data"])

    images = np.concatenate(all_images, axis=0)
    images = jnp.array(images, dtype=jnp.float32) / 255.0

    data_mean = jnp.mean(images, axis=0)
    data_std = jnp.clip(jnp.std(images, axis=0), 1e-6)
    images = (images - data_mean) / data_std
    return images, data_mean, data_std


def ema_update(
    model: EulerMaruyamaFlowMap,
    ema_model: EulerMaruyamaFlowMap,
    decay: float,
) -> EulerMaruyamaFlowMap:
    params, static = eqx.partition(model, eqx.is_array)
    ema_params, _ = eqx.partition(ema_model, eqx.is_array)
    new_ema_params = jax.tree.map(
        lambda e, p: decay * e + (1 - decay) * p, ema_params, params
    )
    return eqx.combine(new_ema_params, static)


@eqx.filter_jit
def train_step(
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    y0_batch: Float[Array, "batch 3 32 32"],
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    joint_loss: UncertaintyJointLoss,
    ema_decay: float,
    key: PRNGKeyArray,
):
    loss, fm_grads, u_grads = joint_loss.value_and_grad(
        flow_map, ema_flow_map, y0_batch, key
    )

    updates, opt_state = optimizer.update(
        (fm_grads, u_grads),
        opt_state,
        (flow_map, joint_loss.u_mlp),  # pyright: ignore
    )
    fm_updates, u_updates = updates  # pyright: ignore
    flow_map = eqx.apply_updates(flow_map, fm_updates)
    new_u_mlp = eqx.apply_updates(joint_loss.u_mlp, u_updates)
    joint_loss = eqx.tree_at(lambda l: l.u_mlp, joint_loss, new_u_mlp)

    ema_flow_map = ema_update(flow_map, ema_flow_map, ema_decay)

    return flow_map, ema_flow_map, opt_state, joint_loss, loss


def sample_flow_map(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    n_samples: int,
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
):
    flow_map = eqx.nn.inference_mode(flow_map)

    time_grid = jnp.linspace(1.0, t_eps, n_steps + 1)

    def sample_one(key):
        key_init, key_vbt = jax.random.split(key)
        y = jax.random.normal(key_init, data_shape)

        vbt = diffrax.VirtualBrownianTree(
            t0=t_eps,
            t1=1.0,
            tol=step_size / 4,
            shape=data_shape,
            key=key_vbt,
            levy_area=diffrax.SpaceTimeTimeLevyArea,
        )

        def scan_fn(y, i):
            s = jnp.clip(time_grid[i], t_eps, 1.0)
            t = jnp.clip(time_grid[i + 1], t_eps, 1.0)
            levy = vbt.evaluate(s, t, use_levy=True)
            y = flow_map(y, s, t, levy.W, levy.H, levy.K)  # pyright: ignore
            return y, None

        y, _ = jax.lax.scan(scan_fn, y, jnp.arange(n_steps))
        return y

    keys = jax.random.split(key, n_samples)
    return jax.vmap(sample_one)(keys)


def sample_flow_map_batched(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    n_samples: int,
    data_shape: tuple[int, ...],
    step_size: float,
    n_steps: int,
    batch_size: int = 256,
):
    all_samples = []
    remaining = n_samples
    while remaining > 0:
        key, key_batch = jax.random.split(key)
        bs = min(batch_size, remaining)
        batch = sample_flow_map(
            flow_map, t_eps, key_batch, bs, data_shape, step_size, n_steps
        )
        all_samples.append(batch)
        remaining -= bs
    return jnp.concatenate(all_samples, axis=0)


def make_sample_grid(
    flow_map: EulerMaruyamaFlowMap,
    t_eps: float,
    key: PRNGKeyArray,
    data_shape: tuple[int, ...],
    step_sizes: list[float],
    data_mean: jnp.ndarray,
    data_std: jnp.ndarray,
    n_cols: int = 8,
):
    n_rows = len(step_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ss in enumerate(step_sizes):
        key, key_sample = jax.random.split(key)
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        samples = sample_flow_map(
            flow_map, t_eps, key_sample, n_cols, data_shape, ss, n_steps
        )
        samples = samples * data_std[None] + data_mean[None]
        samples = jnp.clip(samples, 0.0, 1.0)

        for col in range(n_cols):
            ax = axes[row, col]
            img = np.array(samples[col].transpose(1, 2, 0))
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(f"h={ss}", fontsize=10)

    fig.tight_layout()
    return fig


def format_step(step: int) -> str:
    if step >= 1000 and step % 1000 == 0:
        return f"{step // 1000}k"
    return str(step)


def parse_step(name: str) -> int | None:
    if name.endswith("k"):
        try:
            return int(name[:-1]) * 1000
        except ValueError:
            return None
    try:
        return int(name)
    except ValueError:
        return None


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None
    best_step = -1
    best_path = None
    for name in os.listdir(checkpoint_dir):
        full = os.path.join(checkpoint_dir, name)
        if not os.path.isdir(full):
            continue
        step = parse_step(name)
        if step is None:
            continue
        if step > best_step:
            best_step = step
            best_path = full
    return best_path


def save_checkpoint(
    path: str,
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    opt_state: optax.OptState,
    u_mlp: UncertaintyMLP,
    step: int,
    key: PRNGKeyArray,
):
    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(path, "flow_map.eqx"), flow_map)
    eqx.tree_serialise_leaves(os.path.join(path, "ema_flow_map.eqx"), ema_flow_map)
    eqx.tree_serialise_leaves(os.path.join(path, "u_mlp.eqx"), u_mlp)
    with open(os.path.join(path, "opt_state.pkl"), "wb") as f:
        pickle.dump(opt_state, f)
    with open(os.path.join(path, "train_state.pkl"), "wb") as f:
        pickle.dump({"step": step, "key": key}, f)
    print(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str,
    flow_map: EulerMaruyamaFlowMap,
    ema_flow_map: EulerMaruyamaFlowMap,
    u_mlp: UncertaintyMLP,
    optimizer: optax.GradientTransformation,
):
    flow_map = eqx.tree_deserialise_leaves(os.path.join(path, "flow_map.eqx"), flow_map)
    ema_flow_map = eqx.tree_deserialise_leaves(
        os.path.join(path, "ema_flow_map.eqx"), ema_flow_map
    )
    u_mlp = eqx.tree_deserialise_leaves(os.path.join(path, "u_mlp.eqx"), u_mlp)
    with open(os.path.join(path, "opt_state.pkl"), "rb") as f:
        opt_state = pickle.load(f)
    with open(os.path.join(path, "train_state.pkl"), "rb") as f:
        train_state = pickle.load(f)
    print(f"Resumed from checkpoint at step {train_state['step']} from {path}")
    return (
        flow_map,
        ema_flow_map,
        opt_state,
        u_mlp,
        train_state["step"],
        train_state["key"],
    )


def build_model(
    key: PRNGKeyArray,
    *,
    in_channels: int = 3,
    resolution: int = 32,
    base_channels: int = 128,
    channel_mult: tuple[int, ...] = (2, 2, 2),
    num_groups: int = 8,
    dropout_rate: float = 0.13,
    attn_resolutions: tuple[int, ...] = (16,),
    head_dim: int = 64,
    num_res_blocks: int = 4,
    diff_base_channels: int = 64,
    diff_channel_mult: tuple[int, ...] = (2, 2, 2),
    diff_attn_resolutions: tuple[int, ...] = (16,),
) -> EulerMaruyamaFlowMap:
    """Construct the CIFAR-10 SSFM model used by training and evaluation."""
    key_drift, key_diff = jax.random.split(key)
    drift_net = EDM2_UNet(
        in_channels=4 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_groups=num_groups,
        num_res_blocks=num_res_blocks,
        dropout_rate=dropout_rate,
        attn_resolutions=attn_resolutions,
        head_dim=head_dim,
        key=key_drift,
    )
    diffusion_net = EDM2_UNet(
        in_channels=3 * in_channels,
        out_channels=in_channels,
        resolution=resolution,
        base_channels=diff_base_channels,
        channel_mult=diff_channel_mult,
        num_groups=num_groups,
        num_res_blocks=num_res_blocks,
        dropout_rate=0.0,
        attn_resolutions=diff_attn_resolutions,
        head_dim=head_dim,
        key=key_diff,
    )
    return EulerMaruyamaFlowMap(
        step_model=EDM2EMStepModel(
            drift_net=drift_net,
            diffusion_net=diffusion_net,
        )
    )


def main():
    in_channels = 3
    resolution = 32
    data_shape = (in_channels, resolution, resolution)
    batch_size = 512
    n_train_steps = 400_000
    lr = 1e-3
    eta = 0.75
    ema_decay = 0.999
    dt = 0.01
    h_max = 0.52
    t_eps = 1e-5
    beta_min = 0.1
    beta_max = 20.0
    sample_every = 10_000
    checkpoint_every = 50_000
    checkpoint_dir = "checkpoints"
    warmup_steps = 5_000
    sample_step_sizes = [1 / 16, 1 / 8, 1 / 4, 1 / 2]

    base_channels = 128
    channel_mult = (2, 2, 2)
    num_groups = 8
    dropout_rate = 0.13
    attn_resolutions = [16]
    head_dim = 64
    num_res_blocks = 4

    diff_base_channels = 64
    diff_channel_mult = (2, 2, 2)
    diff_attn_resolutions = [16]

    config = {
        "architecture": "edm2",
        "dataset": "cifar10",
        "batch_size": batch_size,
        "n_train_steps": n_train_steps,
        "lr": lr,
        "eta": eta,
        "ema_decay": ema_decay,
        "dt": dt,
        "h_max": h_max,
        "t_eps": t_eps,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "sample_every": sample_every,
        "checkpoint_every": checkpoint_every,
        "warmup_steps": warmup_steps,
        "sample_step_sizes": sample_step_sizes,
        "base_channels": base_channels,
        "channel_mult": channel_mult,
        "attn_resolutions": attn_resolutions,
        "diff_base_channels": diff_base_channels,
        "diff_channel_mult": diff_channel_mult,
        "diff_attn_resolutions": diff_attn_resolutions,
        "num_res_blocks": num_res_blocks,
        "dropout_rate": dropout_rate,
    }

    wandb.init(project="Stochastic Flow Map", config=config)

    devices = jax.devices()
    n_devices = len(devices)
    mesh = Mesh(np.array(devices), axis_names=("batch",))
    data_sharding = NamedSharding(mesh, P("batch", None, None, None))
    replicate_sharding = NamedSharding(mesh, P())
    assert batch_size % n_devices == 0, (
        f"batch_size {batch_size} not divisible by {n_devices} devices"
    )
    print(f"Training on {n_devices} devices: {devices}")

    key = jax.random.PRNGKey(0)
    key, key_model, key_u = jax.random.split(key, 3)

    dataset, data_mean, data_std = cifar10()
    dataset = dataset.reshape(-1, *data_shape)
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)
    print(f"Dataset: {dataset.shape}")

    diffusion = VPDiffusion(reverse_eta=1.0, beta_min=beta_min, beta_max=beta_max)

    flow_map = build_model(
        key_model,
        in_channels=in_channels,
        resolution=resolution,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_groups=num_groups,
        dropout_rate=dropout_rate,
        attn_resolutions=tuple(attn_resolutions),
        head_dim=head_dim,
        num_res_blocks=num_res_blocks,
        diff_base_channels=diff_base_channels,
        diff_channel_mult=diff_channel_mult,
        diff_attn_resolutions=tuple(diff_attn_resolutions),
    )
    ema_flow_map = flow_map

    n_drift = sum(
        x.size
        for x in jax.tree.leaves(eqx.filter(flow_map.step_model.drift_net, eqx.is_array))
    )
    n_diff = sum(
        x.size
        for x in jax.tree.leaves(
            eqx.filter(flow_map.step_model.diffusion_net, eqx.is_array)
        )
    )
    print(f"Drift EDM2 UNet params: {n_drift:,}")
    print(f"Diffusion EDM2 UNet params: {n_diff:,}")
    print(f"Total params: {n_drift + n_diff:,}")
    wandb.config.update({"n_drift_params": n_drift, "n_diff_params": n_diff})

    score_loss = UncertaintyScoreLoss(diffusion=diffusion, dt=dt, t_eps=t_eps)
    distill_loss = UncertaintyDistillationLoss(
        diffusion=diffusion, dt=dt, h_max=h_max, t_eps=t_eps
    )
    u_mlp = UncertaintyMLP(fourier_dim=128, key=key_u)
    joint_loss = UncertaintyJointLoss(
        score_loss=score_loss, distill_loss=distill_loss, u_mlp=u_mlp, eta=eta
    )

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=n_train_steps,
        end_value=1e-5,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule, b2=0.99),
    )
    opt_state = optimizer.init(
        (eqx.filter(flow_map, eqx.is_array), eqx.filter(joint_loss.u_mlp, eqx.is_array))
    )

    flow_map = eqx.filter_shard(flow_map, replicate_sharding)
    ema_flow_map = eqx.filter_shard(ema_flow_map, replicate_sharding)
    joint_loss = eqx.filter_shard(joint_loss, replicate_sharding)
    opt_state = eqx.filter_shard(opt_state, replicate_sharding)

    start_step = 0
    latest_ckpt = find_latest_checkpoint(checkpoint_dir)
    if latest_ckpt is not None:
        flow_map, ema_flow_map, opt_state, u_mlp, start_step, key = load_checkpoint(
            latest_ckpt,
            flow_map,
            ema_flow_map,
            joint_loss.u_mlp,
            optimizer,
        )
        joint_loss = eqx.tree_at(lambda l: l.u_mlp, joint_loss, u_mlp)
        flow_map = eqx.filter_shard(flow_map, replicate_sharding)
        ema_flow_map = eqx.filter_shard(ema_flow_map, replicate_sharding)
        joint_loss = eqx.filter_shard(joint_loss, replicate_sharding)
        opt_state = eqx.filter_shard(opt_state, replicate_sharding)

    for step in range(start_step, n_train_steps):
        key, key_batch, key_step = jax.random.split(key, 3)
        idx = jax.random.randint(key_batch, (batch_size,), 0, dataset.shape[0])
        y0_batch = dataset[idx]
        y0_batch = jax.device_put(y0_batch, data_sharding)

        flow_map, ema_flow_map, opt_state, joint_loss, loss = train_step(
            flow_map,
            ema_flow_map,
            y0_batch,
            opt_state,
            optimizer,
            joint_loss,
            ema_decay,
            key_step,
        )

        if step % 1000 == 0:
            loss_val = loss.item()
            print(f"Step {step:>6d} | Loss: {loss_val:.4f}")
            wandb.log({"loss": loss_val}, step=step)

        if step > 0 and step % sample_every == 0:
            key, key_sample = jax.random.split(key)
            fig = make_sample_grid(
                ema_flow_map,
                t_eps,
                key_sample,
                data_shape,
                sample_step_sizes,
                data_mean,
                data_std,
            )
            wandb.log({"samples": wandb.Image(fig)}, step=step)
            plt.close(fig)

        if step > 0 and step % checkpoint_every == 0:
            save_checkpoint(
                os.path.join(checkpoint_dir, format_step(step)),
                flow_map,
                ema_flow_map,
                opt_state,
                joint_loss.u_mlp,
                step,
                key,
            )

    loss_val = loss.item()
    print(f"Step {n_train_steps:>6d} | Loss: {loss_val:.4f}")
    wandb.log({"loss": loss_val}, step=n_train_steps)

    save_checkpoint(
        os.path.join(checkpoint_dir, format_step(n_train_steps)),
        flow_map,
        ema_flow_map,
        opt_state,
        joint_loss.u_mlp,
        n_train_steps,
        key,
    )

    key, key_sample = jax.random.split(key)
    fig = make_sample_grid(
        ema_flow_map,
        t_eps,
        key_sample,
        data_shape,
        sample_step_sizes,
        data_mean,
        data_std,
    )
    fig.savefig("cifar10_edm2_em_samples.png", dpi=150)
    print("Saved cifar10_edm2_em_samples.png")
    wandb.log({"samples": wandb.Image(fig)}, step=n_train_steps)
    plt.close(fig)

    wandb.finish()


if __name__ == "__main__":
    main()

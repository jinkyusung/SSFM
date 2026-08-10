import math
import os
from collections.abc import Iterator
import dataclasses

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.7")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from scipy import linalg
from ssfm_cifar import (
    build_model,
    cifar10,
    sample_flow_map_batched,
)

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
except ImportError as exc:
    raise ImportError(
        "FID evaluation requires torch and torchvision. "
        "Re-run setup_ssfm_cifar_env.sh with the metrics option."
    ) from exc


@dataclasses.dataclass
class FIDStats:
    mu: np.ndarray
    sigma: np.ndarray
    n_samples: int

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mu, sigma=self.sigma, n_samples=self.n_samples)

    @classmethod
    def load(cls, path: str) -> "FIDStats":
        data = np.load(path)
        return cls(
            mu=data["mu"],
            sigma=data["sigma"],
            n_samples=int(data["n_samples"]),
        )


class InceptionFeatureExtractor:
    def __init__(self, device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()  # pyright: ignore
        model.eval()
        self.model = model.to(self.device)

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> np.ndarray:
        output = self.model(images.to(self.device))
        return output.cpu().numpy()


def preprocess_images(
    images: np.ndarray,
    target_size: int = 299,
    batch_size: int = 64,
) -> Iterator[torch.Tensor]:
    for start in range(0, images.shape[0], batch_size):
        processed = []
        for image in images[start : start + batch_size]:
            image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            pil_image = Image.fromarray(image.transpose(1, 2, 0))
            pil_image = pil_image.resize((target_size, target_size), Image.BICUBIC)
            processed.append(np.array(pil_image).transpose(2, 0, 1))
        yield torch.from_numpy(np.stack(processed)).float() / 255.0


def compute_statistics(features: np.ndarray) -> FIDStats:
    return FIDStats(
        mu=np.mean(features, axis=0),
        sigma=np.cov(features, rowvar=False),
        n_samples=features.shape[0],
    )


def frechet_distance(stats1: FIDStats, stats2: FIDStats) -> float:
    difference = stats1.mu - stats2.mu
    covariance_mean = linalg.sqrtm(stats1.sigma @ stats2.sigma)
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(
                f"Imaginary component too large: "
                f"{np.max(np.abs(covariance_mean.imag))}"
            )
        covariance_mean = covariance_mean.real
    return float(
        difference @ difference
        + np.trace(stats1.sigma)
        + np.trace(stats2.sigma)
        - 2 * np.trace(covariance_mean)
    )


def compute_fid(
    generated_samples: np.ndarray,
    real_stats: FIDStats | str,
    batch_size: int = 64,
    device: str | None = None,
) -> float:
    if isinstance(real_stats, str):
        real_stats = FIDStats.load(real_stats)
    extractor = InceptionFeatureExtractor(device=device)
    features = [
        extractor.features(batch)
        for batch in preprocess_images(
            np.asarray(generated_samples), batch_size=batch_size
        )
    ]
    return frechet_distance(compute_statistics(np.concatenate(features)), real_stats)


def compute_and_cache_real_stats(
    dataset: jnp.ndarray,
    cache_path: str,
    batch_size: int = 64,
) -> FIDStats:
    if os.path.exists(cache_path):
        print(f"Loading cached real stats from {cache_path}")
        return FIDStats.load(cache_path)

    print("Computing Inception stats for real dataset...")
    images = np.asarray(dataset)
    extractor = InceptionFeatureExtractor()
    all_features = []
    for batch in preprocess_images(images, batch_size=batch_size):
        all_features.append(extractor.features(batch))
    features = np.concatenate(all_features, axis=0)
    stats = compute_statistics(features)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    stats.save(cache_path)
    print(f"Saved real stats to {cache_path}")
    return stats


def main(
    model_path="models/cifar10_edm2_em.eqx",
    checkpoint_path="checkpoints/400k",
    n_samples=50_000,
    sample_batch_size=256,
    step_sizes=(1 / 16, 1 / 8, 1 / 4, 1 / 2),
    t_eps=1e-5,
    seed=42,
):
    in_channels = 3
    resolution = 32
    data_shape = (in_channels, resolution, resolution)

    dataset, data_mean, data_std = cifar10()
    dataset = dataset.reshape(-1, *data_shape)
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)

    real_stats_path = "data/cifar10_inception_stats.npz"
    dataset_for_fid = dataset * data_std[None] + data_mean[None]
    dataset_for_fid = dataset_for_fid * 2.0 - 1.0
    real_stats = compute_and_cache_real_stats(
        dataset_for_fid, cache_path=real_stats_path
    )

    key = jax.random.PRNGKey(0)
    flow_map = build_model(key, in_channels=in_channels, resolution=resolution)
    if checkpoint_path is not None:
        ema_path = os.path.join(checkpoint_path, "ema_flow_map.eqx")
        flow_map = eqx.tree_deserialise_leaves(ema_path, flow_map)
        print(f"Loaded EMA model from checkpoint {checkpoint_path}")
    else:
        flow_map = eqx.tree_deserialise_leaves(model_path, flow_map)
        print(f"Loaded model from {model_path}")

    key = jax.random.PRNGKey(seed)
    for ss in step_sizes:
        n_steps = max(1, math.ceil((1.0 - t_eps) / ss))
        key, key_fid = jax.random.split(key)
        fid_samples = sample_flow_map_batched(
            flow_map,
            t_eps,
            key_fid,
            n_samples,
            data_shape,
            ss,
            n_steps,
            batch_size=sample_batch_size,
        )
        fid_samples = fid_samples * data_std[None] + data_mean[None]
        fid_samples = jnp.clip(fid_samples, 0.0, 1.0)
        fid_samples = fid_samples * 2.0 - 1.0
        fid = compute_fid(np.array(fid_samples), real_stats, batch_size=64)
        print(f"FID (h={ss}, steps={n_steps}): {fid:.2f}")


if __name__ == "__main__":
    main()

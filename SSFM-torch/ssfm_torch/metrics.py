"""FID utilities implemented entirely with PyTorch feature extraction."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import linalg
import torch
from torch import Tensor, nn
from torchvision.models import Inception_V3_Weights, inception_v3


@dataclass
class FIDStats:
    mean: np.ndarray
    covariance: np.ndarray
    n_samples: int

    def save(self, path: str) -> None:
        np.savez(path, mu=self.mean, sigma=self.covariance, n_samples=self.n_samples)

    @classmethod
    def load(cls, path: str) -> "FIDStats":
        data = np.load(path)
        return cls(data["mu"], data["sigma"], int(data["n_samples"]))


class InceptionFeatureExtractor:
    def __init__(self, device: str | torch.device | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.fc = nn.Identity()
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, images: Tensor) -> np.ndarray:
        return self.model(images.to(self.device)).cpu().numpy()


def preprocess_images(
    images: Tensor | np.ndarray, target_size: int = 299, batch_size: int = 64
) -> Iterator[Tensor]:
    images = np.asarray(images)
    for start in range(0, images.shape[0], batch_size):
        processed = []
        for image in images[start : start + batch_size]:
            image = ((image + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            pil_image = Image.fromarray(image.transpose(1, 2, 0))
            pil_image = pil_image.resize((target_size, target_size), Image.Resampling.BICUBIC)
            processed.append(np.asarray(pil_image).transpose(2, 0, 1))
        yield torch.from_numpy(np.stack(processed).copy()).float() / 255.0


def compute_statistics(features: np.ndarray) -> FIDStats:
    return FIDStats(features.mean(axis=0), np.cov(features, rowvar=False), features.shape[0])


def frechet_distance(first: FIDStats, second: FIDStats) -> float:
    difference = first.mean - second.mean
    covariance_mean = linalg.sqrtm(first.covariance @ second.covariance)
    if np.iscomplexobj(covariance_mean):
        if not np.allclose(np.diagonal(covariance_mean).imag, 0, atol=1e-3):
            raise ValueError(
                f"imaginary covariance component is too large: {np.abs(covariance_mean.imag).max()}"
            )
        covariance_mean = covariance_mean.real
    return float(
        difference @ difference
        + np.trace(first.covariance)
        + np.trace(second.covariance)
        - 2 * np.trace(covariance_mean)
    )


def compute_fid(
    generated_samples: Tensor | np.ndarray,
    real_stats: FIDStats | str,
    batch_size: int = 64,
    device: str | torch.device | None = None,
) -> float:
    if isinstance(real_stats, str):
        real_stats = FIDStats.load(real_stats)
    extractor = InceptionFeatureExtractor(device)
    features = [extractor(batch) for batch in preprocess_images(generated_samples, batch_size=batch_size)]
    return frechet_distance(compute_statistics(np.concatenate(features)), real_stats)


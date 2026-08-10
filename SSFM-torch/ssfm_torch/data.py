"""CIFAR-10 loading with the original per-pixel normalization."""

from __future__ import annotations

import os
import pickle
import tarfile
import urllib.request

import numpy as np
import torch
from torch import Tensor


CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def cifar10(path: str = "data", *, download: bool = True) -> tuple[Tensor, Tensor, Tensor]:
    tar_path = os.path.join(path, "cifar-10-python.tar.gz")
    os.makedirs(path, exist_ok=True)
    if not os.path.exists(tar_path):
        if not download:
            raise FileNotFoundError(tar_path)
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)

    batches = []
    with tarfile.open(tar_path, "r:gz") as archive:
        for index in range(1, 6):
            member = archive.extractfile(f"cifar-10-batches-py/data_batch_{index}")
            if member is None:
                raise RuntimeError(f"CIFAR archive is missing data batch {index}")
            batch = pickle.load(member, encoding="bytes")
            batches.append(batch[b"data"])

    images = torch.from_numpy(np.concatenate(batches)).to(torch.float32) / 255.0
    data_mean = images.mean(dim=0)
    data_std = images.std(dim=0, correction=0).clamp_min(1e-6)
    return (images - data_mean) / data_std, data_mean, data_std


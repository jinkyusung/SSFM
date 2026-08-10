#!/usr/bin/env python3
"""Train the Torch SSFM model on CIFAR-10."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import math
import os
import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import wandb

from ssfm_torch.checkpoint import (
    find_latest_checkpoint,
    format_step,
    load_checkpoint,
    save_checkpoint,
)
from ssfm_torch.data import cifar10
from ssfm_torch.diffusion import VPDiffusion
from ssfm_torch.losses import (
    UncertaintyDistillationLoss,
    UncertaintyJointLoss,
    UncertaintyMLP,
    UncertaintyScoreLoss,
    ema_update,
)
from ssfm_torch.model import build_model
from ssfm_torch.paths import CHECKPOINT_DIR, DATA_DIR, PROJECT_DIR
from ssfm_torch.sampling import make_sample_grid


@dataclass
class TrainConfig:
    data_dir: str = str(DATA_DIR)
    checkpoint_dir: str = str(CHECKPOINT_DIR)
    batch_size: int = 512
    n_train_steps: int = 400_000
    learning_rate: float = 1e-3
    eta: float = 0.75
    ema_decay: float = 0.999
    dt: float = 0.01
    h_max: float = 0.52
    t_eps: float = 1e-5
    beta_min: float = 0.1
    beta_max: float = 20.0
    warmup_steps: int = 5_000
    sample_every: int = 10_000
    checkpoint_every: int = 50_000
    seed: int = 0
    device: str = "auto"
    wandb_mode: str = "online"


class TrainSystem(nn.Module):
    """Wrap all trainable state so DDP synchronizes the model and uncertainty MLP."""

    def __init__(self, flow_map: nn.Module, joint_loss: UncertaintyJointLoss):
        super().__init__()
        self.flow_map = flow_map
        self.joint_loss = joint_loss

    def forward(self, ema_flow_map: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return self.joint_loss(self.flow_map, ema_flow_map, batch)


def cosine_warmup(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return step / max(1, config.warmup_steps)
    progress = min(
        1.0,
        (step - config.warmup_steps)
        / max(1, config.n_train_steps - config.warmup_steps),
    )
    end_ratio = 1e-5 / config.learning_rate
    return end_ratio + 0.5 * (1 - end_ratio) * (1 + math.cos(math.pi * progress))


def resolve_device(name: str, local_rank: int = 0, distributed: bool = False) -> torch.device:
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed training currently requires CUDA")
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(config: TrainConfig) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = resolve_device(config.device, local_rank, distributed)
    if rank == 0:
        print(f"Training on {world_size} process(es), primary device {device}")
    if config.batch_size % world_size:
        raise ValueError("global batch size must be divisible by the distributed world size")
    local_batch_size = config.batch_size // world_size

    if distributed:
        if rank == 0:
            dataset, data_mean, data_std = cifar10(config.data_dir)
        dist.barrier(device_ids=[local_rank])
        if rank != 0:
            dataset, data_mean, data_std = cifar10(config.data_dir, download=False)
    else:
        dataset, data_mean, data_std = cifar10(config.data_dir)
    data_shape = (3, 32, 32)
    dataset = dataset.reshape(-1, *data_shape)
    data_mean = data_mean.reshape(data_shape)
    data_std = data_std.reshape(data_shape)
    if rank == 0:
        print(f"Dataset: {tuple(dataset.shape)}")

    flow_map = build_model().to(device)
    ema_flow_map = copy.deepcopy(flow_map).eval().requires_grad_(False)
    diffusion = VPDiffusion(1.0, config.beta_min, config.beta_max)
    uncertainty_mlp = UncertaintyMLP(128).to(device)
    joint_loss = UncertaintyJointLoss(
        UncertaintyScoreLoss(diffusion, config.dt, config.t_eps),
        UncertaintyDistillationLoss(
            diffusion, config.dt, config.h_max, config.t_eps
        ),
        uncertainty_mlp,
        config.eta,
    )
    train_system = TrainSystem(flow_map, joint_loss).to(device)
    parameters = list(train_system.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cosine_warmup(step, config)
    )

    start_step = 0
    latest = find_latest_checkpoint(config.checkpoint_dir)
    if latest is not None:
        state = load_checkpoint(
            latest,
            flow_map,
            ema_flow_map,
            uncertainty_mlp,
            optimizer,
            scheduler,
            map_location=device,
        )
        start_step = int(state["step"])
        if rank == 0:
            print(f"Resumed step {start_step} from {latest}")

    if distributed:
        train_system = DistributedDataParallel(
            train_system, device_ids=[local_rank], output_device=local_rank
        )
    # Parameters are initialized identically on every rank; training randomness is not.
    torch.manual_seed(config.seed + rank * 1_000_003 + start_step)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed + rank * 1_000_003 + start_step)

    n_drift = sum(p.numel() for p in flow_map.step_model.drift_net.parameters())
    n_diffusion = sum(p.numel() for p in flow_map.step_model.diffusion_net.parameters())
    run = None
    if rank == 0:
        run = wandb.init(
            project="Stochastic Flow Map",
            config={**asdict(config), "n_drift_params": n_drift, "n_diff_params": n_diffusion},
            mode=config.wandb_mode,
            dir=str(PROJECT_DIR),
        )
        print(f"Drift params: {n_drift:,}; diffusion params: {n_diffusion:,}")

    sample_step_sizes = [1 / 16, 1 / 8, 1 / 4, 1 / 2]
    run_start = time.perf_counter()
    cumulative_train_time = 0.0
    for step in range(start_step, config.n_train_steps):
        batch_start = time.perf_counter()
        indices = torch.randint(0, dataset.shape[0], (local_batch_size,))
        batch = dataset[indices].to(device)

        train_system.train()
        optimizer.zero_grad(set_to_none=True)
        loss = train_system(ema_flow_map, batch)
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        scheduler.step()
        ema_update(flow_map, ema_flow_map, config.ema_decay)

        reported_loss = loss.detach().clone()
        if distributed:
            dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
            reported_loss /= world_size
        loss_value = float(reported_loss)
        batch_time = time.perf_counter() - batch_start
        cumulative_train_time += batch_time
        log_data = {
            "loss": loss_value,
            "train/batch": step + 1,
            "train/epoch": (step + 1) * config.batch_size / dataset.shape[0],
            "train/learning_rate": scheduler.get_last_lr()[0],
            "time/batch_sec": batch_time,
            "time/train_sec": cumulative_train_time,
            "time/wall_sec": time.perf_counter() - run_start,
            "performance/samples_per_sec": config.batch_size / max(batch_time, 1e-12),
        }
        if rank == 0 and step % 1000 == 0:
            print(
                f"Step {step:>6d} | Loss: {loss_value:.4f} | Batch: {batch_time:.2f}s | "
                f"Throughput: {log_data['performance/samples_per_sec']:.1f} img/s"
            )
        sample_figure = None
        if rank == 0 and step > 0 and step % config.sample_every == 0:
            sample_figure = make_sample_grid(
                ema_flow_map,
                config.t_eps,
                data_shape,
                sample_step_sizes,
                data_mean,
                data_std,
                device=device,
            )
            log_data["samples"] = wandb.Image(sample_figure)
        if run is not None:
            run.log(log_data, step=step)
        if sample_figure is not None:
            plt.close(sample_figure)
        if rank == 0 and step > 0 and step % config.checkpoint_every == 0:
            save_checkpoint(
                os.path.join(config.checkpoint_dir, format_step(step)),
                flow_map,
                ema_flow_map,
                uncertainty_mlp,
                optimizer,
                scheduler,
                step,
                asdict(config),
            )

    if rank == 0:
        save_checkpoint(
            os.path.join(config.checkpoint_dir, format_step(config.n_train_steps)),
            flow_map,
            ema_flow_map,
            uncertainty_mlp,
            optimizer,
            scheduler,
            config.n_train_steps,
            asdict(config),
        )
        if run is not None:
            run.finish()
    if distributed:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()
    return TrainConfig(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        batch_size=args.batch_size,
        n_train_steps=args.steps,
        device=args.device,
        seed=args.seed,
        wandb_mode=args.wandb_mode,
    )


if __name__ == "__main__":
    train(parse_args())

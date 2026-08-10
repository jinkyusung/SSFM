"""Native Torch checkpoint helpers."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn


def format_step(step: int) -> str:
    return f"{step // 1000}k" if step >= 1000 and step % 1000 == 0 else str(step)


def parse_step(name: str) -> int | None:
    try:
        return int(name[:-1]) * 1000 if name.endswith("k") else int(name)
    except ValueError:
        return None


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = []
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        step = parse_step(name)
        if os.path.isdir(path) and step is not None and os.path.isfile(os.path.join(path, "state.pt")):
            candidates.append((step, path))
    return max(candidates, default=(-1, None))[1]


def save_checkpoint(
    path: str,
    flow_map: nn.Module,
    ema_flow_map: nn.Module,
    uncertainty_mlp: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    config: dict[str, Any],
) -> None:
    os.makedirs(path, exist_ok=True)
    torch.save(
        {
            "flow_map": flow_map.state_dict(),
            "ema_flow_map": ema_flow_map.state_dict(),
            "uncertainty_mlp": uncertainty_mlp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        os.path.join(path, "state.pt"),
    )


def load_checkpoint(
    path: str,
    flow_map: nn.Module,
    ema_flow_map: nn.Module,
    uncertainty_mlp: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    state = torch.load(os.path.join(path, "state.pt"), map_location=map_location, weights_only=False)
    flow_map.load_state_dict(state["flow_map"])
    ema_flow_map.load_state_dict(state["ema_flow_map"])
    uncertainty_mlp.load_state_dict(state["uncertainty_mlp"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng_state"].cpu())
    if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["cuda_rng_state"]])
    return state

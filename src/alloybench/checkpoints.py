"""Checkpoint persistence for the trained potentials."""

from __future__ import annotations

from pathlib import Path

import torch

from alloybench.models import compact_from_config, pairnet_from_config


def save_checkpoint(path: str | Path, model, train_info: dict | None = None) -> None:
    """Save a model with its config and training provenance."""
    payload = {
        "config": model.config(),
        "state": model.state_dict(),
        "train_info": dict(train_info or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(path: str | Path):
    """Load a model checkpoint.

    Returns:
        (model, train_info); the model is in eval mode.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    cfg = payload["config"]
    if cfg["kind"] == "compact":
        model = compact_from_config(cfg)
    elif cfg["kind"] == "pairnet":
        model = pairnet_from_config(cfg)
    else:
        raise ValueError(f"unknown checkpoint kind {cfg['kind']!r}")
    model.load_state_dict(payload["state"])
    model.eval()
    return model, payload.get("train_info", {})

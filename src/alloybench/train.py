"""Training loop for the compact and pair potentials.

Both models train under the identical protocol: Adam on a combined
per-atom-energy plus force MSE loss, the same epoch budget, the same
hyperparameter grid screened at a reduced epoch count on the validation
split, and the winner retrained at the full budget. Selection uses one
fixed scalar score for every model so no model is hand-tuned.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from alloybench.frames import Frame
from alloybench.models import (
    CompactPotential,
    FrameTensors,
    PairNet,
    fit_element_refs,
    prepare,
    set_element_refs,
)


def prepare_frames(frames: list[Frame], cutoff: float, angular_cutoff: float) -> list[FrameTensors]:
    """Precompute FrameTensors (static geometry) for a list of frames."""
    return [prepare(fr, cutoff, angular_cutoff) for fr in frames]


def frame_loss(
    model: torch.nn.Module, ft: FrameTensors, force_weight: float
) -> tuple[torch.Tensor, float, float]:
    """Combined loss of one frame plus detached per-frame error components."""
    positions = ft.positions.detach().clone().requires_grad_(True)
    energy = model(ft, positions, None)
    (grad,) = torch.autograd.grad(energy, positions, create_graph=True)
    forces = -grad
    n = ft.n_atoms
    e_err = (energy - ft.energy) / n
    e_loss = e_err**2
    f_loss = ((forces - ft.forces) ** 2).mean()
    loss = e_loss + force_weight * f_loss
    return loss, float(e_err.detach().abs()), float((forces - ft.forces).detach().abs().mean())


def train_potential(
    model: torch.nn.Module,
    train_fts: list[FrameTensors],
    val_fts: list[FrameTensors],
    epochs: int,
    lr: float,
    force_weight: float,
    batch_size: int = 8,
    seed: int = 0,
) -> dict:
    """Train in place; returns the history dict with timings.

    The learning rate is halved at 60% and 85% of the budget.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(0.6 * epochs), int(0.85 * epochs)], gamma=0.5
    )
    history: dict = {"epoch": [], "train_e_mae": [], "train_f_mae": [], "val_score": []}
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_fts))
        e_maes, f_maes = [], []
        for lo in range(0, len(order), batch_size):
            opt.zero_grad()
            batch = order[lo : lo + batch_size]
            total = None
            for k in batch:
                loss, e_mae, f_mae = frame_loss(model, train_fts[k], force_weight)
                total = loss if total is None else total + loss
                e_maes.append(e_mae)
                f_maes.append(f_mae)
            (total / len(batch)).backward()
            opt.step()
        sched.step()
        history["epoch"].append(epoch)
        history["train_e_mae"].append(float(np.mean(e_maes)))
        history["train_f_mae"].append(float(np.mean(f_maes)))
        history["val_score"].append(selection_score(model, val_fts))
    history["seconds"] = time.perf_counter() - start
    model.eval()
    return history


def selection_score(model: torch.nn.Module, fts: list[FrameTensors]) -> float:
    """Fixed selection scalar: E MAE (meV/atom) + 0.3 * F MAE (meV/A)."""
    from alloybench.evaluate import evaluate_model

    m = evaluate_model(model, fts)
    return float(m["e_mae_mev_atom"] + 0.3 * m["f_mae_mev_a"])


def build_model(kind: str, seed: int = 0) -> torch.nn.Module:
    """Fresh untrained model with seeded initialization (process-independent)."""
    torch.manual_seed(seed)
    if kind == "compact":
        return CompactPotential()
    if kind == "pairnet":
        return PairNet()
    raise ValueError(f"unknown model kind {kind!r}")


def fit_and_select(
    kind: str,
    train_frames: list[Frame],
    val_frames: list[Frame],
    grid: list[dict],
    screen_epochs: int,
    final_epochs: int,
    batch_size: int = 8,
    seed: int = 0,
) -> tuple[torch.nn.Module, dict]:
    """Screen the grid at reduced epochs, retrain the winner at full budget.

    Returns:
        (model, info) where info records the screen, the chosen config,
        parameter count, and wall timings.
    """
    probe = build_model(kind)
    cutoff = probe.cutoff
    angular = getattr(probe, "angular_cutoff", cutoff)
    train_fts = prepare_frames(train_frames, cutoff, angular)
    val_fts = prepare_frames(val_frames, cutoff, angular)
    refs = fit_element_refs(train_frames)

    screen = []
    for cfg in grid:
        model = build_model(kind, seed=seed)
        set_element_refs(model, refs)
        if hasattr(model, "fit_scaler"):
            model.fit_scaler(train_fts)
        hist = train_potential(
            model,
            train_fts,
            val_fts,
            epochs=screen_epochs,
            lr=cfg["lr"],
            force_weight=cfg["force_weight"],
            batch_size=batch_size,
            seed=seed,
        )
        screen.append({**cfg, "val_score": hist["val_score"][-1], "seconds": hist["seconds"]})
    best = min(screen, key=lambda s: s["val_score"])

    model = build_model(kind, seed=seed)
    set_element_refs(model, refs)
    if hasattr(model, "fit_scaler"):
        model.fit_scaler(train_fts)
    hist = train_potential(
        model,
        train_fts,
        val_fts,
        epochs=final_epochs,
        lr=best["lr"],
        force_weight=best["force_weight"],
        batch_size=batch_size,
        seed=seed,
    )
    info = {
        "kind": kind,
        "screen": screen,
        "chosen": {"lr": best["lr"], "force_weight": best["force_weight"]},
        "screen_epochs": screen_epochs,
        "final_epochs": final_epochs,
        "batch_size": batch_size,
        "seed": seed,
        "n_parameters": model.n_parameters(),
        "element_refs_ev": [float(r) for r in refs],
        "train_seconds": hist["seconds"],
        "history": {k: hist[k] for k in ("epoch", "train_e_mae", "train_f_mae", "val_score")},
    }
    return model, info

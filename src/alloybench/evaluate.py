"""Accuracy metrics against teacher labels: meV/atom and meV/Angstrom."""

from __future__ import annotations

import numpy as np
import torch

from alloybench.models import FrameTensors, predict_frame


def evaluate_model(
    model: torch.nn.Module, fts: list[FrameTensors], collect_parity: bool = False
) -> dict:
    """Energy and force errors of a model on prepared labeled frames.

    Args:
        model: Trained potential.
        fts: Prepared frames with labels.
        collect_parity: Also return per-frame energies and flattened force
            components for parity plots.

    Returns:
        Dict with e_mae_mev_atom, e_rmse_mev_atom, f_mae_mev_a,
        f_rmse_mev_a, and optionally parity arrays.
    """
    e_true, e_pred, n_atoms = [], [], []
    f_true_parts, f_pred_parts = [], []
    for ft in fts:
        out = predict_frame(model, ft, forces=True, stress=False)
        e_true.append(float(ft.energy))
        e_pred.append(float(out["energy"].detach()))
        n_atoms.append(ft.n_atoms)
        f_true_parts.append(ft.forces.numpy().ravel())
        f_pred_parts.append(out["forces"].detach().numpy().ravel())
    e_true = np.array(e_true)
    e_pred = np.array(e_pred)
    n_atoms = np.array(n_atoms, dtype=float)
    de = (e_pred - e_true) / n_atoms * 1000.0
    f_true = np.concatenate(f_true_parts)
    f_pred = np.concatenate(f_pred_parts)
    df = (f_pred - f_true) * 1000.0
    out = {
        "n_frames": len(fts),
        "e_mae_mev_atom": float(np.abs(de).mean()),
        "e_rmse_mev_atom": float(np.sqrt((de**2).mean())),
        "f_mae_mev_a": float(np.abs(df).mean()),
        "f_rmse_mev_a": float(np.sqrt((df**2).mean())),
    }
    if collect_parity:
        out["parity"] = {
            "e_true_ev_atom": np.round(e_true / n_atoms, 6).tolist(),
            "e_pred_ev_atom": np.round(e_pred / n_atoms, 6).tolist(),
            "f_true_ev_a": np.round(f_true, 5).tolist(),
            "f_pred_ev_a": np.round(f_pred, 5).tolist(),
        }
    return out

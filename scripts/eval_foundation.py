"""Force and energy disagreement of MACE-MP-0 small against the teacher.

The trained models are scored against held-out teacher labels; this script
gives the foundation-model point of the same comparison. Total-energy zeros
differ between foundation potentials, so a 3-parameter per-element offset is
fitted on the VALIDATION split and applied to the test split; forces need no
alignment. The teacher itself scores zero on its own labels by construction,
which the Pareto figure states rather than hides. This disagreement is
model-versus-model, not an error against the ground truth.

Usage:
    python scripts/eval_foundation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    import yaml

    from alloybench.calculators import get_calculator
    from alloybench.frames import read_frames, to_atoms
    from alloybench.splits import group_split

    cfg = yaml.safe_load((REPO / "configs" / "train.yaml").read_text())
    frames = read_frames(REPO / cfg["dataset"])
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=int(cfg["split_seed"]),
    )
    calc, info = get_calculator("mace-mp0-small")

    def predict(indices: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        energies, forces = [], []
        for k in indices:
            atoms = to_atoms(frames[k])
            atoms.calc = calc
            energies.append(float(atoms.get_potential_energy()))
            forces.append(np.asarray(atoms.get_forces()))
        return np.array(energies), forces

    t0 = time.perf_counter()
    e_val, _ = predict(split.val)
    counts_val = np.array(
        [[(frames[k].species == z).sum() for z in range(3)] for k in split.val], dtype=float
    )
    teacher_val = np.array([frames[k].energy for k in split.val])
    offsets, *_ = np.linalg.lstsq(counts_val, e_val - teacher_val, rcond=None)

    e_test, f_test = predict(split.test)
    counts_test = np.array(
        [[(frames[k].species == z).sum() for z in range(3)] for k in split.test], dtype=float
    )
    teacher_test = np.array([frames[k].energy for k in split.test])
    n_atoms = counts_test.sum(axis=1)
    de = (e_test - counts_test @ offsets - teacher_test) / n_atoms * 1000.0
    df_parts = [(f_test[i] - frames[k].forces).ravel() * 1000.0 for i, k in enumerate(split.test)]
    df = np.concatenate(df_parts)
    payload = {
        "model": info,
        "reference": "teacher labels (chgnet) on the held-out group-split test set",
        "note": (
            "Model-versus-model disagreement, not error against ground truth. "
            "Per-element energy offsets fitted on the validation split."
        ),
        "element_offsets_ev": [float(o) for o in offsets],
        "n_test_frames": len(split.test),
        "e_mae_mev_atom": float(np.abs(de).mean()),
        "e_rmse_mev_atom": float(np.sqrt((de**2).mean())),
        "f_mae_mev_a": float(np.abs(df).mean()),
        "f_rmse_mev_a": float(np.sqrt((df**2).mean())),
        "wall_seconds": time.perf_counter() - t0,
    }
    out = REPO / "results" / "foundation_accuracy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("e_mae_mev_atom", "f_rmse_mev_a")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Choose the labeling teacher by measured CPU speed and record provenance.

Both candidate universal potentials (chgnet, MACE-MP-0 small) are timed on
one energy+forces call for a 54-atom TiZrNb cell; the faster one becomes
the teacher. Package, version, checkpoint, and license are recorded to
data/teacher_choice.json. Labels made with the winner are surrogate model
labels, never DFT.

Usage:
    python scripts/choose_teacher.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from alloybench.calculators import get_calculator
    from alloybench.labeling import time_calculator
    from alloybench.lattice import bcc_supercell

    frame = bcc_supercell("TiZrNb", reps=3, seed=20260807)
    rows = []
    for name in ("chgnet", "mace-mp0-small"):
        calc, info = get_calculator(name)
        seconds = time_calculator(calc, frame, repeats=5)
        rows.append({"name": name, "seconds_per_call_54_atoms": round(seconds, 4), **info})
        print(f"{name}: {seconds:.3f} s per energy+forces call (54 atoms)", flush=True)
    winner = min(rows, key=lambda r: r["seconds_per_call_54_atoms"])
    payload = {
        "choice": winner["name"],
        "criterion": "faster CPU wall time per energy+forces call, measured in this session",
        "candidates": rows,
        "note": "Labels produced with the chosen teacher are surrogate model labels, not DFT.",
    }
    out = REPO / "data" / "teacher_choice.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"teacher: {winner['name']} -> wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

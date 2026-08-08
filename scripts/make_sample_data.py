"""Carve the committed sample from the full generated dataset.

Takes a deterministic, stratified subset (every composition and batch kind
represented) small enough to commit, so the demo and the committed-artifact
regression tests run offline.

Usage:
    python scripts/make_sample_data.py [n_frames]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(n_target: int = 288) -> int:
    from alloybench.frames import read_frames, write_frames
    from alloybench.splits import batch_kind

    frames = read_frames(REPO / "data" / "full" / "dataset.extxyz")
    rng = np.random.default_rng(20260808)
    by_stratum: dict[tuple[str, str], list[int]] = {}
    for k, fr in enumerate(frames):
        by_stratum.setdefault((fr.composition, batch_kind(fr.group, fr.composition)), []).append(k)
    per = max(1, n_target // len(by_stratum))
    chosen: list[int] = []
    for key in sorted(by_stratum):
        idx = np.array(by_stratum[key])
        take = min(per, len(idx))
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    chosen = sorted(chosen)
    write_frames(REPO / "data" / "sample_frames.extxyz", [frames[k] for k in chosen])
    print(f"wrote {len(chosen)} frames (from {len(frames)}) to data/sample_frames.extxyz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 288))

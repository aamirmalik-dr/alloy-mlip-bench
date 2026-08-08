"""Reproduce every committed number from scratch, in order.

Steps: teacher choice, dataset generation and labeling, training with the
grid screen, the split-protocol control, the full property suite for all
four models, metrics, and figures. Expect a few hours of CPU time; each
step is skipped if its output already exists only where noted (the suite
resumes per property).

Usage:
    python scripts/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(args: list[str]) -> None:
    print(f"\n=== {' '.join(args)} ===", flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=REPO)


def main() -> int:
    run(["scripts/choose_teacher.py"])
    run(["-m", "alloybench.cli", "generate"])
    run(["-m", "alloybench.cli", "train"])
    run(["-m", "alloybench.cli", "splitgap"])
    run(
        [
            "-m",
            "alloybench.cli",
            "suite",
            "--models",
            "pairnet",
            "compact",
            "chgnet",
            "mace-mp0-small",
        ]
    )
    run(["scripts/eval_foundation.py"])
    run(["scripts/make_sample_data.py"])
    run(["scripts/make_metrics.py"])
    run(["scripts/make_figures.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

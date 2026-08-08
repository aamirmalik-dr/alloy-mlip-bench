"""CPU speed benchmark: milliseconds per atom per MD step.

The measured quantity is the wall cost of real velocity-Verlet MD driven
through the ASE interface, which is the number that matters for deploying
a potential, and the x-axis of the accuracy-versus-speed Pareto plot.
"""

from __future__ import annotations

import time

import numpy as np

from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell


def speed_benchmark(
    calc,
    composition: str,
    a0: float,
    reps: int = 3,
    seed: int = 0,
    timestep_fs: float = 2.0,
    warmup_steps: int = 20,
    timed_steps: int = 100,
    temperature_k: float = 300.0,
) -> dict:
    """Wall-clock MD throughput of one calculator on a fixed structure."""
    from ase import units
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.md.verlet import VelocityVerlet

    frame = bcc_supercell(composition, reps, seed=seed, a=a0)
    n = frame.n_atoms
    atoms = to_atoms(frame)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.default_rng(seed)
    )
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    dyn.run(warmup_steps)
    start = time.perf_counter()
    dyn.run(timed_steps)
    wall = time.perf_counter() - start
    return {
        "composition": composition,
        "n_atoms": n,
        "timed_steps": timed_steps,
        "wall_seconds": wall,
        "ms_per_step": wall / timed_steps * 1000.0,
        "ms_per_atom_step": wall / (timed_steps * n) * 1000.0,
    }

"""Teacher labeling with a pretrained universal potential.

Every energy and force label in this project is a SURROGATE MODEL LABEL
from a pip-installed universal potential that was itself trained on DFT
data. The labels are never DFT and are never described as DFT. Real DFT
enters only through the Materials Project validation anchors.
"""

from __future__ import annotations

import time

import numpy as np

from alloybench.frames import Frame, from_atoms, to_atoms


def label_frames(frames: list[Frame], calc) -> float:
    """Label frames in place with teacher energies and forces.

    Returns:
        Mean wall-clock seconds per frame.
    """
    start = time.perf_counter()
    for fr in frames:
        atoms = to_atoms(fr)
        atoms.calc = calc
        fr.energy = float(atoms.get_potential_energy())
        fr.forces = np.asarray(atoms.get_forces())
    return (time.perf_counter() - start) / max(1, len(frames))


def time_calculator(calc, frame: Frame, repeats: int = 5) -> float:
    """Mean wall seconds for one energy+forces call on frame.

    Each repeat perturbs the positions slightly; ASE calculators cache
    results for unchanged structures, which would otherwise time as zero.
    """
    rng = np.random.default_rng(0)

    def one_call() -> None:
        atoms = to_atoms(frame)
        atoms.positions = atoms.positions + rng.normal(0.0, 1e-3, atoms.positions.shape)
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.get_forces()

    for _ in range(2):
        one_call()
    start = time.perf_counter()
    for _ in range(repeats):
        one_call()
    return (time.perf_counter() - start) / repeats


def teacher_md_batch(
    composition: str,
    reps: int,
    calc,
    temperature_k: float,
    n_steps: int,
    sample_every: int,
    seed: int,
    batch: int,
    timestep_fs: float = 3.0,
) -> list[Frame]:
    """Short teacher-driven Langevin MD; snapshots form one generation group."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from alloybench.lattice import bcc_supercell

    fr0 = bcc_supercell(composition, reps, seed=seed)
    atoms = to_atoms(fr0)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.default_rng(seed)
    )
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=0.02,
        rng=np.random.default_rng(seed + 1),
    )
    frames: list[Frame] = []

    def snapshot() -> None:
        fr = from_atoms(atoms, composition=composition)
        fr.group = f"{composition}_md{int(temperature_k)}_b{batch}"
        frames.append(fr)

    dyn.attach(snapshot, interval=sample_every)
    dyn.run(n_steps)
    return frames

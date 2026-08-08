"""BCC TiZrNb structure generation: supercells, rattle and strain batches.

The dataset recipe (compositions, cell sizes, amplitudes, temperatures, and
the master seed) is kept identical to the other repos of this TiZrNb MLIP
cluster so cross-repo benchmark numbers stay internally consistent. The code
here is a fresh implementation; nothing is imported from a sibling project.
"""

from __future__ import annotations

import numpy as np

from alloybench.frames import ELEMENTS, Frame

# Approximate BCC lattice constants (Angstrom) used only to seed generation;
# each model's own equilibrium geometry is measured later by the EOS benchmark.
BCC_A = {"Ti": 3.25, "Zr": 3.57, "Nb": 3.31}

COMPOSITIONS: dict[str, tuple[float, float, float]] = {
    "Ti": (1.0, 0.0, 0.0),
    "Zr": (0.0, 1.0, 0.0),
    "Nb": (0.0, 0.0, 1.0),
    "TiZr": (0.5, 0.5, 0.0),
    "TiNb": (0.5, 0.0, 0.5),
    "ZrNb": (0.0, 0.5, 0.5),
    "TiZrNb": (1 / 3, 1 / 3, 1 / 3),
    "Ti2ZrNb": (0.5, 0.25, 0.25),
    "TiZrNb2": (0.25, 0.25, 0.5),
}


def vegard_a(composition: str) -> float:
    """Composition-weighted BCC lattice constant (Vegard rule)."""
    frac = COMPOSITIONS[composition]
    return float(sum(f * BCC_A[el] for f, el in zip(frac, ELEMENTS)))


def bcc_supercell(composition: str, reps: int, seed: int, a: float | None = None) -> Frame:
    """Ideal BCC supercell with a seeded random site occupancy.

    Args:
        composition: Key into COMPOSITIONS.
        reps: Cubic repetitions per axis; atoms = 2 * reps**3.
        seed: RNG seed for the occupancy shuffle (same seed, same structure).
        a: Lattice constant override; defaults to the Vegard estimate.

    Returns:
        Unlabeled Frame on ideal lattice sites.
    """
    rng = np.random.default_rng(seed)
    a0 = vegard_a(composition) if a is None else float(a)
    grid = np.stack(
        np.meshgrid(np.arange(reps), np.arange(reps), np.arange(reps), indexing="ij"), axis=-1
    ).reshape(-1, 3)
    frac = np.concatenate([grid, grid + 0.5], axis=0) / reps
    n_atoms = len(frac)
    target = np.array(COMPOSITIONS[composition])
    counts = np.floor(target * n_atoms).astype(int)
    while counts.sum() < n_atoms:
        counts[int(np.argmax(target - counts / n_atoms))] += 1
    species = np.repeat(np.arange(3), counts)
    rng.shuffle(species)
    cell = np.eye(3) * a0 * reps
    return Frame(species=species, positions=frac @ cell, cell=cell, composition=composition)


def rattle_batch(
    composition: str, reps: int, amplitude: float, n_frames: int, seed: int, batch: int
) -> list[Frame]:
    """Gaussian-displaced frames from independent random occupancies."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n_frames):
        fr = bcc_supercell(composition, reps, seed=int(rng.integers(2**31)))
        fr.positions = fr.positions + rng.normal(0.0, amplitude, fr.positions.shape)
        fr.group = f"{composition}_rattle{int(round(amplitude * 100)):02d}_b{batch}"
        frames.append(fr)
    return frames


def strain_batch(
    composition: str,
    reps: int,
    n_frames: int,
    seed: int,
    batch: int,
    max_volumetric: float = 0.06,
    max_shear: float = 0.05,
    jitter: float = 0.03,
) -> list[Frame]:
    """Alternating volumetric and shear strains with a symmetry-breaking rattle."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(n_frames):
        fr = bcc_supercell(composition, reps, seed=int(rng.integers(2**31)))
        strain = np.eye(3)
        if k % 2 == 0:
            strain *= 1.0 + rng.uniform(-max_volumetric, max_volumetric)
            tag = "vol"
        else:
            pairs = [(0, 1), (0, 2), (1, 2)]
            i, j = pairs[int(rng.integers(0, 3))]
            strain[i, j] = rng.uniform(-max_shear, max_shear)
            tag = "shear"
        frac = fr.positions @ np.linalg.inv(fr.cell)
        fr.cell = fr.cell @ strain.T
        fr.positions = frac @ fr.cell + rng.normal(0.0, jitter, fr.positions.shape)
        fr.group = f"{composition}_strain_{tag}_b{batch}"
        frames.append(fr)
    return frames

"""Radial distribution functions from short NVT Langevin dynamics."""

from __future__ import annotations

import numpy as np

from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell
from alloybench.neighbors import neighbor_table, pair_vectors


def compute_rdf(
    snapshots: list[tuple[np.ndarray, np.ndarray]], r_max: float, n_bins: int = 120
) -> tuple[np.ndarray, np.ndarray]:
    """Total g(r) averaged over (positions, cell) snapshots.

    Ordered pairs are counted, matching the N * rho ideal-gas normalization.
    """
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins)
    norm = 0.0
    for positions, cell in snapshots:
        table = neighbor_table(positions, cell, r_max)
        d = np.linalg.norm(pair_vectors(table, positions, cell), axis=1)
        hist += np.histogram(d, bins=edges)[0]
        n = len(positions)
        volume = float(np.abs(np.linalg.det(cell)))
        norm += n * (n / volume)
    r = 0.5 * (edges[1:] + edges[:-1])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (norm * shell)
    return r, g


def first_peak(r: np.ndarray, g: np.ndarray, lo: float = 2.0, hi: float = 4.0) -> dict:
    """Position and height of the first RDF peak, parabola-refined."""
    mask = (r >= lo) & (r <= hi)
    idx = np.flatnonzero(mask)[np.argmax(g[mask])]
    if 0 < idx < len(r) - 1:
        y0, y1, y2 = g[idx - 1], g[idx], g[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
        pos = r[idx] + offset * (r[1] - r[0])
        height = y1 - 0.25 * (y0 - y2) * offset
    else:
        pos, height = r[idx], g[idx]
    return {"position_angstrom": float(pos), "height": float(height)}


def rdf_benchmark(
    calc,
    composition: str,
    temperature_k: float,
    a0: float,
    reps: int = 3,
    seed: int = 0,
    timestep_fs: float = 2.0,
    equil_steps: int = 300,
    prod_steps: int = 1000,
    sample_every: int = 10,
    n_bins: int = 120,
) -> dict:
    """NVT Langevin RDF at one temperature, at the model's own a0."""
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    frame = bcc_supercell(composition, reps, seed=seed, a=a0)
    atoms = to_atoms(frame)
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
    dyn.run(equil_steps)
    snapshots: list[tuple[np.ndarray, np.ndarray]] = []

    def snap() -> None:
        snapshots.append((np.asarray(atoms.get_positions()), np.asarray(atoms.get_cell()[:])))

    dyn.attach(snap, interval=sample_every)
    dyn.run(prod_steps)
    r_max = 0.49 * float(min(np.linalg.norm(atoms.get_cell()[:], axis=1)))
    r, g = compute_rdf(snapshots, r_max=r_max, n_bins=n_bins)
    return {
        "composition": composition,
        "temperature_k": float(temperature_k),
        "n_atoms": frame.n_atoms,
        "a0_angstrom": float(a0),
        "n_snapshots": len(snapshots),
        "timestep_fs": timestep_fs,
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "r_angstrom": r.tolist(),
        "g_r": g.tolist(),
        "first_peak": first_peak(r, g),
    }

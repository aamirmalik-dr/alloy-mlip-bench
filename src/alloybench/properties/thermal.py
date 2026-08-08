"""Thermal expansion from short NPT Berendsen runs.

The mean lattice parameter at each temperature comes from the average cell
volume over the production window; the linear expansion coefficient is the
slope of a linear fit to a(T), normalized by a at the lowest temperature.
Classical point-particle MD, so low-temperature quantum effects are out of
scope by construction.
"""

from __future__ import annotations

import numpy as np

from alloybench.calculators import EV_A3_TO_GPA
from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell


def npt_lattice_parameter(
    calc,
    composition: str,
    temperature_k: float,
    a0: float,
    b0_gpa: float,
    reps: int = 3,
    seed: int = 0,
    timestep_fs: float = 2.0,
    equil_steps: int = 300,
    prod_steps: int = 800,
    sample_every: int = 5,
) -> dict:
    """Mean conventional lattice parameter from one NPT run."""
    from ase import units
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    frame = bcc_supercell(composition, reps, seed=seed, a=a0)
    n = frame.n_atoms
    atoms = to_atoms(frame)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.default_rng(seed)
    )
    compressibility = 1.0 / (b0_gpa / EV_A3_TO_GPA)  # 1 / (eV/A^3)
    dyn = NPTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        pressure_au=0.0,
        taut=100.0 * units.fs,
        taup=1000.0 * units.fs,
        compressibility_au=compressibility,
    )
    dyn.run(equil_steps)
    volumes: list[float] = []

    def snap() -> None:
        volumes.append(float(np.abs(np.linalg.det(atoms.get_cell()[:]))))

    dyn.attach(snap, interval=sample_every)
    dyn.run(prod_steps)
    v = np.array(volumes)
    a_mean = (2.0 * v.mean() / n) ** (1.0 / 3.0)
    a_std = a_mean * float(v.std() / v.mean()) / 3.0
    return {
        "composition": composition,
        "temperature_k": float(temperature_k),
        "n_atoms": n,
        "timestep_fs": timestep_fs,
        "equil_steps": equil_steps,
        "prod_steps": prod_steps,
        "a_mean_angstrom": float(a_mean),
        "a_std_angstrom": a_std,
        "n_samples": len(volumes),
    }


def expansion_coefficient(temps_k: list[float], a_means: list[float]) -> float:
    """Linear expansion coefficient (1/K) from a(T), normalized at min T."""
    t = np.asarray(temps_k, dtype=float)
    a = np.asarray(a_means, dtype=float)
    slope = np.polyfit(t, a, 1)[0]
    return float(slope / a[np.argmin(t)])

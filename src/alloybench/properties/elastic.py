"""Cubic elastic constants by finite differences of the total energy.

The bulk modulus comes from the EOS fit; the two shear constants come from
volume-conserving deformations about each model's OWN equilibrium lattice
constant, so every model is probed at its own minimum:

    orthorhombic  diag(1+d, 1-d, 1/(1-d^2)):  dE = (C11 - C12) V d^2
    monoclinic    xy shear d with zz correction:  dE = C44 V d^2 / 2

For the random alloy, internal coordinates are relaxed at every deformed
cell (and in the reference), since local relaxations carry real elastic
response in a solid solution. C11 and C12 are then recovered from
B = (C11 + 2 C12)/3 and C' = (C11 - C12)/2.
"""

from __future__ import annotations

import numpy as np

from alloybench.calculators import EV_A3_TO_GPA
from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell
from alloybench.properties.relax import relax_positions


def _deformed_energy(atoms0, calc, deform: np.ndarray, relax_internal: bool, fmax, steps) -> float:
    atoms = atoms0.copy()
    cell = atoms.get_cell()[:] @ deform.T
    atoms.set_cell(cell, scale_atoms=True)
    atoms.calc = calc
    if relax_internal:
        relax_positions(atoms, fmax=fmax, steps=steps)
    return float(atoms.get_potential_energy())


def _fit_even_quadratic(deltas: np.ndarray, denergies: np.ndarray) -> tuple[float, float]:
    """Least-squares k in dE = k d^2; returns (k, rms residual)."""
    x = deltas**2
    k = float((x * denergies).sum() / (x * x).sum())
    resid = denergies - k * x
    return k, float(np.sqrt((resid**2).mean()))


def elastic_constants(
    calc,
    composition: str,
    a0: float,
    b0_gpa: float,
    reps: int = 3,
    seed: int = 0,
    deltas: tuple[float, ...] = (0.006, 0.012),
    relax_internal: bool = False,
    fmax: float = 0.03,
    relax_steps: int = 120,
) -> dict:
    """C11, C12, C44 in GPa about the model's own equilibrium.

    Args:
        calc: ASE calculator.
        composition: Composition tag (occupancy fixed by seed).
        a0: This model's own equilibrium conventional lattice constant.
        b0_gpa: This model's own EOS bulk modulus.
        reps: Supercell repetitions.
        seed: Occupancy seed (same for every model).
        deltas: Deformation magnitudes; each is applied with both signs.
        relax_internal: Relax positions at every deformed cell (alloys).
        fmax: Relaxation force threshold.
        relax_steps: Maximum relaxation steps.
    """
    frame = bcc_supercell(composition, reps, seed=seed, a=a0)
    atoms0 = to_atoms(frame)
    atoms0.calc = calc
    if relax_internal:
        relax_positions(atoms0, fmax=fmax, steps=relax_steps)
    e_ref = float(atoms0.get_potential_energy())
    volume = float(np.abs(np.linalg.det(atoms0.get_cell()[:])))

    signed = np.array([s * d for d in deltas for s in (+1.0, -1.0)])
    ortho_de, mono_de = [], []
    for d in signed:
        ortho = np.diag([1.0 + d, 1.0 - d, 1.0 / (1.0 - d * d)])
        mono = np.array(
            [[1.0, d / 2.0, 0.0], [d / 2.0, 1.0, 0.0], [0.0, 0.0, 1.0 / (1.0 - d * d / 4.0)]]
        )
        ortho_de.append(
            _deformed_energy(atoms0, calc, ortho, relax_internal, fmax, relax_steps) - e_ref
        )
        mono_de.append(
            _deformed_energy(atoms0, calc, mono, relax_internal, fmax, relax_steps) - e_ref
        )
    k_ortho, rms_ortho = _fit_even_quadratic(signed, np.array(ortho_de))
    k_mono, rms_mono = _fit_even_quadratic(signed, np.array(mono_de))
    c_prime = k_ortho / (2.0 * volume) * EV_A3_TO_GPA
    c44 = 2.0 * k_mono / volume * EV_A3_TO_GPA
    return {
        "composition": composition,
        "n_atoms": frame.n_atoms,
        "a0_angstrom": float(a0),
        "b0_gpa": float(b0_gpa),
        "relax_internal": bool(relax_internal),
        "deltas": signed.tolist(),
        "ortho_denergies_ev": ortho_de,
        "mono_denergies_ev": mono_de,
        "fit_rms_ev": {"ortho": rms_ortho, "mono": rms_mono},
        "c_prime_gpa": float(c_prime),
        "c11_gpa": float(b0_gpa + 4.0 * c_prime / 3.0),
        "c12_gpa": float(b0_gpa - 2.0 * c_prime / 3.0),
        "c44_gpa": float(c44),
    }

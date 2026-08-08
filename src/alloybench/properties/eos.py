"""Equation of state and lattice constant via a Birch-Murnaghan fit.

The third-order Birch-Murnaghan energy is exactly linear in powers of
eta = V^(-2/3), so the fit is a plain least-squares polynomial with an
analytic minimum and curvature; no nonlinear optimizer is involved.
"""

from __future__ import annotations

import numpy as np

from alloybench.calculators import EV_A3_TO_GPA
from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell, vegard_a
from alloybench.properties.relax import relax_positions


def fit_birch_murnaghan(volumes: np.ndarray, energies: np.ndarray) -> dict:
    """Fit E(V) = a + b eta + c eta^2 + d eta^3 with eta = V^(-2/3).

    Returns:
        Dict with v0 (A^3), e0 (eV), b0_gpa, and the fit residual RMS (eV).
    """
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    eta = volumes ** (-2.0 / 3.0)
    coeffs = np.polyfit(eta, energies, 3)  # highest power first
    d, c, b, a = coeffs
    # dE/deta = 3 d eta^2 + 2 c eta + b = 0 at the minimum.
    roots = np.roots([3.0 * d, 2.0 * c, b])
    roots = roots[np.isreal(roots)].real
    roots = roots[roots > 0]
    if len(roots) == 0:
        raise ValueError("Birch-Murnaghan fit found no physical minimum")
    # Pick the root with positive curvature (a true minimum) closest to data.
    curv = 6.0 * d * roots + 2.0 * c
    minima = roots[curv > 0]
    if len(minima) == 0:
        raise ValueError("Birch-Murnaghan fit has no minimum with positive curvature")
    eta_mid = float(np.median(eta))
    eta0 = float(minima[np.argmin(np.abs(minima - eta_mid))])
    v0 = eta0 ** (-1.5)
    e0 = float(np.polyval(coeffs, eta0))
    # B0 = V d2E/dV2 at V0; with E(eta), d2E/dV2 = E''(eta) (deta/dV)^2 there,
    # and (deta/dV)^2 = (4/9) V^(-10/3), so B0 = (4/9) E''(eta0) V0^(-7/3).
    epp = 6.0 * d * eta0 + 2.0 * c
    b0_ev_a3 = (4.0 / 9.0) * epp * v0 ** (-7.0 / 3.0)
    resid = energies - np.polyval(coeffs, eta)
    return {
        "v0_a3": float(v0),
        "e0_ev": e0,
        "b0_gpa": float(b0_ev_a3 * EV_A3_TO_GPA),
        "fit_rms_ev": float(np.sqrt((resid**2).mean())),
    }


def eos_scan(
    calc,
    composition: str,
    reps: int = 3,
    seed: int = 0,
    span: float = 0.08,
    n_points: int = 11,
    relax_internal: bool = False,
    fmax: float = 0.03,
    relax_steps: int = 120,
    a: float | None = None,
) -> dict:
    """Energy-volume scan of a BCC cell, optionally with internal relaxation.

    The site occupancy is fixed by the seed, so every model scans the same
    structure. Internal relaxation (for alloys) runs at fixed cell. If a
    model's minimum falls outside the scanned window (a weak model can be
    badly off the Vegard guess), the scan recenters on the boundary and
    walks toward the minimum, up to max_recenter times, so every model is
    fitted around its OWN equilibrium; the walk is recorded in the payload.

    Returns:
        Dict with the final scan, the Birch-Murnaghan fit, the conventional
        lattice constant a0, and per-atom minimum energy e0_per_atom.
    """
    a_center = float(a) if a is not None else vegard_a(composition)
    max_recenter = 6
    n_scans = 0
    for attempt in range(max_recenter):
        frame = bcc_supercell(composition, reps, seed=seed, a=a_center)
        n = frame.n_atoms
        factors = np.linspace(1.0 - span, 1.0 + span, n_points)
        volumes, energies = [], []
        for f in factors:
            scale = f ** (1.0 / 3.0)
            atoms = to_atoms(frame)
            atoms.set_cell(frame.cell * scale, scale_atoms=True)
            atoms.calc = calc
            if relax_internal:
                relax_positions(atoms, fmax=fmax, steps=relax_steps)
            volumes.append(float(np.abs(np.linalg.det(atoms.get_cell()[:]))))
            energies.append(float(atoms.get_potential_energy()))
        n_scans += 1
        k = int(np.argmin(energies))
        if 0 < k < n_points - 1:
            try:
                fit = fit_birch_murnaghan(np.array(volumes), np.array(energies))
            except ValueError:
                a_center *= float(factors[k]) ** (1.0 / 3.0)
                continue
            if volumes[0] <= fit["v0_a3"] <= volumes[-1]:
                break
        # Minimum at (or beyond) the window edge: walk the center toward it.
        a_center *= float(factors[k]) ** (1.0 / 3.0)
    else:
        raise ValueError(
            f"eos_scan: no interior minimum for {composition} after "
            f"{max_recenter} recentering scans (last center {a_center:.3f} A)"
        )
    v_atom = fit["v0_a3"] / n
    return {
        "composition": composition,
        "n_atoms": n,
        "volume_factors": factors.tolist(),
        "volumes_a3": volumes,
        "energies_ev": energies,
        "relax_internal": bool(relax_internal),
        "n_scans": n_scans,
        "scan_center_a": float(a_center),
        **fit,
        "a0_angstrom": float((2.0 * v_atom) ** (1.0 / 3.0)),
        "e0_per_atom_ev": fit["e0_ev"] / n,
        "vegard_a_angstrom": vegard_a(composition),
    }

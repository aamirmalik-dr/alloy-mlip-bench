"""Fixed-cell structure relaxation shared by the property benchmarks."""

from __future__ import annotations

from ase.optimize import FIRE


def relax_positions(atoms, fmax: float = 0.03, steps: int = 120) -> dict:
    """Relax atomic positions at fixed cell with FIRE.

    Args:
        atoms: ASE Atoms with a calculator attached; modified in place.
        fmax: Force convergence threshold in eV/Angstrom.
        steps: Maximum optimizer steps.

    Returns:
        Dict with converged flag, steps taken, and final energy (eV).
    """
    opt = FIRE(atoms, logfile=None)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    return {
        "converged": converged,
        "steps": int(opt.get_number_of_steps()),
        "energy_ev": float(atoms.get_potential_energy()),
    }

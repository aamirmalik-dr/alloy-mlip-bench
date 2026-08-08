"""Monovacancy formation energies with an elemental-reservoir convention.

In an alloy the formation energy needs a reservoir for the removed atom;
here the reservoir is the per-atom energy of that element's own BCC crystal
at the SAME model's equilibrium (from its EOS), so every model is scored
against its own thermodynamics:

    Ef(site) = E[defect, relaxed] - E[bulk, relaxed] + e_atom[species removed]

For an elemental crystal this reduces to the textbook expression. The cell
is a 54-site supercell for every model; the residual finite-size error is
shared by construction, which is what a cross-model comparison needs.
"""

from __future__ import annotations

import numpy as np

from alloybench.frames import ELEMENTS, to_atoms
from alloybench.lattice import bcc_supercell
from alloybench.properties.relax import relax_positions


def vacancy_formation(
    calc,
    composition: str,
    a0: float,
    reservoir_e_atom: dict[str, float],
    reps: int = 3,
    seed: int = 0,
    n_sites: int = 1,
    fmax: float = 0.03,
    relax_steps: int = 150,
) -> dict:
    """Formation energies for one or more vacancy sites.

    Args:
        calc: ASE calculator.
        composition: Composition tag (occupancy fixed by seed).
        a0: The model's own equilibrium lattice constant.
        reservoir_e_atom: element symbol -> per-atom elemental BCC energy
            (eV) from the SAME model's EOS.
        reps: Supercell repetitions (2 * reps^3 sites).
        seed: Occupancy seed, and site-choice seed.
        n_sites: Number of distinct vacancy sites to average over.
        fmax: Relaxation force threshold (eV/A).
        relax_steps: Maximum FIRE steps per relaxation.
    """
    frame = bcc_supercell(composition, reps, seed=seed, a=a0)
    bulk = to_atoms(frame)
    bulk.calc = calc
    bulk_info = relax_positions(bulk, fmax=fmax, steps=relax_steps)

    rng = np.random.default_rng(seed + 7)
    # One site per present species first, then extra random sites.
    present = sorted(set(int(s) for s in frame.species))
    chosen: list[int] = []
    for z in present:
        candidates = np.flatnonzero(frame.species == z)
        chosen.append(int(rng.choice(candidates)))
    while len(chosen) < n_sites:
        site = int(rng.integers(0, frame.n_atoms))
        if site not in chosen:
            chosen.append(site)
    chosen = chosen[: max(n_sites, len(present)) if n_sites > 1 else 1]

    sites = []
    for site in chosen:
        element = ELEMENTS[int(frame.species[site])]
        defect = to_atoms(frame)
        del defect[site]
        defect.calc = calc
        defect_info = relax_positions(defect, fmax=fmax, steps=relax_steps)
        ef = defect_info["energy_ev"] - bulk_info["energy_ev"] + reservoir_e_atom[element]
        sites.append(
            {
                "site": site,
                "element": element,
                "ef_ev": float(ef),
                "defect_converged": defect_info["converged"],
                "defect_relax_steps": defect_info["steps"],
            }
        )
    efs = np.array([s["ef_ev"] for s in sites])
    by_element = {
        el: float(np.mean([s["ef_ev"] for s in sites if s["element"] == el]))
        for el in sorted({s["element"] for s in sites})
    }
    return {
        "composition": composition,
        "n_sites_total": frame.n_atoms,
        "a0_angstrom": float(a0),
        "bulk_converged": bulk_info["converged"],
        "reservoir_e_atom_ev": {k: float(v) for k, v in reservoir_e_atom.items()},
        "sites": sites,
        "ef_mean_ev": float(efs.mean()),
        "ef_std_ev": float(efs.std()),
        "ef_by_element_ev": by_element,
    }

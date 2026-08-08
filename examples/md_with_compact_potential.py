"""Run your own MD with the committed compact potential.

Works offline from a fresh clone: loads the committed checkpoint, builds a
BCC TiZrNb cell, runs a short NVT trajectory, and prints the temperature
trace. Swap "compact" for "chgnet" or "mace-mp0-small" (with the foundation
extra installed) and the rest of the script is unchanged; that is the point
of the calculator interface.

Usage:
    python examples/md_with_compact_potential.py
"""

from __future__ import annotations

import numpy as np
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from alloybench import get_calculator
from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell


def main() -> int:
    calc, info = get_calculator("compact")
    print(f"model: {info['package']}, {info['n_parameters']} parameters ({info['license']})")

    atoms = to_atoms(bcc_supercell("TiZrNb", reps=3, seed=1))
    atoms.calc = calc
    MaxwellBoltzmannDistribution(atoms, temperature_K=600, rng=np.random.default_rng(1))
    dyn = Langevin(
        atoms,
        timestep=2 * units.fs,
        temperature_K=600,
        friction=0.02,
        rng=np.random.default_rng(2),
    )

    def report() -> None:
        step = dyn.get_number_of_steps()
        print(
            f"step {step:4d}: T = {atoms.get_temperature():6.1f} K, "
            f"E_pot = {atoms.get_potential_energy() / len(atoms):8.4f} eV/atom"
        )

    report()
    dyn.attach(report, interval=50)
    dyn.run(200)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

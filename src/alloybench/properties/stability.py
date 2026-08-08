"""NVE energy-drift stability gate.

A deployable potential must conserve total energy under plain velocity
Verlet. Every benchmarked model runs the same NVE protocol; the gate is a
threshold on the linear drift of total energy per atom. Models that fail
are reported as failing, not dropped.
"""

from __future__ import annotations

import numpy as np

from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell


def nve_drift(
    calc,
    composition: str,
    a0: float,
    temperature_k: float = 300.0,
    reps: int = 3,
    seed: int = 0,
    timestep_fs: float = 2.0,
    steps: int = 1500,
    record_every: int = 5,
    settle_steps: int = 100,
    gate_mev_atom_ps: float = 1.0,
) -> dict:
    """Total-energy drift under velocity Verlet.

    Returns:
        Dict with the energy trace, the linear drift in meV/atom/ps
        (fitted after settle_steps), and the pass/fail gate.
    """
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
    times_ps: list[float] = []
    e_tot: list[float] = []

    def record() -> None:
        times_ps.append(dyn.get_number_of_steps() * timestep_fs / 1000.0)
        e_tot.append((atoms.get_potential_energy() + atoms.get_kinetic_energy()) / n)

    record()
    dyn.attach(record, interval=record_every)
    dyn.run(steps)
    t = np.array(times_ps)
    e = np.array(e_tot)
    fit_mask = t >= settle_steps * timestep_fs / 1000.0
    slope_ev_ps = float(np.polyfit(t[fit_mask], e[fit_mask], 1)[0])
    drift = abs(slope_ev_ps) * 1000.0
    rms_fluct = float(np.std(e[fit_mask])) * 1000.0
    return {
        "composition": composition,
        "temperature_k": float(temperature_k),
        "n_atoms": n,
        "timestep_fs": timestep_fs,
        "steps": steps,
        "total_ps": steps * timestep_fs / 1000.0,
        "times_ps": t.tolist(),
        "e_tot_ev_atom": e.tolist(),
        "drift_mev_atom_ps": drift,
        "rms_fluctuation_mev_atom": rms_fluct,
        "gate_mev_atom_ps": gate_mev_atom_ps,
        "gate_passed": bool(drift < gate_mev_atom_ps),
    }

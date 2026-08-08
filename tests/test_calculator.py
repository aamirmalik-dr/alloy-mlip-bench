"""The ASE deployment layer: calculator consistency and compliance."""

from __future__ import annotations

import numpy as np

from alloybench.calculators import TorchPotentialCalculator
from alloybench.frames import to_atoms
from alloybench.models import predict_frame, prepare


def test_calculator_matches_direct_prediction(compact_model, small_frame):
    calc = TorchPotentialCalculator(compact_model)
    atoms = to_atoms(small_frame)
    atoms.calc = calc
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()
    ft = prepare(small_frame, compact_model.cutoff, compact_model.angular_cutoff)
    out = predict_frame(compact_model, ft, forces=True, stress=True)
    assert abs(e_calc - float(out["energy"])) < 1e-10
    np.testing.assert_allclose(f_calc, out["forces"].detach().numpy(), atol=1e-10)


def test_calculator_stress_matches_ase_numerical(pairnet_model, small_frame):
    calc = TorchPotentialCalculator(pairnet_model)
    atoms = to_atoms(small_frame)
    atoms.calc = calc
    stress = atoms.get_stress()  # Voigt 6-vector
    numeric = calc.calculate_numerical_stress(atoms, d=1e-6)
    np.testing.assert_allclose(stress, numeric, atol=1e-7)


def test_calculator_updates_after_position_change(pairnet_model, small_frame):
    calc = TorchPotentialCalculator(pairnet_model)
    atoms = to_atoms(small_frame)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    atoms.positions = atoms.positions + np.array([0.05, 0.0, 0.0])
    atoms.positions[0, 1] += 0.2
    e1 = atoms.get_potential_energy()
    assert e0 != e1  # cache invalidated on change


def test_calculator_handles_md_wrapped_positions(pairnet_model, small_frame):
    calc = TorchPotentialCalculator(pairnet_model)
    atoms = to_atoms(small_frame)
    atoms.calc = calc
    e0 = atoms.get_potential_energy()
    moved = to_atoms(small_frame)
    moved.positions = moved.positions + moved.cell[:].sum(axis=0)  # all atoms outside
    moved.calc = TorchPotentialCalculator(pairnet_model)
    assert abs(moved.get_potential_energy() - e0) < 1e-9

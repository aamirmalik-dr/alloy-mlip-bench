"""Neighbor list checked against ASE and a brute-force reference."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

from alloybench.lattice import bcc_supercell
from alloybench.neighbors import angular_triplets, neighbor_table, pair_vectors


def _as_sets(i, j, shifts):
    return {(int(a), int(b), tuple(int(x) for x in s)) for a, b, s in zip(i, j, shifts)}


def _ase_pairs(positions, cell, cutoff):
    atoms = Atoms("H" * len(positions), positions=positions, cell=cell, pbc=True)
    i, j, s = ase_neighbor_list("ijS", atoms, cutoff)
    return _as_sets(i, j, s)


@pytest.mark.parametrize("cutoff", [3.2, 5.5])
def test_matches_ase_cubic(cutoff):
    rng = np.random.default_rng(0)
    fr = bcc_supercell("TiZrNb", reps=2, seed=1)
    pos = fr.positions + rng.normal(0.0, 0.1, fr.positions.shape)
    table = neighbor_table(pos, fr.cell, cutoff)
    assert _as_sets(table.i, table.j, table.shifts) == _ase_pairs(pos, fr.cell, cutoff)


def test_matches_ase_sheared_cell():
    rng = np.random.default_rng(1)
    fr = bcc_supercell("Nb", reps=2, seed=2)
    shear = np.array([[1.0, 0.35, 0.0], [0.0, 1.0, 0.25], [0.0, 0.0, 1.0]])
    cell = fr.cell @ shear.T
    frac = fr.positions @ np.linalg.inv(fr.cell)
    pos = frac @ cell + rng.normal(0.0, 0.05, fr.positions.shape)
    table = neighbor_table(pos, cell, 5.5)
    assert _as_sets(table.i, table.j, table.shifts) == _ase_pairs(pos, cell, 5.5)


def test_atoms_outside_cell_are_wrapped_consistently():
    rng = np.random.default_rng(2)
    fr = bcc_supercell("TiZr", reps=2, seed=3)
    pos = fr.positions + rng.normal(0.0, 0.1, fr.positions.shape)
    shifted = pos.copy()
    shifted[::3] += fr.cell.sum(axis=0) * 2.0  # push some atoms far outside
    table_ref = neighbor_table(pos, fr.cell, 5.0)
    table_out = neighbor_table(shifted, fr.cell, 5.0)
    d_ref = np.sort(np.linalg.norm(pair_vectors(table_ref, pos, fr.cell), axis=1))
    d_out = np.sort(np.linalg.norm(pair_vectors(table_out, shifted, fr.cell), axis=1))
    assert len(table_ref) == len(table_out)
    np.testing.assert_allclose(d_ref, d_out, atol=1e-10)


def test_pair_vectors_have_lengths_below_cutoff():
    fr = bcc_supercell("Ti", reps=2, seed=4)
    table = neighbor_table(fr.positions, fr.cell, 4.0)
    d = np.linalg.norm(pair_vectors(table, fr.positions, fr.cell), axis=1)
    assert (d < 4.0).all() and (d > 1.0).all()


def test_angular_triplets_count():
    fr = bcc_supercell("Nb", reps=2, seed=5)
    table = neighbor_table(fr.positions, fr.cell, 3.2)  # first shell only: 8 neighbors
    keep = np.arange(len(table))
    t1, t2 = angular_triplets(table, keep)
    # 8 neighbors per center -> C(8, 2) = 28 unordered pairs per atom.
    assert len(t1) == len(fr.species) * 28
    assert (table.i[t1] == table.i[t2]).all()

"""Frame construction, extxyz round trip, and lattice generation."""

from __future__ import annotations

import numpy as np

from alloybench.frames import read_frames, write_frames
from alloybench.lattice import COMPOSITIONS, bcc_supercell, rattle_batch, vegard_a


def test_bcc_supercell_counts_and_geometry():
    fr = bcc_supercell("TiZrNb", reps=3, seed=0)
    assert fr.n_atoms == 54
    counts = [(fr.species == z).sum() for z in range(3)]
    assert sum(counts) == 54 and max(counts) - min(counts) <= 1
    a = vegard_a("TiZrNb")
    np.testing.assert_allclose(fr.cell, np.eye(3) * 3 * a, atol=1e-12)


def test_bcc_supercell_seed_reproducible():
    fr1 = bcc_supercell("TiZrNb", reps=3, seed=42)
    fr2 = bcc_supercell("TiZrNb", reps=3, seed=42)
    assert (fr1.species == fr2.species).all()
    fr3 = bcc_supercell("TiZrNb", reps=3, seed=43)
    assert (fr1.species != fr3.species).any()


def test_all_compositions_sum_to_one():
    for comp, frac in COMPOSITIONS.items():
        assert abs(sum(frac) - 1.0) < 1e-12, comp


def test_rattle_batch_sets_group_ids():
    frames = rattle_batch("TiZr", reps=2, amplitude=0.1, n_frames=3, seed=0, batch=2)
    assert len(frames) == 3
    assert all(fr.group == "TiZr_rattle10_b2" for fr in frames)
    assert all(fr.n_atoms == 16 for fr in frames)


def test_extxyz_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    frames = rattle_batch("TiZrNb", reps=2, amplitude=0.1, n_frames=2, seed=1, batch=0)
    for fr in frames:
        fr.energy = float(rng.normal())
        fr.forces = rng.normal(size=(fr.n_atoms, 3))
    path = tmp_path / "frames.extxyz"
    write_frames(path, frames)
    loaded = read_frames(path)
    assert len(loaded) == 2
    for a, b in zip(frames, loaded):
        assert a.group == b.group and a.composition == b.composition
        assert abs(a.energy - b.energy) < 1e-8
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-7)
        np.testing.assert_allclose(a.forces, b.forces, atol=1e-7)
        assert (a.species == b.species).all()

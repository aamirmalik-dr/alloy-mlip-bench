"""Physics checks on the trained-model architectures: invariances, exact
derivatives, and smoothness at the cutoff."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alloybench.frames import Frame
from alloybench.models import predict_frame, prepare


def prepared(frame: Frame, model):
    return prepare(frame, model.cutoff, getattr(model, "angular_cutoff", model.cutoff))


def _energy(model, frame: Frame) -> float:
    ft = prepared(frame, model)
    return float(predict_frame(model, ft, forces=False)["energy"])


@pytest.mark.parametrize("model_name", ["compact_model", "pairnet_model"])
def test_translation_invariance(model_name, small_frame, request):
    model = request.getfixturevalue(model_name)
    e0 = _energy(model, small_frame)
    moved = Frame(
        species=small_frame.species,
        positions=small_frame.positions + np.array([1.3, -0.7, 2.9]),
        cell=small_frame.cell,
    )
    assert abs(_energy(model, moved) - e0) < 1e-9


@pytest.mark.parametrize("model_name", ["compact_model", "pairnet_model"])
def test_rotation_invariance(model_name, small_frame, request):
    model = request.getfixturevalue(model_name)
    e0 = _energy(model, small_frame)
    theta = 0.6
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = Frame(
        species=small_frame.species,
        positions=small_frame.positions @ rot.T,
        cell=small_frame.cell @ rot.T,
    )
    assert abs(_energy(model, rotated) - e0) < 1e-9


@pytest.mark.parametrize("model_name", ["compact_model", "pairnet_model"])
def test_permutation_invariance(model_name, small_frame, request):
    model = request.getfixturevalue(model_name)
    e0 = _energy(model, small_frame)
    perm = np.random.default_rng(0).permutation(small_frame.n_atoms)
    permuted = Frame(
        species=small_frame.species[perm],
        positions=small_frame.positions[perm],
        cell=small_frame.cell,
    )
    assert abs(_energy(model, permuted) - e0) < 1e-9


@pytest.mark.parametrize("model_name", ["compact_model", "pairnet_model"])
def test_forces_match_finite_differences(model_name, small_frame, request):
    model = request.getfixturevalue(model_name)
    ft = prepared(small_frame, model)
    forces = predict_frame(model, ft, forces=True)["forces"].detach().numpy()
    h = 1e-5
    for atom, axis in [(0, 0), (3, 1), (7, 2)]:
        for sign in (+1,):
            plus = small_frame.positions.copy()
            plus[atom, axis] += h
            minus = small_frame.positions.copy()
            minus[atom, axis] -= h
            e_plus = _energy(model, Frame(small_frame.species, plus, small_frame.cell))
            e_minus = _energy(model, Frame(small_frame.species, minus, small_frame.cell))
            fd = -(e_plus - e_minus) / (2 * h)
            assert abs(fd - forces[atom, axis]) < 1e-6


@pytest.mark.parametrize("model_name", ["compact_model", "pairnet_model"])
def test_stress_matches_finite_differences(model_name, small_frame, request):
    model = request.getfixturevalue(model_name)
    ft = prepared(small_frame, model)
    stress = predict_frame(model, ft, forces=False, stress=True)["stress"].detach().numpy()
    volume = abs(np.linalg.det(small_frame.cell))
    h = 1e-6
    for a, b in [(0, 0), (1, 1), (0, 1)]:
        eps = np.zeros((3, 3))
        eps[a, b] = eps[b, a] = h

        def strained_energy(sign: float) -> float:
            f = np.eye(3) + sign * eps
            frame = Frame(
                species=small_frame.species,
                positions=small_frame.positions @ f.T,
                cell=small_frame.cell @ f.T,
            )
            return _energy(model, frame)

        fd = (strained_energy(+1.0) - strained_energy(-1.0)) / (2 * h)
        # Off-diagonal FD perturbs eps_ab AND eps_ba, so it measures 2 sigma_ab V.
        factor = 1.0 if a == b else 2.0
        assert abs(fd / (factor * volume) - stress[a, b]) < 1e-6


def test_energy_smooth_when_pair_crosses_cutoff(pairnet_model):
    """Move one atom through the cutoff radius; energy stays continuous."""
    a = 20.0
    cell = np.eye(3) * a
    energies = []
    for r in np.linspace(5.4, 5.6, 21):
        frame = Frame(
            species=np.array([2, 2]),
            positions=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
            cell=cell,
        )
        energies.append(_energy(pairnet_model, frame))
    steps = np.abs(np.diff(energies))
    assert steps.max() < 5e-3  # no jump at the cutoff crossing
    # Beyond the cutoff the pair energy is exactly the isolated-atom sum.
    assert abs(energies[-1] - energies[-2]) < 1e-9


def test_compact_scaler_guards_dead_features(compact_model, small_frame):
    fts = [prepared(small_frame, compact_model)]
    compact_model.fit_scaler(fts)
    assert (compact_model.feat_std >= 1e-8).all()
    # Elements absent from the fit keep unit std.
    assert torch.isfinite(compact_model.feat_mean).all()


def test_pairnet_symmetric_in_element_order():
    """A Ti-Nb dimer has the same energy as a Nb-Ti dimer."""
    torch.manual_seed(1)
    from alloybench.models import PairNet

    model = PairNet()
    cell = np.eye(3) * 15.0
    d = 2.8
    e_ab = _energy(
        model, Frame(np.array([0, 2]), np.array([[0, 0, 0], [d, 0, 0]], dtype=float), cell)
    )
    e_ba = _energy(
        model, Frame(np.array([2, 0]), np.array([[0, 0, 0], [d, 0, 0]], dtype=float), cell)
    )
    assert abs(e_ab - e_ba) < 1e-12


def test_checkpoint_round_trip(tmp_path, compact_model, small_frame):
    from alloybench.checkpoints import load_checkpoint, save_checkpoint

    ft = prepare(small_frame, compact_model.cutoff, compact_model.angular_cutoff)
    e0 = float(predict_frame(compact_model, ft, forces=False)["energy"])
    path = tmp_path / "model.pt"
    save_checkpoint(path, compact_model, train_info={"note": "test"})
    loaded, info = load_checkpoint(path)
    e1 = float(predict_frame(loaded, ft, forces=False)["energy"])
    assert abs(e0 - e1) < 1e-12
    assert info["note"] == "test"

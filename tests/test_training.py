"""Training loop sanity on a tiny synthetic distillation task."""

from __future__ import annotations

import numpy as np
import torch

from alloybench.frames import Frame
from alloybench.lattice import bcc_supercell
from alloybench.models import (
    PairNet,
    fit_element_refs,
    predict_frame,
    prepare,
    set_element_refs,
)
from alloybench.train import build_model, prepare_frames, train_potential


def _synthetic_labeled_frames(n: int = 10) -> list[Frame]:
    """Tiny frames labeled by a fixed random PairNet acting as teacher."""
    torch.manual_seed(0)
    teacher = PairNet()
    for p in teacher.parameters():
        p.data = p.data + 0.3 * torch.randn_like(p)
    rng = np.random.default_rng(1)
    frames = []
    for k in range(n):
        fr = bcc_supercell("TiNb", reps=2, seed=k)
        fr.positions = fr.positions + rng.normal(0.0, 0.08, fr.positions.shape)
        fr.group = f"TiNb_rattle08_b{k % 2}"
        ft = prepare(fr, teacher.cutoff, teacher.cutoff)
        out = predict_frame(teacher, ft, forces=True)
        fr.energy = float(out["energy"])
        fr.forces = out["forces"].detach().numpy()
        frames.append(fr)
    return frames


def test_training_reduces_error():
    frames = _synthetic_labeled_frames(10)
    model = build_model("pairnet")
    set_element_refs(model, fit_element_refs(frames))
    fts = prepare_frames(frames, model.cutoff, model.cutoff)
    hist = train_potential(
        model, fts[:8], fts[8:], epochs=12, lr=3e-3, force_weight=0.5, batch_size=4, seed=0
    )
    assert hist["train_e_mae"][-1] < hist["train_e_mae"][0]
    assert hist["train_f_mae"][-1] < hist["train_f_mae"][0]
    assert hist["seconds"] > 0


def test_element_refs_recover_linear_energies():
    rng = np.random.default_rng(2)
    true_refs = np.array([-5.0, -6.5, -8.0])
    frames = []
    for k in range(6):
        fr = bcc_supercell(["Ti", "Nb", "TiZrNb"][k % 3], reps=2, seed=k)
        counts = np.array([(fr.species == z).sum() for z in range(3)])
        fr.energy = float(counts @ true_refs + rng.normal(0, 1e-9))
        frames.append(fr)
    refs = fit_element_refs(frames)
    np.testing.assert_allclose(refs, true_refs, atol=1e-6)


def test_training_is_seed_deterministic():
    frames = _synthetic_labeled_frames(6)
    results = []
    for _ in range(2):
        model = build_model("pairnet")
        set_element_refs(model, fit_element_refs(frames))
        fts = prepare_frames(frames, model.cutoff, model.cutoff)
        train_potential(
            model, fts[:5], fts[5:], epochs=3, lr=3e-3, force_weight=0.5, batch_size=4, seed=7
        )
        ft = fts[0]
        results.append(float(predict_frame(model, ft, forces=False)["energy"]))
    assert results[0] == results[1]

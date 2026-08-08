"""Committed-artifact regression: the shipped checkpoints must keep
reproducing their recorded accuracy on the committed sample frames."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Measured once at commit time on data/sample_frames.extxyz (252 frames,
# includes the transfer compositions); the artifacts are deterministic, so
# any drift here means checkpoint and code have diverged.
EXPECTED = {
    "compact": {"params": 6243, "e_mae": 5.305581528502322, "f_mae": 35.529129839425586},
    "pairnet": {"params": 433, "e_mae": 24.887075366524087, "f_mae": 85.56910337932571},
}


@pytest.fixture(scope="module")
def sample_frames():
    from alloybench.frames import read_frames

    path = REPO / "data" / "sample_frames.extxyz"
    if not path.exists():
        pytest.skip("committed sample not present")
    return read_frames(path)


@pytest.mark.parametrize("kind", ["compact", "pairnet"])
def test_committed_checkpoint_reproduces_recorded_metrics(kind, sample_frames):
    from alloybench.checkpoints import load_checkpoint
    from alloybench.evaluate import evaluate_model
    from alloybench.train import prepare_frames

    path = REPO / "models" / f"{kind}.pt"
    if not path.exists():
        pytest.skip("committed checkpoint not present")
    model, info = load_checkpoint(path)
    assert model.n_parameters() == EXPECTED[kind]["params"]
    assert info.get("n_parameters") == EXPECTED[kind]["params"]
    fts = prepare_frames(
        sample_frames, model.cutoff, getattr(model, "angular_cutoff", model.cutoff)
    )
    m = evaluate_model(model, fts)
    assert m["e_mae_mev_atom"] == pytest.approx(EXPECTED[kind]["e_mae"], rel=1e-9)
    assert m["f_mae_mev_a"] == pytest.approx(EXPECTED[kind]["f_mae"], rel=1e-9)


def test_metrics_json_matches_committed_pareto():
    metrics_path = REPO / "results" / "metrics.json"
    if not metrics_path.exists():
        pytest.skip("metrics.json not present")
    pareto = json.loads(metrics_path.read_text())["pareto"]
    assert pareto["compact"]["n_parameters"] == EXPECTED["compact"]["params"]
    assert pareto["pairnet"]["n_parameters"] == EXPECTED["pairnet"]["params"]
    assert pareto["chgnet"]["f_rmse_vs_teacher_mev_a"] == 0.0

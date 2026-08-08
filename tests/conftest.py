"""Shared fixtures: small structures and seeded untrained models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from alloybench.frames import Frame
from alloybench.lattice import bcc_supercell
from alloybench.models import CompactPotential, PairNet


@pytest.fixture()
def small_frame() -> Frame:
    """16-atom rattled TiZrNb cell."""
    rng = np.random.default_rng(3)
    fr = bcc_supercell("TiZrNb", reps=2, seed=11)
    fr.positions = fr.positions + rng.normal(0.0, 0.08, fr.positions.shape)
    return fr


@pytest.fixture()
def compact_model() -> CompactPotential:
    torch.manual_seed(5)
    model = CompactPotential()
    for p in model.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    return model


@pytest.fixture()
def pairnet_model() -> PairNet:
    torch.manual_seed(7)
    model = PairNet()
    for p in model.parameters():
        p.data = p.data + 0.05 * torch.randn_like(p)
    return model

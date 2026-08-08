"""Leakage-safety of the split protocols."""

from __future__ import annotations

import numpy as np

from alloybench.frames import Frame
from alloybench.splits import batch_kind, group_split, random_frame_split


def _toy_frames() -> list[Frame]:
    frames = []
    rng = np.random.default_rng(0)
    comps = ["Ti", "Nb", "TiZrNb", "Ti2ZrNb"]
    kinds = ["rattle05_b0", "rattle18_b1", "strain_vol_b0", "md600_b0", "md1400_b1"]
    for comp in comps:
        for kind in kinds:
            for _ in range(6):
                frames.append(
                    Frame(
                        species=np.zeros(2, dtype=int),
                        positions=rng.normal(size=(2, 3)),
                        cell=np.eye(3) * 5,
                        composition=comp,
                        group=f"{comp}_{kind}",
                    )
                )
    return frames


def test_batch_kind_strips_composition_and_batch():
    assert batch_kind("TiZrNb_rattle18_b2", "TiZrNb") == "rattle18"
    assert batch_kind("Nb_strain_vol_b0", "Nb") == "strain_vol"
    assert batch_kind("Ti_md1400_b1", "Ti") == "md1400"


def test_group_split_keeps_groups_whole():
    frames = _toy_frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    part_of = {}
    for part in ("train", "val", "test"):
        for k in getattr(split, part):
            g = frames[k].group
            assert part_of.setdefault(g, part) == part  # never in two parts


def test_group_split_holds_out_whole_compositions():
    frames = _toy_frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    for k in split.transfer:
        assert frames[k].composition == "Ti2ZrNb"
    for part in ("train", "val", "test"):
        for k in getattr(split, part):
            assert frames[k].composition != "Ti2ZrNb"


def test_group_split_test_covers_every_batch_kind():
    frames = _toy_frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    test_kinds = {batch_kind(frames[k].group, frames[k].composition) for k in split.test}
    all_kinds = {
        batch_kind(fr.group, fr.composition) for fr in frames if fr.composition != "Ti2ZrNb"
    }
    assert test_kinds == all_kinds  # stratification: no kind missing from test


def test_random_frame_split_mixes_groups():
    frames = _toy_frames()
    split = random_frame_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    train_groups = {frames[k].group for k in split.train}
    test_groups = {frames[k].group for k in split.test}
    assert train_groups & test_groups  # the leak the control quantifies


def test_split_partitions_are_disjoint_and_complete():
    frames = _toy_frames()
    for splitter in (group_split, random_frame_split):
        split = splitter(frames, holdout_compositions=("Ti2ZrNb",), seed=1)
        all_idx = sorted(split.train + split.val + split.test + split.transfer)
        assert all_idx == list(range(len(frames)))

"""Leakage-safe dataset splits by generation group and composition.

Frames from one rattle, strain, or MD batch are near-duplicates; letting
siblings from one batch land in both train and test overstates accuracy.
The honest split moves whole generation groups, holds out whole
compositions for transfer testing, and stratifies held-out groups by batch
kind so test difficulty matches the population. The random-frame split
exists only to measure that optimism gap, once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from alloybench.frames import Frame


@dataclass
class Split:
    """Index split of a frame list.

    Attributes:
        train: Training indices.
        val: Validation indices (hyperparameter selection only).
        test: In-distribution test indices.
        transfer: Held-out-composition indices.
    """

    train: list[int]
    val: list[int]
    test: list[int]
    transfer: list[int] = field(default_factory=list)


def batch_kind(group: str, composition: str) -> str:
    """Batch kind of a group id: '<comp>_<kind>_b<n>' -> '<kind>'."""
    kind = group
    if composition and kind.startswith(composition + "_"):
        kind = kind[len(composition) + 1 :]
    return re.sub(r"_b\d+$", "", kind)


def group_split(
    frames: list[Frame],
    holdout_compositions: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """Split whole generation groups, stratified by batch kind."""
    rng = np.random.default_rng(seed)
    transfer = [k for k, fr in enumerate(frames) if fr.composition in holdout_compositions]
    in_dist = [k for k, fr in enumerate(frames) if fr.composition not in holdout_compositions]
    kind_of = {frames[k].group: batch_kind(frames[k].group, frames[k].composition) for k in in_dist}
    by_kind: dict[str, list[str]] = {}
    for g in sorted(kind_of):
        by_kind.setdefault(kind_of[g], []).append(g)
    test_groups: set[str] = set()
    val_groups: set[str] = set()
    for kind in sorted(by_kind):
        groups = by_kind[kind]
        rng.shuffle(groups)
        n_test = max(1, int(round(test_fraction * len(groups))))
        n_val = max(1, int(round(val_fraction * len(groups))))
        test_groups.update(groups[:n_test])
        val_groups.update(groups[n_test : n_test + n_val])
    split = Split(train=[], val=[], test=[], transfer=transfer)
    for k in in_dist:
        g = frames[k].group
        if g in test_groups:
            split.test.append(k)
        elif g in val_groups:
            split.val.append(k)
        else:
            split.train.append(k)
    return split


def random_frame_split(
    frames: list[Frame],
    holdout_compositions: tuple[str, ...] = (),
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> Split:
    """The optimistic protocol: random frames regardless of group.

    Measured once per benchmark to quantify how much it flatters accuracy;
    never used for reported model quality.
    """
    rng = np.random.default_rng(seed)
    transfer = [k for k, fr in enumerate(frames) if fr.composition in holdout_compositions]
    in_dist = np.array(
        [k for k, fr in enumerate(frames) if fr.composition not in holdout_compositions]
    )
    perm = rng.permutation(len(in_dist))
    n_test = int(round(test_fraction * len(in_dist)))
    n_val = int(round(val_fraction * len(in_dist)))
    return Split(
        train=in_dist[perm[n_test + n_val :]].tolist(),
        val=in_dist[perm[n_test : n_test + n_val]].tolist(),
        test=in_dist[perm[:n_test]].tolist(),
        transfer=transfer,
    )

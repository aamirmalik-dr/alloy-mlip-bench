"""Periodic neighbor list for general triclinic cells, written from scratch.

The number of periodic images needed along lattice direction k follows from
the spacing of the lattice planes normal to that direction: the dual
reciprocal vector is column k of inv(cell) (rows of cell are lattice
vectors), and the plane spacing is 1 / |column k|. Positions are wrapped
into the cell before the search and the integer shifts are corrected back
afterwards, so structures whose atoms have drifted outside the box (as MD
produces) are handled exactly. Verified against ase.neighborlist and a
brute-force reference in the test suite, including sheared cells.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeighborTable:
    """Directed neighbor pairs of one periodic structure.

    For pair p, the vector from atom i[p] to the neighbor image of atom j[p]
    is positions[j[p]] + shifts[p] @ cell - positions[i[p]].

    Attributes:
        i: (P,) center-atom indices, sorted.
        j: (P,) neighbor-atom indices.
        shifts: (P, 3) integer lattice shifts of the neighbor image.
    """

    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray

    def __len__(self) -> int:
        return len(self.i)


def image_counts(cell: np.ndarray, cutoff: float) -> tuple[int, int, int]:
    """Periodic images required per lattice direction for a cutoff search."""
    inv = np.linalg.inv(np.asarray(cell, dtype=float))
    spacing = 1.0 / np.linalg.norm(inv, axis=0)
    return tuple(int(n) for n in np.ceil(cutoff / spacing))


def neighbor_table(positions: np.ndarray, cell: np.ndarray, cutoff: float) -> NeighborTable:
    """All directed pairs (i, j, shift) with |r_ij| < cutoff.

    Args:
        positions: (N, 3) Cartesian coordinates; may lie outside the cell.
        cell: (3, 3) lattice matrix, rows are lattice vectors.
        cutoff: Cutoff radius in Angstrom.

    Returns:
        NeighborTable whose shifts are valid for the ORIGINAL positions.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n = len(positions)
    frac = positions @ np.linalg.inv(cell)
    wrap = np.floor(frac).astype(int)
    wrapped = (frac - wrap) @ cell

    n1, n2, n3 = image_counts(cell, cutoff)
    g1, g2, g3 = np.meshgrid(
        np.arange(-n1, n1 + 1), np.arange(-n2, n2 + 1), np.arange(-n3, n3 + 1), indexing="ij"
    )
    all_shifts = np.stack([g1, g2, g3], axis=-1).reshape(-1, 3)  # (S, 3)
    # delta[a, b, s] = wrapped[b] + shift[s] @ cell - wrapped[a]
    delta = wrapped[None, :, None, :] + (all_shifts @ cell)[None, None, :, :]
    delta = delta - wrapped[:, None, None, :]
    within = (delta**2).sum(axis=-1) < cutoff**2
    zero = int(np.flatnonzero((all_shifts == 0).all(axis=1))[0])
    within[np.arange(n), np.arange(n), zero] = False
    ai, aj, ashift = np.nonzero(within)
    shifts = all_shifts[ashift] + wrap[ai] - wrap[aj]
    order = np.lexsort((aj, ai))
    return NeighborTable(i=ai[order], j=aj[order], shifts=shifts[order])


def pair_vectors(table: NeighborTable, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """(P, 3) displacement vectors from center atoms to neighbor images."""
    return positions[table.j] + table.shifts @ cell - positions[table.i]


def angular_triplets(table: NeighborTable, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unordered pairs of kept table entries that share a center atom.

    Args:
        table: Neighbor table sorted by center index i.
        keep: Indices into the table (e.g. pairs within a shorter cutoff).

    Returns:
        (t1, t2): index arrays into the FULL table; each unordered neighbor
        pair around a center appears exactly once.
    """
    centers = table.i[keep]
    order = np.argsort(centers, kind="stable")
    keep = np.asarray(keep)[order]
    centers = centers[order]
    t1_parts, t2_parts = [], []
    boundaries = np.flatnonzero(np.diff(centers)) + 1
    for block in np.split(keep, boundaries):
        m = len(block)
        if m < 2:
            continue
        a, b = np.triu_indices(m, k=1)
        t1_parts.append(block[a])
        t2_parts.append(block[b])
    if not t1_parts:
        empty = np.zeros(0, dtype=int)
        return empty, empty.copy()
    return np.concatenate(t1_parts), np.concatenate(t2_parts)

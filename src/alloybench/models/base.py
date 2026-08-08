"""Shared building blocks for the trained potentials.

Both trained models consume FrameTensors, a precomputed torch view of one
periodic structure and its neighbor table. Energies are decomposed as a
least-squares per-element reference plus a learned residual, and forces and
virial stress come from exact autograd: forces as -dE/dpositions, stress as
(1/V) dE/dstrain through the uniform-strain parameter trick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from alloybench.frames import Frame
from alloybench.neighbors import angular_triplets, neighbor_table, pair_vectors

DTYPE = torch.float64


@dataclass
class FrameTensors:
    """Torch tensors of one structure plus its (static) neighbor indexing."""

    species: torch.Tensor  # (N,) long
    positions: torch.Tensor  # (N, 3)
    cell: torch.Tensor  # (3, 3)
    idx_i: torch.Tensor  # (P,) long
    idx_j: torch.Tensor  # (P,) long
    shifts: torch.Tensor  # (P, 3)
    trip_a: torch.Tensor  # (T,) long, indices into the pair axis
    trip_b: torch.Tensor  # (T,) long
    energy: torch.Tensor | None = None
    forces: torch.Tensor | None = None

    @property
    def n_atoms(self) -> int:
        return int(self.species.shape[0])


def prepare(frame: Frame, cutoff: float, angular_cutoff: float) -> FrameTensors:
    """Build FrameTensors for one frame (neighbor search on the CPU in numpy)."""
    table = neighbor_table(frame.positions, frame.cell, cutoff)
    dists = np.linalg.norm(pair_vectors(table, frame.positions, frame.cell), axis=1)
    keep = np.flatnonzero(dists < angular_cutoff)
    trip_a, trip_b = angular_triplets(table, keep)
    return FrameTensors(
        species=torch.as_tensor(frame.species, dtype=torch.long),
        positions=torch.as_tensor(frame.positions, dtype=DTYPE),
        cell=torch.as_tensor(frame.cell, dtype=DTYPE),
        idx_i=torch.as_tensor(table.i, dtype=torch.long),
        idx_j=torch.as_tensor(table.j, dtype=torch.long),
        shifts=torch.as_tensor(table.shifts, dtype=DTYPE),
        trip_a=torch.as_tensor(trip_a, dtype=torch.long),
        trip_b=torch.as_tensor(trip_b, dtype=torch.long),
        energy=None if frame.energy is None else torch.tensor(frame.energy, dtype=DTYPE),
        forces=None if frame.forces is None else torch.as_tensor(frame.forces, dtype=DTYPE),
    )


def strained_pair_vectors(
    ft: FrameTensors, positions: torch.Tensor, strain: torch.Tensor | None
) -> torch.Tensor:
    """(P, 3) pair vectors, optionally deformed by a uniform strain.

    A uniform strain maps every Cartesian vector v to v @ (I + eps); taking
    dE/deps at eps = 0 yields V times the virial stress.
    """
    vec = positions[ft.idx_j] + ft.shifts @ ft.cell - positions[ft.idx_i]
    if strain is not None:
        sym = 0.5 * (strain + strain.T)
        vec = vec @ (torch.eye(3, dtype=vec.dtype) + sym)
    return vec


def cosine_cutoff(r: torch.Tensor, rc: float) -> torch.Tensor:
    """Smooth cutoff 0.5 (cos(pi r / rc) + 1), zero at and beyond rc."""
    return torch.where(r < rc, 0.5 * (torch.cos(torch.pi * r / rc) + 1.0), torch.zeros_like(r))


def fit_element_refs(frames: list[Frame]) -> np.ndarray:
    """Least-squares per-element reference energies from labeled frames."""
    counts = np.zeros((len(frames), 3))
    energies = np.zeros(len(frames))
    for k, fr in enumerate(frames):
        for s in range(3):
            counts[k, s] = np.sum(fr.species == s)
        energies[k] = fr.energy
    refs, *_ = np.linalg.lstsq(counts, energies, rcond=None)
    return refs


class PotentialOutput(dict):
    """Dict of energy / forces / stress tensors from a model predict call."""


def predict_frame(
    model: torch.nn.Module,
    ft: FrameTensors,
    forces: bool = True,
    stress: bool = False,
) -> PotentialOutput:
    """Energy, and optionally autograd forces and virial stress, of one frame.

    Args:
        model: Module whose forward(ft, positions, strain) returns total energy.
        ft: Prepared frame tensors.
        forces: Compute -dE/dpositions.
        stress: Compute (1/V) dE/dstrain (3, 3), symmetric.

    Returns:
        PotentialOutput with "energy" (float tensor), "forces", "stress".
    """
    positions = ft.positions.detach().clone().requires_grad_(forces)
    strain = torch.zeros((3, 3), dtype=DTYPE, requires_grad=stress) if stress else None
    energy = model(ft, positions, strain)
    out = PotentialOutput(energy=energy)
    grads_of = [t for t, want in ((positions, forces), (strain, stress)) if want]
    if grads_of:
        grads = torch.autograd.grad(energy, grads_of, create_graph=False)
        k = 0
        if forces:
            out["forces"] = -grads[k]
            k += 1
        if stress:
            volume = torch.abs(torch.det(ft.cell))
            g = grads[k]
            out["stress"] = 0.5 * (g + g.T) / volume
    return out

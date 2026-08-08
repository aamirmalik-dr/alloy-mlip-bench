"""The compact distilled potential: reduced symmetry functions + small MLPs.

An element-resolved Behler-Parrinello-style network sized for CPU MD
throughput rather than maximum accuracy: 24 element-resolved radial features
(8 Gaussian centers per neighbor element) plus 6 element-agnostic angular
features, feeding one small SiLU network per center element. SiLU keeps the
energy surface smooth, which matters for NVE stability. Forces and virial
stress are exact autograd derivatives.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from alloybench.models.base import (
    DTYPE,
    FrameTensors,
    cosine_cutoff,
    strained_pair_vectors,
)

# (lambda, zeta) exponents of the angular terms.
ANGULAR_TERMS = ((1.0, 1.0), (1.0, 2.0), (1.0, 4.0), (-1.0, 1.0), (-1.0, 2.0), (-1.0, 4.0))


class CompactPotential(nn.Module):
    """Element-resolved descriptor potential with per-element readout MLPs.

    Args:
        cutoff: Radial cutoff in Angstrom.
        angular_cutoff: Shorter cutoff for the angular terms.
        n_radial: Gaussian centers per neighbor element.
        hidden: Hidden width of the per-element MLPs.
        angular_eta: Radial damping of the angular terms (1/Angstrom^2).
    """

    def __init__(
        self,
        cutoff: float = 5.5,
        angular_cutoff: float = 4.5,
        n_radial: int = 8,
        hidden: int = 32,
        angular_eta: float = 0.12,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.angular_cutoff = float(angular_cutoff)
        self.n_radial = int(n_radial)
        self.n_features = 3 * self.n_radial + len(ANGULAR_TERMS)
        centers = torch.linspace(1.9, cutoff - 0.2, n_radial, dtype=DTYPE)
        width = float(centers[1] - centers[0])
        self.register_buffer("centers", centers)
        self.register_buffer("eta", torch.tensor(1.0 / (2.0 * width**2), dtype=DTYPE))
        self.angular_eta = float(angular_eta)
        self.register_buffer("feat_mean", torch.zeros(3, self.n_features, dtype=DTYPE))
        self.register_buffer("feat_std", torch.ones(3, self.n_features, dtype=DTYPE))
        self.register_buffer("element_refs", torch.zeros(3, dtype=DTYPE))
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.n_features, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(3)
            ]
        )
        self.to(DTYPE)

    def descriptors(self, ft: FrameTensors, vec: torch.Tensor) -> torch.Tensor:
        """(N, n_features) symmetry-function features from pair vectors."""
        n = ft.n_atoms
        r = torch.linalg.norm(vec, dim=1)
        fc = cosine_cutoff(r, self.cutoff)
        gauss = torch.exp(-self.eta * (r[:, None] - self.centers[None, :]) ** 2) * fc[:, None]
        # Scatter each pair into the radial slot of its neighbor's element.
        slot = ft.species[ft.idx_j]
        radial = vec.new_zeros(n, 3, self.n_radial)
        radial.index_put_((ft.idx_i, slot), gauss, accumulate=True)
        feats = [radial.reshape(n, 3 * self.n_radial)]

        if len(ft.trip_a):
            v1, v2 = vec[ft.trip_a], vec[ft.trip_b]
            r1, r2 = r[ft.trip_a], r[ft.trip_b]
            cos_t = (v1 * v2).sum(dim=1) / (r1 * r2)
            damp = torch.exp(-self.angular_eta * (r1**2 + r2**2))
            damp = damp * cosine_cutoff(r1, self.angular_cutoff)
            damp = damp * cosine_cutoff(r2, self.angular_cutoff)
            ang_cols = []
            for lam, zeta in ANGULAR_TERMS:
                ang_cols.append(((0.5 * (1.0 + lam * cos_t)) ** zeta) * damp)
            ang = torch.stack(ang_cols, dim=1)
            angular = vec.new_zeros(n, len(ANGULAR_TERMS))
            angular.index_add_(0, ft.idx_i[ft.trip_a], ang)
        else:
            angular = vec.new_zeros(n, len(ANGULAR_TERMS))
        feats.append(angular)
        return torch.cat(feats, dim=1)

    def forward(
        self, ft: FrameTensors, positions: torch.Tensor, strain: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Total energy in eV."""
        vec = strained_pair_vectors(ft, positions, strain)
        feats = self.descriptors(ft, vec)
        scaled = (feats - self.feat_mean[ft.species]) / self.feat_std[ft.species]
        energy = self.element_refs[ft.species].sum()
        for z in range(3):
            mask = ft.species == z
            if mask.any():
                energy = energy + self.nets[z](scaled[mask]).sum()
        return energy

    @torch.no_grad()
    def fit_scaler(self, fts: list[FrameTensors]) -> None:
        """Standardize features per center element over the training frames.

        Features that are constant on the training set get unit standard
        deviation so unseen geometries cannot blow up the scaled inputs.
        """
        per_element: list[list[torch.Tensor]] = [[], [], []]
        for ft in fts:
            vec = strained_pair_vectors(ft, ft.positions, None)
            feats = self.descriptors(ft, vec)
            for z in range(3):
                mask = ft.species == z
                if mask.any():
                    per_element[z].append(feats[mask])
        for z in range(3):
            if not per_element[z]:
                continue
            stacked = torch.cat(per_element[z], dim=0)
            self.feat_mean[z] = stacked.mean(dim=0)
            std = stacked.std(dim=0)
            self.feat_std[z] = torch.where(std > 1e-8, std, torch.ones_like(std))

    def config(self) -> dict:
        return {
            "kind": "compact",
            "cutoff": self.cutoff,
            "angular_cutoff": self.angular_cutoff,
            "n_radial": self.n_radial,
            "hidden": int(self.nets[0][0].out_features),
            "angular_eta": self.angular_eta,
        }

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def compact_from_config(cfg: dict) -> CompactPotential:
    """Rebuild a CompactPotential from its config() dict."""
    return CompactPotential(
        cutoff=cfg["cutoff"],
        angular_cutoff=cfg["angular_cutoff"],
        n_radial=cfg["n_radial"],
        hidden=cfg["hidden"],
        angular_eta=cfg["angular_eta"],
    )


def set_element_refs(model: nn.Module, refs: np.ndarray) -> None:
    """Install least-squares element reference energies into a model buffer."""
    model.element_refs.copy_(torch.as_tensor(refs, dtype=DTYPE))

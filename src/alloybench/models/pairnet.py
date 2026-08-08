"""PairNet: a smooth learned pair potential, the fast-but-limited floor.

Each directed pair contributes half of a symmetric pair energy predicted by
one small MLP from a Gaussian radial expansion concatenated with a one-hot
of the unordered element pair. It is trained on the same data and budget as
the compact model and exists to anchor the fast end of the accuracy-versus-
speed Pareto front. Pair potentials cannot represent angular physics, which
is exactly the limitation the benchmark makes visible.
"""

from __future__ import annotations

import torch
from torch import nn

from alloybench.models.base import DTYPE, FrameTensors, cosine_cutoff, strained_pair_vectors

# Unordered element-pair slots for (Ti, Zr, Nb).
PAIR_SLOTS = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5}


class PairNet(nn.Module):
    """Learned pair potential with a smooth cutoff.

    Args:
        cutoff: Pair cutoff in Angstrom.
        n_basis: Gaussian radial basis size.
        hidden: Hidden width of the shared MLP.
    """

    def __init__(self, cutoff: float = 5.5, n_basis: int = 10, hidden: int = 24) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.n_basis = int(n_basis)
        centers = torch.linspace(1.6, cutoff - 0.2, n_basis, dtype=DTYPE)
        width = float(centers[1] - centers[0])
        self.register_buffer("centers", centers)
        self.register_buffer("eta", torch.tensor(1.0 / (2.0 * width**2), dtype=DTYPE))
        self.register_buffer("element_refs", torch.zeros(3, dtype=DTYPE))
        slot_table = torch.zeros(3, 3, dtype=torch.long)
        for (a, b), s in PAIR_SLOTS.items():
            slot_table[a, b] = s
            slot_table[b, a] = s
        self.register_buffer("slot_table", slot_table)
        self.net = nn.Sequential(
            nn.Linear(n_basis + len(PAIR_SLOTS), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.to(DTYPE)

    def forward(
        self, ft: FrameTensors, positions: torch.Tensor, strain: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Total energy in eV (half-sum over directed pairs)."""
        vec = strained_pair_vectors(ft, positions, strain)
        r = torch.linalg.norm(vec, dim=1)
        fc = cosine_cutoff(r, self.cutoff)
        rbf = torch.exp(-self.eta * (r[:, None] - self.centers[None, :]) ** 2)
        slots = self.slot_table[ft.species[ft.idx_i], ft.species[ft.idx_j]]
        onehot = torch.zeros(len(r), len(PAIR_SLOTS), dtype=DTYPE)
        onehot[torch.arange(len(r)), slots] = 1.0
        pair_e = self.net(torch.cat([rbf, onehot], dim=1)).squeeze(-1) * fc
        return self.element_refs[ft.species].sum() + 0.5 * pair_e.sum()

    def config(self) -> dict:
        return {
            "kind": "pairnet",
            "cutoff": self.cutoff,
            "n_basis": self.n_basis,
            "hidden": int(self.net[0].out_features),
        }

    def n_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def pairnet_from_config(cfg: dict) -> PairNet:
    """Rebuild a PairNet from its config() dict."""
    return PairNet(cutoff=cfg["cutoff"], n_basis=cfg["n_basis"], hidden=cfg["hidden"])

"""Labeled periodic structures and extxyz persistence.

A Frame is one periodic supercell with optional teacher labels. Labels are
SURROGATE MODEL LABELS from a pretrained universal potential, never DFT.
Persistence uses the extxyz format through ASE (structure handling only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ELEMENTS = ("Ti", "Zr", "Nb")
ELEMENT_Z = {"Ti": 22, "Zr": 40, "Nb": 41}
MASSES_AMU = {"Ti": 47.867, "Zr": 91.224, "Nb": 92.906}


@dataclass
class Frame:
    """One periodic structure with optional labels.

    Attributes:
        species: (N,) integer element indices into ELEMENTS.
        positions: (N, 3) Cartesian coordinates in Angstrom.
        cell: (3, 3) lattice matrix, rows are lattice vectors, Angstrom.
        composition: Composition tag, e.g. "TiZrNb".
        group: Generation-group id (the unit of the leakage-safe split).
        energy: Teacher total energy in eV, or None.
        forces: (N, 3) teacher forces in eV/Angstrom, or None.
    """

    species: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    composition: str = ""
    group: str = ""
    energy: float | None = None
    forces: np.ndarray | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    def symbols(self) -> list[str]:
        return [ELEMENTS[int(s)] for s in self.species]


def to_atoms(frame: Frame):
    """Convert a Frame to an ASE Atoms object (no labels attached)."""
    from ase import Atoms

    return Atoms(symbols=frame.symbols(), positions=frame.positions, cell=frame.cell, pbc=True)


def from_atoms(atoms, composition: str = "", group: str = "") -> Frame:
    """Build an unlabeled Frame from an ASE Atoms object."""
    sym_to_idx = {el: k for k, el in enumerate(ELEMENTS)}
    species = np.array([sym_to_idx[s] for s in atoms.get_chemical_symbols()], dtype=int)
    return Frame(
        species=species,
        positions=np.asarray(atoms.get_positions(), dtype=float),
        cell=np.asarray(atoms.get_cell()[:], dtype=float),
        composition=composition,
        group=group,
    )


def write_frames(path: str | Path, frames: list[Frame]) -> None:
    """Write labeled frames to an extxyz file."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    images = []
    for fr in frames:
        atoms = Atoms(symbols=fr.symbols(), positions=fr.positions, cell=fr.cell, pbc=True)
        atoms.info["composition"] = fr.composition
        atoms.info["group"] = fr.group
        if fr.energy is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=fr.energy, forces=fr.forces)
        images.append(atoms)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write(str(path), images, format="extxyz")


def read_frames(path: str | Path) -> list[Frame]:
    """Read labeled frames from an extxyz file."""
    from ase.io import read

    frames = []
    for atoms in read(str(path), index=":", format="extxyz"):
        fr = from_atoms(
            atoms,
            composition=str(atoms.info.get("composition", "")),
            group=str(atoms.info.get("group", "")),
        )
        if atoms.calc is not None and "energy" in atoms.calc.results:
            fr.energy = float(atoms.calc.results["energy"])
            if "forces" in atoms.calc.results:
                fr.forces = np.asarray(atoms.calc.results["forces"], dtype=float)
        frames.append(fr)
    return frames

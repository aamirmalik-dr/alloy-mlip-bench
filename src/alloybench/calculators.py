"""Every potential behind one ASE Calculator interface.

This is the deployment layer the whole benchmark runs through: the repo's
own trained potentials (compact, pairnet) and the pip-installed pretrained
foundation potentials (CHGNet, MACE-MP-0 small) are all constructed by
get_calculator() and consumed only through the standard ASE energy /
forces / stress properties. Foundation packages are dependencies, never
vendored; their weights carry their own licenses, recorded in the info
dict. The trained models run in float64; CHGNet runs in its native float32
and MACE in float64 as its authors recommend for dynamics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from alloybench.checkpoints import load_checkpoint
from alloybench.frames import from_atoms
from alloybench.models import predict_frame, prepare

MODEL_NAMES = ("pairnet", "compact", "chgnet", "mace-mp0-small")
# eV/A^3 -> GPa
EV_A3_TO_GPA = 160.21766208


class TorchPotentialCalculator(Calculator):
    """ASE calculator around a trained alloybench potential.

    Energy, forces, and virial stress are exact autograd derivatives of one
    torch energy model; the neighbor table is rebuilt on every call, so any
    cell and any wrapped or unwrapped positions are valid.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self, model, label: str = "alloybench", **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.label = label

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "TorchPotentialCalculator":
        model, _ = load_checkpoint(path)
        return cls(model, label=Path(path).stem)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        frame = from_atoms(atoms)
        cutoff = self.model.cutoff
        angular = getattr(self.model, "angular_cutoff", cutoff)
        ft = prepare(frame, cutoff, angular)
        want_stress = "stress" in properties
        out = predict_frame(self.model, ft, forces=True, stress=want_stress)
        energy = float(out["energy"].detach())
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = out["forces"].detach().numpy()
        if want_stress:
            s = out["stress"].detach().numpy()
            self.results["stress"] = np.array(
                [s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]]
            )


def get_calculator(name: str, checkpoint: str | Path | None = None) -> tuple[Calculator, dict]:
    """Construct a benchmark calculator by name.

    Args:
        name: One of MODEL_NAMES.
        checkpoint: Checkpoint path override for the trained models
            (defaults to the committed models/<name>.pt).

    Returns:
        (ase_calculator, info) where info records package, version,
        checkpoint identity, license, and parameter count.
    """
    if name in ("compact", "pairnet"):
        path = Path(checkpoint) if checkpoint else Path("models") / f"{name}.pt"
        model, train_info = load_checkpoint(path)
        info = {
            "package": "alloybench (this repository)",
            "version": "0.1.0",
            "checkpoint": str(path),
            "license": "MIT",
            "n_parameters": model.n_parameters(),
            "trained_on": "surrogate teacher labels (never DFT)",
            "train_info_keys": sorted(train_info),
        }
        return TorchPotentialCalculator(model, label=name), info
    if name == "chgnet":
        import chgnet
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        model = CHGNet.load()
        calc = CHGNetCalculator(model=model, use_device="cpu")
        info = {
            "package": "chgnet",
            "version": chgnet.__version__,
            "checkpoint": f"CHGNet v{getattr(model, 'version', '0.3.0')} default pretrained "
            "weights (MPtrj-trained)",
            "license": "BSD-3-Clause (code and weights)",
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }
        return calc, info
    if name == "mace-mp0-small":
        import mace
        from mace.calculators import mace_mp

        calc = mace_mp(model="small", device="cpu", default_dtype="float64")
        n_params = int(sum(p.numel() for m in calc.models for p in m.parameters()))
        info = {
            "package": "mace-torch",
            "version": mace.__version__,
            "checkpoint": "MACE-MP-0 small",
            "license": "MIT (code and MACE-MP-0 weights)",
            "n_parameters": n_params,
        }
        return calc, info
    raise ValueError(f"unknown model {name!r}; expected one of {MODEL_NAMES}")

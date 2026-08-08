"""alloybench: MLIPs as ASE calculators, benchmarked on TiZrNb properties.

The package trains a compact distilled potential on surrogate teacher
labels, wraps it and the pip-installed pretrained foundation potentials
behind one ASE Calculator interface, and runs a property benchmark suite
(EOS, elastic constants, RDF, thermal expansion, vacancy formation, NVE
stability, CPU speed) on BCC TiZrNb. All property values are model
predictions; anchors are Materials Project DFT or cited experiments.
"""

from alloybench.calculators import MODEL_NAMES, TorchPotentialCalculator, get_calculator
from alloybench.checkpoints import load_checkpoint, save_checkpoint
from alloybench.frames import ELEMENTS, Frame, read_frames, write_frames

__all__ = [
    "MODEL_NAMES",
    "TorchPotentialCalculator",
    "get_calculator",
    "load_checkpoint",
    "save_checkpoint",
    "ELEMENTS",
    "Frame",
    "read_frames",
    "write_frames",
]

__version__ = "0.1.0"

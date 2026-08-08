# alloybench API

Everything below assumes the repository root as working directory and the
package installed with `pip install -e .`.

## Calculators: any potential, one interface

```python
from alloybench import get_calculator

calc, info = get_calculator("compact")            # committed checkpoint
calc, info = get_calculator("compact", checkpoint="path/to/other.pt")
calc, info = get_calculator("chgnet")             # needs the foundation extra
calc, info = get_calculator("mace-mp0-small")
print(info)  # package, version, checkpoint, license, parameter count
```

The returned object is a standard ASE calculator, so it drops into any ASE
workflow:

```python
from ase.md.verlet import VelocityVerlet
from ase import units
from alloybench.frames import to_atoms
from alloybench.lattice import bcc_supercell

atoms = to_atoms(bcc_supercell("TiZrNb", reps=3, seed=0))
atoms.calc = calc
print(atoms.get_potential_energy(), atoms.get_stress())
VelocityVerlet(atoms, timestep=2 * units.fs).run(100)
```

`TorchPotentialCalculator` implements energy, free_energy, forces, and
virial stress; forces are exact autograd derivatives, stress comes from the
uniform-strain trick, and both are verified against finite differences in
the test suite.

## Property benchmarks

Each benchmark takes an ASE calculator and returns a plain dict; the suite
runner ties them together and makes results restartable.

```python
from alloybench.properties import (
    eos_scan, elastic_constants, rdf_benchmark, npt_lattice_parameter,
    vacancy_formation, nve_drift, speed_benchmark,
)

eos = eos_scan(calc, "Nb", reps=3, seed=20260807)
print(eos["a0_angstrom"], eos["b0_gpa"])

el = elastic_constants(
    calc, "TiZrNb", a0=eos["a0_angstrom"], b0_gpa=eos["b0_gpa"],
    seed=20260807, relax_internal=True,
)
print(el["c11_gpa"], el["c12_gpa"], el["c44_gpa"])
```

```python
from alloybench.suite import run_model_suite
import yaml

config = yaml.safe_load(open("configs/suite.yaml"))
results = run_model_suite("compact", config, "results/suite/compact")
```

## Training a potential

```python
from alloybench.frames import read_frames
from alloybench.splits import group_split
from alloybench.train import fit_and_select
from alloybench.checkpoints import save_checkpoint

frames = read_frames("data/full/dataset.extxyz")
split = group_split(frames, holdout_compositions=("Ti2ZrNb", "TiZrNb2"), seed=0)
model, info = fit_and_select(
    "compact",
    [frames[k] for k in split.train],
    [frames[k] for k in split.val],
    grid=[{"lr": 3e-3, "force_weight": 1.0}],
    screen_epochs=5,
    final_epochs=40,
)
save_checkpoint("models/my_compact.pt", model, train_info=info)
```

## Command line

```
alloybench generate            # build + teacher-label the dataset
alloybench train               # grid screen + final training, both models
alloybench splitgap            # group vs random-frame split control
alloybench suite --models pairnet compact chgnet mace-mp0-small
alloybench suite --models compact --checkpoint path/to/external.pt
alloybench demo                # offline demo on committed artifacts
```

Every command reads its parameters from `configs/*.yaml`; seeds are fixed
there, so reruns reproduce the committed numbers.

"""Generate and execute notebooks/tutorial.ipynb.

The tutorial runs a miniature version of the whole pipeline live (small
dataset, short training) so every number in it is actually computed at
execution time, then closes with the committed full-benchmark Pareto
replotted from results/metrics.json, clearly labeled as such.

Usage:
    python scripts/build_notebook.py [--no-execute]
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text.strip())


CELLS = [
    md("""\
# From teacher labels to a deployed potential: a walkthrough

This notebook runs a miniature version of the alloy-mlip-bench pipeline end
to end: generate a small teacher-labeled dataset, train a compact potential,
deploy it as an ASE calculator, and benchmark it against the teacher on a
real property. Every number below is computed live when the notebook
executes; the committed full-run results appear only in the last section and
are labeled as such.

One thing to keep straight throughout: the labels are **surrogate model
labels** from a pretrained universal potential (CHGNet), not DFT. Real DFT
enters only as Materials Project validation anchors."""),
    code("""\
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alloybench.anchors import load_anchors
from alloybench.calculators import TorchPotentialCalculator, get_calculator
from alloybench.evaluate import evaluate_model
from alloybench.labeling import label_frames
from alloybench.lattice import rattle_batch, strain_batch
from alloybench.models import fit_element_refs, set_element_refs
from alloybench.properties import eos_scan, nve_drift, speed_benchmark
from alloybench.splits import group_split
from alloybench.train import build_model, prepare_frames, train_potential

# chgnet's internal energy call trips a torch tensor-to-scalar UserWarning;
# it is upstream noise, not something this pipeline controls.
warnings.filterwarnings("ignore", message="Converting a tensor with requires_grad")

COLOR = {"compact": "#2a78d6", "chgnet": "#eb6834", "mace-mp0-small": "#1baf7a",
         "pairnet": "#eda100"}
ANCHOR = "#52514e"
rng_seed = 20260808
print("imports ready")"""),
    md("""\
## Step 1: generate and label a miniature dataset

Three compositions, rattle and strain batches on 16-atom BCC cells, labeled
by the teacher. Each batch is one generation group, which matters for the
split in step 2: frames inside a batch are near-duplicates of each other."""),
    code("""\
teacher, teacher_info = get_calculator("chgnet")
print({k: teacher_info[k] for k in ("package", "version", "checkpoint", "license")})

frames = []
for ci, comp in enumerate(["Ti", "Nb", "TiZrNb"]):
    for b, amp in enumerate([0.06, 0.06, 0.14, 0.14]):
        frames.extend(rattle_batch(comp, reps=2, amplitude=amp, n_frames=8,
                                   seed=rng_seed + 100 * ci + b, batch=b))
    for b in (0, 1):
        frames.extend(strain_batch(comp, reps=2, n_frames=8,
                                   seed=rng_seed + 100 * ci + 50 + b, batch=b))
sec = label_frames(frames, teacher)
print(f"{len(frames)} frames labeled at {sec:.2f} s/frame (surrogate teacher labels, not DFT)")"""),
    code("""\
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for comp, color in zip(["Ti", "Nb", "TiZrNb"], ["#2a78d6", "#eb6834", "#1baf7a"]):
    e = [fr.energy / fr.n_atoms for fr in frames if fr.composition == comp]
    axes[0].hist(e, bins=18, alpha=0.65, color=color, label=comp)
    f = np.concatenate([np.linalg.norm(fr.forces, axis=1)
                        for fr in frames if fr.composition == comp])
    axes[1].hist(f, bins=24, alpha=0.65, color=color, label=comp)
axes[0].set_xlabel("teacher energy (eV/atom)"); axes[0].set_ylabel("frames")
axes[1].set_xlabel("teacher |force| (eV/A)"); axes[1].set_ylabel("atoms")
axes[0].legend(frameon=False); axes[0].set_title("Label distributions", loc="left")
plt.tight_layout(); plt.show()"""),
    md("""\
## Step 2: split by generation group and train the compact potential

The split moves whole batches into train or test, never random frames from
the same batch. The compact model is the repo's distilled potential:
element-resolved radial symmetry functions plus a small angular block,
one small network per element, forces by autograd."""),
    code("""\
split = group_split(frames, seed=0)
print(f"{len(split.train)} train / {len(split.val)} val / {len(split.test)} test frames")

model = build_model("compact", seed=0)
set_element_refs(model, fit_element_refs([frames[k] for k in split.train]))
train_fts = prepare_frames([frames[k] for k in split.train], model.cutoff, model.angular_cutoff)
val_fts = prepare_frames([frames[k] for k in split.val], model.cutoff, model.angular_cutoff)
model.fit_scaler(train_fts)
t0 = time.perf_counter()
hist = train_potential(model, train_fts, val_fts, epochs=30, lr=3e-3,
                       force_weight=1.0, batch_size=8, seed=0)
n_params = model.n_parameters()
print(f"trained {n_params} parameters in {time.perf_counter() - t0:.0f} s")"""),
    code("""\
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(hist["epoch"], np.array(hist["train_e_mae"]) * 1000, color="#2a78d6", lw=2,
        label="train energy MAE (meV/atom)")
ax.plot(hist["epoch"], np.array(hist["train_f_mae"]) * 1000, color="#eda100", lw=2,
        label="train force MAE (meV/A)")
ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("error (log)")
ax.set_title("Training curves (miniature run)", loc="left")
ax.legend(frameon=False); plt.tight_layout(); plt.show()"""),
    md("""\
## Step 3: evaluate on held-out groups and deploy as an ASE calculator

The parity plot uses only frames from batches the model never saw. The same
trained object then becomes a standard ASE calculator, which is the whole
deployment story of this repository: after this line, any ASE workflow
(optimizers, MD engines, property scripts) runs on the compact potential."""),
    code("""\
test_fts = prepare_frames([frames[k] for k in split.test], model.cutoff, model.angular_cutoff)
m = evaluate_model(model, test_fts, collect_parity=True)
print(f"held-out miniature test: {m['e_mae_mev_atom']:.1f} meV/atom, "
      f"{m['f_mae_mev_a']:.0f} meV/A force MAE")

p = m["parity"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
et, ep = np.array(p["e_true_ev_atom"]), np.array(p["e_pred_ev_atom"])
axes[0].plot([et.min(), et.max()], [et.min(), et.max()], color="#898781", lw=1)
axes[0].scatter(et, ep, s=14, color="#2a78d6", alpha=0.6)
axes[0].set_xlabel("teacher (eV/atom)"); axes[0].set_ylabel("compact (eV/atom)")
axes[0].set_title(f"Energy, MAE {m['e_mae_mev_atom']:.1f} meV/atom", loc="left")
ft, fp = np.array(p["f_true_ev_a"]), np.array(p["f_pred_ev_a"])
axes[1].plot([ft.min(), ft.max()], [ft.min(), ft.max()], color="#898781", lw=1)
axes[1].scatter(ft[::3], fp[::3], s=5, color="#2a78d6", alpha=0.35)
axes[1].set_xlabel("teacher force (eV/A)"); axes[1].set_ylabel("compact force (eV/A)")
axes[1].set_title(f"Forces, RMSE {m['f_rmse_mev_a']:.0f} meV/A", loc="left")
plt.tight_layout(); plt.show()

calc = TorchPotentialCalculator(model, label="compact-mini")
print("deployed:", type(calc).__name__, "with properties", calc.implemented_properties)"""),
    md("""\
## Step 4: a real property benchmark through the ASE interface

Equation of state of elemental Nb: the miniature compact potential against
its teacher, with the Materials Project DFT anchor marked. The anchor is the
only non-model number in the plot."""),
    code("""\
eos_c = eos_scan(calc, "Nb", reps=2, seed=rng_seed)
eos_t = eos_scan(teacher, "Nb", reps=2, seed=rng_seed)
anchors = load_anchors("data")
a_mp = anchors["anchors"]["Nb"]["a0_angstrom"]

fig, ax = plt.subplots(figsize=(6.8, 4.0))
for eos, name in ((eos_c, "compact"), (eos_t, "chgnet")):
    n = eos["n_atoms"]
    a = (2.0 * np.array(eos["volumes_a3"]) / n) ** (1 / 3)
    e = (np.array(eos["energies_ev"]) - eos["e0_ev"]) / n * 1000
    label = f"{name}: a0 {eos['a0_angstrom']:.3f} A, B0 {eos['b0_gpa']:.0f} GPa"
    ax.plot(a, e, color=COLOR[name], lw=2, marker="o", ms=4, label=label)
ax.axvline(a_mp, color=ANCHOR, ls="--", lw=1.4)
ax.text(a_mp, ax.get_ylim()[1] * 0.9, " MP DFT a0", color=ANCHOR, fontsize=9)
ax.set_xlabel("lattice constant (A)"); ax.set_ylabel("E - E0 (meV/atom)")
ax.set_title("Nb equation of state, miniature model vs teacher", loc="left")
ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.show()"""),
    md("""\
## Step 5: the deployment gates, and the final metric

Two things a potential must earn before anyone runs MD with it: energy
conservation under NVE (the stability gate) and a measured cost per step.
Together with the force error from step 3 these place the miniature model
on the same accuracy-versus-speed axes the full benchmark uses."""),
    code("""\
nve = nve_drift(calc, "TiZrNb", a0=eos_c["a0_angstrom"], reps=2, seed=rng_seed,
                steps=400, record_every=4)
spd = speed_benchmark(calc, "TiZrNb", a0=eos_c["a0_angstrom"], reps=2, seed=rng_seed,
                      warmup_steps=5, timed_steps=50)
gate = "pass" if nve["gate_passed"] else "FAIL"
fig, ax = plt.subplots(figsize=(6.8, 3.4))
de = (np.array(nve["e_tot_ev_atom"]) - nve["e_tot_ev_atom"][0]) * 1000
ax.plot(nve["times_ps"], de, color="#2a78d6", lw=1.6)
ax.set_xlabel("time (ps)"); ax.set_ylabel("total-energy change (meV/atom)")
ax.set_title(f"NVE gate: drift {nve['drift_mev_atom_ps']:.3g} meV/atom/ps ({gate})",
             loc="left")
plt.tight_layout(); plt.show()

print("miniature-model final metrics (computed in this notebook):")
print(f"  force RMSE vs teacher : {m['f_rmse_mev_a']:.0f} meV/A")
print(f"  Nb a0 vs MP DFT       : {eos_c['a0_angstrom']:.3f} vs {a_mp} A")
print(f"  NVE drift             : {nve['drift_mev_atom_ps']:.3g} meV/atom/ps ({gate})")
print(f"  MD speed              : {spd['ms_per_atom_step']:.3f} ms/atom/step")"""),
    md("""\
## The committed full benchmark

Everything above was a miniature run inside one notebook; the committed
benchmark trains on the full dataset (about 1,550 frames, 70 generation
groups, 9 compositions) and runs the whole property suite for four models.
The cell below replots the committed accuracy-versus-speed Pareto from
`results/metrics.json`; these numbers come from the full run, not from this
notebook. The teacher's zero force error is by construction, and MACE's
force number is model-to-model disagreement, not error against truth."""),
    code("""\
metrics = json.loads(Path("results/metrics.json").read_text())
pareto = metrics["pareto"]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for name, p in pareto.items():
    if p["ms_per_atom_step"] is None or p["f_rmse_vs_teacher_mev_a"] is None:
        continue
    ax.scatter(p["ms_per_atom_step"], p["f_rmse_vs_teacher_mev_a"], s=110,
               color=COLOR[name], edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(name, (p["ms_per_atom_step"], p["f_rmse_vs_teacher_mev_a"]),
                textcoords="offset points", xytext=(9, 4), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("CPU cost (ms per atom per MD step, log)")
ax.set_ylabel("force RMSE vs teacher labels (meV/A)")
ax.set_title("Committed full-benchmark Pareto (from results/metrics.json)", loc="left")
plt.tight_layout(); plt.show()
print(json.dumps(pareto, indent=2))"""),
    md("""\
That is the tradeoff this repository is about: the compact model gives up a
measured amount of fidelity to its teacher and, in exchange, steps MD an
order of magnitude faster on the same CPU. Whether that trade is worth it
depends on the property being computed, which is exactly what the full suite
in `results/` and `RESULTS.md` quantifies, property by property, against
Materials Project DFT anchors and cited experiments."""),
]


def main() -> int:
    nb = nbf.v4.new_notebook()
    nb.cells = CELLS
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    out = REPO / "notebooks" / "tutorial.ipynb"
    out.parent.mkdir(exist_ok=True)
    if "--no-execute" not in sys.argv:
        from nbclient import NotebookClient

        client = NotebookClient(
            nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(REPO)}}
        )
        client.execute()
    nbf.write(nb, out)
    n_err = sum(
        1
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    )
    print(f"wrote {out} ({n_err} error cells)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())

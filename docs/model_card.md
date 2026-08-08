# Model card: the trained potentials

Two potentials are trained and committed in this repository. Both are
original code, trained in one session on surrogate teacher labels; neither
has seen DFT data directly.

## compact (models/compact.pt)

- Architecture: element-resolved Behler-Parrinello-style network. 24
  element-resolved radial Gaussian features (8 centers per neighbor element,
  cutoff 5.5 A) plus 6 element-agnostic angular features (lambda in {+1, -1},
  zeta in {1, 2, 4}, angular cutoff 4.5 A), standardized per center element,
  one 30-32-32-1 SiLU MLP per element on top of least-squares element
  reference energies. 6,243 parameters, float64.
- Forces and stress: exact autograd; forces as -dE/dpositions, virial stress
  by the uniform-strain trick. Both verified against finite differences.
- Training: 882 group-split frames, Adam, combined per-atom-energy + force
  MSE loss. Grid {lr 1e-3, 3e-3} x {force weight 0.2, 1.0} screened at 25
  epochs on the fixed validation score (E MAE meV/atom + 0.3 F MAE meV/A);
  winner (lr 3e-3, w 1.0) retrained 80 epochs, batch 8, seed 0. 1,396 s on
  a Ryzen 5 5600U CPU.
- Held-out accuracy vs teacher labels (group split): 5.33 meV/atom E MAE,
  33.6 meV/A F MAE, 53.0 meV/A F RMSE. Transfer to unseen compositions:
  8.61 meV/atom, 56.3 meV/A MAE with a heavy tail (256 meV/A RMSE).
- Deployment: 0.264 ms/atom/MD step (54 atoms, CPU); NVE drift 0.0003
  meV/atom/ps over 3 ps (gate passed).
- Known limitations, measured: inherits its teacher's biases (Nb vacancy
  formation 1.58 eV vs 2.6 to 3.1 expt; Ti B0 low), Zr bulk modulus badly
  stiff (170 vs teacher 97 GPa), Nb C44 poor (22 GPa, anchor 43), rare
  large force errors on unseen compositions. Intended range: near-BCC
  Ti-Zr-Nb configurations up to roughly 1400 K; nothing else is claimed.

## pairnet (models/pairnet.pt)

- Architecture: learned pair potential. 10 Gaussian radial basis functions
  (cutoff 5.5 A) concatenated with a one-hot of the unordered element pair,
  one shared 16-24-1 SiLU MLP, half-sum over directed pairs, smooth cosine
  cutoff, element reference energies. 433 parameters, float64.
- Training: identical data, grid, budget, and selection protocol as compact;
  chose lr 1e-3, force weight 1.0. 539 s.
- Held-out accuracy vs teacher labels: 18.44 meV/atom, 84.8 meV/A MAE,
  129.1 meV/A RMSE.
- Role and honest verdict: it exists to anchor the fast end of the Pareto
  front, and the benchmark shows a pair potential's failure is structural,
  not statistical: equilibrium lattice constants 6 to 11% too large, bulk
  moduli 5x too soft, vacancy formation 0.17 eV, thermal expansion 5x too
  high, and C12 = C44 to half a percent on Nb (the Cauchy relation, obligatory
  for central pair forces, strongly violated by real BCC metals). Do not use
  it for anything except as a baseline.

## Provenance and licensing

- Labels: chgnet 0.4.2 (CHGNet v0.3.0 default pretrained weights,
  MPtrj-trained, BSD-3-Clause), selected by measured CPU speed
  (data/teacher_choice.json). Labels are surrogate model outputs, not DFT.
- Dataset: generated in-repo (data/provenance.json); 1,566 frames, 72
  generation groups, seed 20260807.
- Checkpoint format: torch.save dict of config, state_dict, and training
  provenance; loads with weights_only=True. The committed-artifact tests
  regression-guard both checkpoints against results/metrics.json.
- License: MIT, same as the repository.

# Data provenance

## Training labels are surrogate model labels, not DFT

Every energy and force label in this repository comes from a pretrained
universal potential (the "teacher") selected by measured CPU speed in
`data/teacher_choice.json` and recorded with exact package, version,
checkpoint, and license. The teacher was itself trained on DFT data, but its
outputs are model predictions. Nothing in this repository describes them as
DFT, and no result here is a DFT calculation.

- `data/teacher_choice.json`: the timed teacher selection (chgnet vs
  MACE-MP-0 small) with full provenance.
- `data/provenance.json`: dataset generation record (counts, groups,
  compositions, timings, config) written by `alloybench generate`.
- `data/sample_frames.extxyz`: a committed sample of teacher-labeled frames
  (a few hundred, drawn across compositions and batch kinds) so the demo and
  the committed-artifact tests run with no generation step. It is a subset of
  the generated dataset, not a download; the full dataset regenerates with
  `alloybench generate` (roughly 1,500 frames, few minutes of CPU labeling).
- `data/full/` (gitignored): the full generated dataset lives here locally.

## Real DFT enters only as validation anchors

- `data/anchors_mp.json`: fetched from the Materials Project API
  (`scripts/fetch_mp_anchors.py`, needs a free API key in `MP_API_KEY`).
  Materials Project data is CC BY 4.0: A. Jain et al., APL Materials 1,
  011002 (2013), materialsproject.org. BCC Ti (mp-73), Zr (mp-41),
  Nb (mp-75): lattice constant, bulk modulus, and elastic constants.
- `data/anchors_literature.json`: committed fallback used when no key is
  configured, with citations, plus cited experimental values (Nb elastic
  constants, thermal expansion, vacancy formation) used where MP has no
  entry. Experimental numbers are room-temperature measurements; the
  benchmarks compare 0 K or classical-MD model predictions against them
  with that caveat stated.

## What is never committed

Full generated datasets, foundation-model weights (installed or downloaded
by their own packages under their own licenses: chgnet BSD-3-Clause,
mace-torch MIT), and anything above the repository size policy.

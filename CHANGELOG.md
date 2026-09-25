# Changelog

## 0.1.1 (2026-09-25)

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`; citation file added.

## 0.1.0 (2026-08-09)

- First public release: `alloybench` package with the ASE calculator layer for a compact distilled potential and for the pretrained foundation potentials, plus the property suite (equation of state, elastic constants, radial distribution, NPT, vacancy formation, NVE energy-conservation gate, speed).
- Teacher choice recorded, Materials Project DFT anchors with a cited literature fallback, labeled sample, and provenance files.
- Trained checkpoints, the full set of benchmark JSON outputs, `results/metrics.json`, and figures including the accuracy-versus-speed Pareto plot.
- Results-first README, benchmark report, model card, and an executed tutorial notebook.
- Fix: notebook lint cleanup with scoped ruff ignores; parity arrays rounded.
- Chore: ruff 0.16.2 and black 26.5.1 pinned, black target set to Python 3.11.
- Test: cutoff smoothness covered for both trained models.
- Fix: duplicated step 0 in the example, notebook warning noise, EOS metadata parity across calculators.
- Docs: every benchmark figure embedded in `RESULTS.md`; vacancy standard deviation corrected; test count corrected.

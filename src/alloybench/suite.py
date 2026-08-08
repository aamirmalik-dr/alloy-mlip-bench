"""Property-suite orchestration: one model, the whole benchmark battery.

Each (model, property) result lands in its own JSON as soon as it is
measured, and existing files are skipped, so an interrupted suite resumes
where it stopped. Every model is benchmarked on identical structures
(same occupancy seeds) but around its OWN equilibrium geometry.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from alloybench.calculators import get_calculator
from alloybench.frames import ELEMENTS
from alloybench.properties import (
    elastic_constants,
    eos_scan,
    expansion_coefficient,
    npt_lattice_parameter,
    nve_drift,
    rdf_benchmark,
    speed_benchmark,
    vacancy_formation,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def run_model_suite(
    name: str,
    config: dict,
    out_dir: str | Path,
    checkpoint: str | Path | None = None,
    log=print,
) -> dict:
    """Run every property benchmark for one model, resumably.

    Args:
        name: Calculator name from alloybench.calculators.MODEL_NAMES.
        config: Parsed configs/suite.yaml.
        out_dir: Directory for per-property JSONs (results/suite/<name>).
        checkpoint: Optional checkpoint override for trained models.
        log: Progress logger.

    Returns:
        Dict of every property payload (loaded from the JSONs).
    """
    out = Path(out_dir)
    seed = int(config["seed"])
    reps = int(config.get("reps", 3))
    calc, info = get_calculator(name, checkpoint=checkpoint)
    _write(out / "info.json", {"model": name, **info})

    def step(prop: str, fn):
        path = out / f"{prop}.json"
        if path.exists():
            log(f"[{name}] {prop}: cached")
            return json.loads(path.read_text())
        t0 = time.perf_counter()
        try:
            payload = fn()
        except Exception as exc:  # record the failure, keep the suite going
            payload = {"failed": f"{type(exc).__name__}: {exc}"}
            log(f"[{name}] {prop}: FAILED ({payload['failed']})")
        payload["wall_seconds"] = time.perf_counter() - t0
        _write(path, payload)
        if "failed" not in payload:
            log(f"[{name}] {prop}: done in {payload['wall_seconds']:.1f} s")
        return payload

    cfg_eos = config["eos"]
    eos = step(
        "eos",
        lambda: {
            "by_composition": {
                comp: eos_scan(
                    calc,
                    comp,
                    reps=reps,
                    seed=seed,
                    span=float(cfg_eos.get("span", 0.08)),
                    n_points=int(cfg_eos.get("n_points", 11)),
                    relax_internal=bool(cfg_eos.get("relax_alloy", True) and comp not in ELEMENTS),
                )
                for comp in cfg_eos["compositions"]
            }
        },
    )
    if "failed" in eos or "by_composition" not in eos:
        log(f"[{name}] eos failed; skipping the dependent benchmarks")
        return {"info": info, "eos": eos}
    by_comp = eos["by_composition"]
    reservoirs = {el: by_comp[el]["e0_per_atom_ev"] for el in ELEMENTS if el in by_comp}

    results = {"info": info, "eos": eos}
    cfg = config.get("elastic")
    if cfg:
        results["elastic"] = step(
            "elastic",
            lambda: {
                "targets": [
                    elastic_constants(
                        calc,
                        t["composition"],
                        a0=by_comp[t["composition"]]["a0_angstrom"],
                        b0_gpa=by_comp[t["composition"]]["b0_gpa"],
                        reps=reps,
                        seed=seed,
                        deltas=tuple(cfg.get("deltas", (0.006, 0.012))),
                        relax_internal=bool(t.get("relax_internal", False)),
                    )
                    for t in cfg["targets"]
                ]
            },
        )
    cfg = config.get("vacancy")
    if cfg:
        results["vacancy"] = step(
            "vacancy",
            lambda: {
                "targets": [
                    vacancy_formation(
                        calc,
                        t["composition"],
                        a0=by_comp[t["composition"]]["a0_angstrom"],
                        reservoir_e_atom=reservoirs,
                        reps=reps,
                        seed=seed,
                        n_sites=int(t.get("n_sites", 1)),
                    )
                    for t in cfg["targets"]
                ]
            },
        )
    cfg = config.get("nve")
    if cfg:
        results["nve"] = step(
            "nve",
            lambda: nve_drift(
                calc,
                cfg["composition"],
                a0=by_comp[cfg["composition"]]["a0_angstrom"],
                temperature_k=float(cfg.get("temperature_k", 300)),
                reps=reps,
                seed=seed,
                steps=int(cfg.get("steps", 1500)),
                gate_mev_atom_ps=float(cfg.get("gate_mev_atom_ps", 1.0)),
            ),
        )
    cfg = config.get("speed")
    if cfg:
        results["speed"] = step(
            "speed",
            lambda: speed_benchmark(
                calc,
                cfg["composition"],
                a0=by_comp[cfg["composition"]]["a0_angstrom"],
                reps=reps,
                seed=seed,
                timed_steps=int(cfg.get("timed_steps", 100)),
            ),
        )
    cfg = config.get("rdf")
    if cfg:
        results["rdf"] = step(
            "rdf",
            lambda: {
                "by_temperature": {
                    str(int(t)): rdf_benchmark(
                        calc,
                        cfg["composition"],
                        temperature_k=float(t),
                        a0=by_comp[cfg["composition"]]["a0_angstrom"],
                        reps=reps,
                        seed=seed,
                        equil_steps=int(cfg.get("equil_steps", 300)),
                        prod_steps=int(cfg.get("prod_steps", 1000)),
                    )
                    for t in cfg["temperatures_k"]
                }
            },
        )
    cfg = config.get("thermal")
    if cfg:

        def run_thermal() -> dict:
            comp = cfg["composition"]
            runs = [
                npt_lattice_parameter(
                    calc,
                    comp,
                    temperature_k=float(t),
                    a0=by_comp[comp]["a0_angstrom"],
                    b0_gpa=by_comp[comp]["b0_gpa"],
                    reps=reps,
                    seed=seed,
                    equil_steps=int(cfg.get("equil_steps", 300)),
                    prod_steps=int(cfg.get("prod_steps", 800)),
                )
                for t in cfg["temperatures_k"]
            ]
            return {
                "runs": runs,
                "alpha_per_k": expansion_coefficient(
                    [r["temperature_k"] for r in runs],
                    [r["a_mean_angstrom"] for r in runs],
                ),
            }

        results["thermal"] = step("thermal", run_thermal)
    return results

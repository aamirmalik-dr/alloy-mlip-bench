"""Aggregate every benchmark JSON into results/metrics.json.

The metrics file is the single source of truth for every number quoted in
the README, RESULTS.md, and the model card, including the Pareto quantities:
MD speed, force disagreement versus teacher labels, and the mean absolute
relative deviation from the Materials Project DFT anchors.

Usage:
    python scripts/make_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

MODELS = ["pairnet", "compact", "chgnet", "mace-mp0-small"]


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def anchor_deviation(suite: dict, anchors: dict) -> dict:
    """Mean absolute relative deviation (%) against MP DFT anchors.

    Covers the nine MP-anchored numbers: a0 and B0 for BCC Ti, Zr, Nb, and
    C11, C12, C44 for Nb. Experimental anchors (Nb vacancy, elastic
    fallback) are shown in the tables but kept out of this score so it stays
    purely DFT-anchored.
    """
    rows = []
    by_comp = suite["eos"]["by_composition"]
    for el in ("Ti", "Zr", "Nb"):
        a = anchors["anchors"][el]
        rows.append(("a0_" + el, by_comp[el]["a0_angstrom"], a["a0_angstrom"]))
        if a.get("b0_gpa"):
            rows.append(("b0_" + el, by_comp[el]["b0_gpa"], a["b0_gpa"]))
    nb_anchor = anchors["anchors"]["Nb"]
    nb_elastic = next(
        (t for t in suite.get("elastic", {}).get("targets", []) if t["composition"] == "Nb"),
        None,
    )
    if nb_elastic and "c11_gpa" in nb_anchor:
        for key in ("c11_gpa", "c12_gpa", "c44_gpa"):
            rows.append((key + "_Nb", nb_elastic[key], nb_anchor[key]))
    devs = {name: 100.0 * abs(model - ref) / abs(ref) for name, model, ref in rows}
    return {
        "per_quantity_percent": {k: round(v, 2) for k, v in devs.items()},
        "mean_abs_percent": round(float(np.mean(list(devs.values()))), 2),
        "n_quantities": len(devs),
    }


def main() -> int:
    from alloybench.anchors import load_anchors

    anchors = load_anchors(REPO / "data")
    metrics: dict = {
        "provenance": load(REPO / "data" / "provenance.json"),
        "teacher_choice": load(REPO / "data" / "teacher_choice.json"),
        "anchors_source": anchors.get("source"),
        "training": load(REPO / "results" / "training.json"),
        "accuracy": load(REPO / "results" / "accuracy.json"),
        "foundation_accuracy": load(REPO / "results" / "foundation_accuracy.json"),
        "split_gap": load(REPO / "results" / "split_gap.json"),
    }
    # Strip bulky parity arrays from the aggregated accuracy block.
    if metrics["accuracy"]:
        for kind in list(metrics["accuracy"]):
            for part in list(metrics["accuracy"][kind]):
                metrics["accuracy"][kind][part] = {
                    k: v for k, v in metrics["accuracy"][kind][part].items() if k != "parity"
                }
    if metrics["training"]:
        for kind in ("compact", "pairnet"):
            if kind in metrics["training"]:
                metrics["training"][kind].pop("history", None)

    suite_summary: dict = {}
    pareto: dict = {}
    for name in MODELS:
        d = REPO / "results" / "suite" / name
        suite = {
            prop: load(d / f"{prop}.json")
            for prop in ("eos", "elastic", "vacancy", "nve", "speed", "rdf", "thermal")
        }
        if suite["eos"] is None:
            continue
        info = load(d / "info.json") or {}
        by_comp = suite["eos"]["by_composition"]
        summary = {
            "n_parameters": info.get("n_parameters"),
            "eos": {
                comp: {
                    "a0_angstrom": round(v["a0_angstrom"], 4),
                    "b0_gpa": round(v["b0_gpa"], 1),
                }
                for comp, v in by_comp.items()
            },
        }
        if suite["elastic"]:
            summary["elastic"] = {
                t["composition"]: {
                    k: round(t[k], 1) for k in ("c11_gpa", "c12_gpa", "c44_gpa", "c_prime_gpa")
                }
                for t in suite["elastic"]["targets"]
            }
        if suite["vacancy"]:
            summary["vacancy_ef_ev"] = {
                t["composition"]: {
                    "mean": round(t["ef_mean_ev"], 3),
                    "std": round(t["ef_std_ev"], 3),
                    "by_element": {k: round(v, 3) for k, v in t["ef_by_element_ev"].items()},
                }
                for t in suite["vacancy"]["targets"]
            }
        if suite["nve"]:
            summary["nve"] = {
                "drift_mev_atom_ps": round(suite["nve"]["drift_mev_atom_ps"], 4),
                "gate_mev_atom_ps": suite["nve"]["gate_mev_atom_ps"],
                "gate_passed": suite["nve"]["gate_passed"],
            }
        if suite["speed"]:
            summary["speed"] = {
                "ms_per_atom_step": round(suite["speed"]["ms_per_atom_step"], 4),
                "ms_per_step": round(suite["speed"]["ms_per_step"], 2),
                "n_atoms": suite["speed"]["n_atoms"],
            }
        if suite["rdf"]:
            summary["rdf_first_peak"] = {
                temp: {
                    "position_angstrom": round(v["first_peak"]["position_angstrom"], 3),
                    "height": round(v["first_peak"]["height"], 2),
                }
                for temp, v in suite["rdf"]["by_temperature"].items()
            }
        if suite["thermal"]:
            summary["thermal"] = {
                "alpha_per_k": float(f"{suite['thermal']['alpha_per_k']:.3e}"),
                "a_by_temp": {
                    str(int(r["temperature_k"])): round(r["a_mean_angstrom"], 4)
                    for r in suite["thermal"]["runs"]
                },
            }
        summary["anchor_deviation"] = anchor_deviation(
            {k: v for k, v in suite.items() if v is not None}, anchors
        )
        suite_summary[name] = summary

        f_vs_teacher = None
        if name in ("compact", "pairnet") and metrics["accuracy"]:
            f_vs_teacher = metrics["accuracy"][name]["test"]["f_rmse_mev_a"]
        elif name == "chgnet":
            f_vs_teacher = 0.0  # the teacher scores itself, by construction
        elif name == "mace-mp0-small" and metrics["foundation_accuracy"]:
            f_vs_teacher = metrics["foundation_accuracy"]["f_rmse_mev_a"]
        pareto[name] = {
            "ms_per_atom_step": summary.get("speed", {}).get("ms_per_atom_step"),
            "f_rmse_vs_teacher_mev_a": (
                round(f_vs_teacher, 1) if f_vs_teacher is not None else None
            ),
            "anchor_mean_abs_deviation_percent": summary["anchor_deviation"]["mean_abs_percent"],
            "n_parameters": summary["n_parameters"],
        }

    metrics["suite"] = suite_summary
    metrics["pareto"] = pareto
    out = REPO / "results" / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    print(json.dumps(pareto, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

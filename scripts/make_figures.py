"""Render every committed figure from the benchmark JSONs.

Series identity is consistent across all figures: pairnet yellow, compact
blue, chgnet (teacher) orange, MACE-MP-0 small aqua; anchors are neutral
gray reference lines, never a series color. Every series and Pareto point
carries a direct label.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
FIG = REPO / "results" / "figures"

COLOR = {
    "compact": "#2a78d6",
    "chgnet": "#eb6834",
    "mace-mp0-small": "#1baf7a",
    "pairnet": "#eda100",
}
LABEL = {
    "compact": "compact (this repo)",
    "chgnet": "CHGNet (teacher)",
    "mace-mp0-small": "MACE-MP-0 small",
    "pairnet": "pairnet (this repo)",
}
MODELS = ["pairnet", "compact", "chgnet", "mace-mp0-small"]
INK, MUTED, GRID, ANCHOR = "#0b0b0b", "#898781", "#e1e0d9", "#52514e"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def suite(name: str, prop: str) -> dict | None:
    return load(REPO / "results" / "suite" / name / f"{prop}.json")


def fig_pareto(metrics: dict) -> None:
    pareto = metrics["pareto"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, key, ylabel, title in (
        (
            axes[0],
            "f_rmse_vs_teacher_mev_a",
            "force RMSE vs teacher labels (meV/A)",
            "Fidelity to the teacher",
        ),
        (
            axes[1],
            "anchor_mean_abs_deviation_percent",
            "mean |deviation| vs MP DFT anchors (%)",
            "Distance from DFT anchors",
        ),
    ):
        points = {
            name: pareto[name]
            for name in MODELS
            if pareto.get(name)
            and pareto[name]["ms_per_atom_step"] is not None
            and pareto[name][key] is not None
        }
        top_name = max(points, key=lambda n: points[n][key]) if points else None
        for name, p in points.items():
            x, y = p["ms_per_atom_step"], p[key]
            marker = "o" if name in ("compact", "pairnet") else "s"
            ax.scatter(
                x,
                y,
                s=110,
                color=COLOR[name],
                marker=marker,
                edgecolor="white",
                linewidth=1.5,
                zorder=3,
            )
            tag = LABEL[name]
            if p.get("n_parameters"):
                n = p["n_parameters"]
                tag += f"\n{n / 1000:.1f}k params" if n < 1e6 else f"\n{n / 1e6:.2f}M params"
            if name == "chgnet" and key == "f_rmse_vs_teacher_mev_a":
                tag += "\n(0 by construction)"
            offset = (9, -26) if name == top_name else (9, 4)
            ax.annotate(
                tag,
                (x, y),
                textcoords="offset points",
                xytext=offset,
                fontsize=8.5,
                color=INK,
            )
        ax.set_xscale("log")
        ax.set_xlabel("CPU cost (ms per atom per MD step, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(ax.get_xlim()[0] * 0.5, ax.get_xlim()[1] * 4.0)
    fig.suptitle(
        "Accuracy versus speed on BCC TiZrNb (54-atom MD, CPU)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "pareto.png")
    plt.close(fig)


def fig_parity(accuracy: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.2))
    for row, kind in enumerate(("compact", "pairnet")):
        parity = accuracy[kind]["test"]["parity"]
        et = np.array(parity["e_true_ev_atom"])
        ep = np.array(parity["e_pred_ev_atom"])
        ax = axes[row, 0]
        lo, hi = min(et.min(), ep.min()), max(et.max(), ep.max())
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, zorder=1)
        ax.scatter(et, ep, s=12, alpha=0.5, color=COLOR[kind], edgecolor="none", zorder=2)
        mae = accuracy[kind]["test"]["e_mae_mev_atom"]
        ax.set_title(f"{LABEL[kind]}: energy, MAE {mae:.1f} meV/atom", loc="left", color=INK)
        ax.set_xlabel("teacher energy (eV/atom)")
        ax.set_ylabel("predicted energy (eV/atom)")
        ax.set_aspect("equal", adjustable="box")

        ft = np.array(parity["f_true_ev_a"])
        fp = np.array(parity["f_pred_ev_a"])
        step = max(1, len(ft) // 4000)
        ax = axes[row, 1]
        lo, hi = ft.min(), ft.max()
        pad = 0.03 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, zorder=1)
        ax.scatter(
            ft[::step], fp[::step], s=6, alpha=0.35, color=COLOR[kind], edgecolor="none", zorder=2
        )
        rmse = accuracy[kind]["test"]["f_rmse_mev_a"]
        ax.set_title(f"{LABEL[kind]}: forces, RMSE {rmse:.0f} meV/A", loc="left", color=INK)
        ax.set_xlabel("teacher force component (eV/A)")
        ax.set_ylabel("predicted force component (eV/A)")
        ax.set_aspect("equal", adjustable="box")
    fig.suptitle(
        "Held-out test parity against surrogate teacher labels (group split)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "parity.png")
    plt.close(fig)


def fig_eos(anchors: dict) -> None:
    comps = ["Ti", "Zr", "Nb", "TiZrNb"]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6))
    for ax, comp in zip(axes.ravel(), comps):
        for name in MODELS:
            eos = suite(name, "eos")
            if not eos:
                continue
            v = eos["by_composition"][comp]
            n = v["n_atoms"]
            a = (2.0 * np.array(v["volumes_a3"]) / n) ** (1.0 / 3.0)
            e = (np.array(v["energies_ev"]) - v["e0_ev"]) / n * 1000.0
            ax.plot(a, e, color=COLOR[name], lw=2, marker="o", ms=3.5, label=LABEL[name])
        anchor = anchors["anchors"].get(comp)
        if anchor:
            ax.axvline(anchor["a0_angstrom"], color=ANCHOR, lw=1.4, ls="--", zorder=1)
            ax.text(
                anchor["a0_angstrom"],
                ax.get_ylim()[1] * 0.94,
                " MP DFT a0",
                fontsize=8,
                color=ANCHOR,
            )
        ax.set_title(f"BCC {comp}", loc="left", color=INK)
        ax.set_xlabel("conventional lattice constant (A)")
        ax.set_ylabel("E - E0 (meV/atom)")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper center")
    fig.suptitle(
        "Equation of state per model (each around its own minimum)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "eos.png")
    plt.close(fig)


def fig_elastic_vacancy(anchors: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    ax = axes[0]
    keys = ["c11_gpa", "c12_gpa", "c44_gpa"]
    ticks = ["C11", "C12", "C44"]
    width = 0.18
    xs = np.arange(len(keys))
    for m, name in enumerate(MODELS):
        el = suite(name, "elastic")
        if not el:
            continue
        nb = next(t for t in el["targets"] if t["composition"] == "Nb")
        vals = [nb[k] for k in keys]
        bars = ax.bar(
            xs + (m - 1.5) * width, vals, width * 0.92, color=COLOR[name], label=LABEL[name]
        )
        ax.bar_label(bars, fmt="%.0f", fontsize=7.5, color=INK, padding=1)
    nb_anchor = anchors["anchors"]["Nb"]
    if "c11_gpa" in nb_anchor:
        for x, k in zip(xs, keys):
            ax.hlines(nb_anchor[k], x - 0.42, x + 0.42, color=ANCHOR, lw=1.6, ls="--", zorder=4)
    ax.hlines([], [], [], color=ANCHOR, lw=1.6, ls="--", label="MP DFT anchor")
    ax.set_xticks(xs, ticks)
    ax.set_ylabel("elastic constant (GPa)")
    ax.set_title("Elemental Nb elastic constants", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    comps = ["Nb", "TiZrNb"]
    xs = np.arange(len(comps))
    for m, name in enumerate(MODELS):
        vac = suite(name, "vacancy")
        if not vac:
            continue
        means = [
            next(t for t in vac["targets"] if t["composition"] == c)["ef_mean_ev"] for c in comps
        ]
        stds = [
            next(t for t in vac["targets"] if t["composition"] == c)["ef_std_ev"] for c in comps
        ]
        bars = ax.bar(
            xs + (m - 1.5) * width,
            means,
            width * 0.92,
            yerr=stds,
            capsize=2,
            color=COLOR[name],
            error_kw={"ecolor": MUTED, "lw": 1},
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=7.5, color=INK, padding=2)
    exp = anchors.get("experimental", {}).get("Nb_vacancy_formation_ev")
    if exp:
        ax.fill_between(
            [-0.45, 0.45], exp["range"][0], exp["range"][1], color=ANCHOR, alpha=0.15, zorder=1
        )
        ax.text(0.02, exp["range"][1] + 0.03, "expt range (Nb)", fontsize=8, color=ANCHOR)
    ax.set_xticks(xs, comps)
    ax.set_ylabel("vacancy formation energy (eV)")
    ax.set_title("Monovacancy formation (elemental-reservoir convention)", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "elastic_vacancy.png")
    plt.close(fig)


def fig_rdf() -> None:
    temps = None
    for name in MODELS:
        r = suite(name, "rdf")
        if r:
            temps = sorted(r["by_temperature"], key=float)
            break
    if not temps:
        return
    fig, axes = plt.subplots(1, len(temps), figsize=(5.2 * len(temps), 4.2), sharey=True)
    for ax, temp in zip(np.atleast_1d(axes), temps):
        for name in MODELS:
            rdf = suite(name, "rdf")
            if not rdf:
                continue
            v = rdf["by_temperature"][temp]
            ax.plot(v["r_angstrom"], v["g_r"], color=COLOR[name], lw=2, label=LABEL[name])
        ax.set_title(f"TiZrNb at {temp} K", loc="left", color=INK)
        ax.set_xlabel("r (A)")
    np.atleast_1d(axes)[0].set_ylabel("g(r)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        "Radial distribution functions from NVT MD (54 atoms)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "rdf.png")
    plt.close(fig)


def fig_thermal() -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name in MODELS:
        th = suite(name, "thermal")
        if not th:
            continue
        t = [r["temperature_k"] for r in th["runs"]]
        a = [r["a_mean_angstrom"] for r in th["runs"]]
        s = [r["a_std_angstrom"] for r in th["runs"]]
        ax.errorbar(
            t,
            a,
            yerr=s,
            color=COLOR[name],
            lw=2,
            marker="o",
            ms=4,
            capsize=3,
            label=f"{LABEL[name]}  (alpha {th['alpha_per_k'] * 1e6:.1f}e-6 /K)",
        )
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("mean lattice parameter (A)")
    ax.set_title(
        "Thermal expansion of TiZrNb from NPT MD (classical, zero pressure)",
        loc="left",
        color=INK,
    )
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "thermal.png")
    plt.close(fig)


def fig_nve() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name in MODELS:
        nve = suite(name, "nve")
        if not nve:
            continue
        t = np.array(nve["times_ps"])
        e = np.array(nve["e_tot_ev_atom"])
        de = (e - e[0]) * 1000.0
        gate = "pass" if nve["gate_passed"] else "FAIL"
        ax.plot(
            t,
            de,
            color=COLOR[name],
            lw=1.6,
            label=f"{LABEL[name]}  ({nve['drift_mev_atom_ps']:.3g} meV/atom/ps, {gate})",
        )
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("total-energy change (meV/atom)")
    ax.set_title("NVE stability gate: velocity Verlet, 2 fs, 300 K, TiZrNb", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG / "nve.png")
    plt.close(fig)


def fig_split_gap(split_gap: dict) -> None:
    s = split_gap["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    for ax, key, ylabel in (
        (axes[0], "e_mae_mev_atom", "test energy MAE (meV/atom)"),
        (axes[1], "f_mae_mev_a", "test force MAE (meV/A)"),
    ):
        vals = [s["group"][key]["mean"], s["random_frame"][key]["mean"]]
        errs = [s["group"][key]["std"], s["random_frame"][key]["std"]]
        bars = ax.bar(
            [0, 1],
            vals,
            0.55,
            yerr=errs,
            capsize=4,
            color=["#2a78d6", "#898781"],
            error_kw={"ecolor": INK, "lw": 1},
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=9, color=INK, padding=3)
        ax.set_xticks([0, 1], ["honest\n(group split)", "leaky\n(random frames)"])
        ax.set_ylabel(ylabel)
    fig.suptitle(
        f"What the split protocol claims (compact model, {len(split_gap['runs']) // 2} seeds)",
        x=0.01,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "split_gap.png")
    plt.close(fig)


def main() -> int:
    from alloybench.anchors import load_anchors

    FIG.mkdir(parents=True, exist_ok=True)
    anchors = load_anchors(REPO / "data")
    metrics = load(REPO / "results" / "metrics.json")
    accuracy = load(REPO / "results" / "accuracy.json")
    split_gap = load(REPO / "results" / "split_gap.json")
    if metrics and metrics.get("pareto"):
        fig_pareto(metrics)
    if accuracy:
        fig_parity(accuracy)
    fig_eos(anchors)
    fig_elastic_vacancy(anchors)
    fig_rdf()
    fig_thermal()
    fig_nve()
    if split_gap:
        fig_split_gap(split_gap)
    print(f"figures written to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

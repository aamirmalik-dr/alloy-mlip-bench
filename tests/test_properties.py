"""Property machinery validated on cheap reference calculators.

The Lennard-Jones calculator from ASE is the workhorse: it is fast,
deterministic, and for a centrosymmetric crystal of pair interactions at
zero pressure the Cauchy relation C12 = C44 must hold, which exercises the
EOS bulk modulus and both finite-difference shear paths at once.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase.calculators.lj import LennardJones

from alloybench.properties import (
    compute_rdf,
    elastic_constants,
    eos_scan,
    expansion_coefficient,
    first_peak,
    fit_birch_murnaghan,
    nve_drift,
    speed_benchmark,
    vacancy_formation,
)

# Minimum near the BCC first-shell distance of Nb (~2.87 A).
LJ = dict(sigma=2.87 / 2 ** (1 / 6), epsilon=0.4, rc=5.5, smooth=True)


def lj():
    return LennardJones(**LJ)


def test_birch_murnaghan_fit_recovers_synthetic_minimum():
    v = np.linspace(14.0, 20.0, 11)
    eta = v ** (-2.0 / 3.0)
    eta0 = 17.0 ** (-2.0 / 3.0)
    e = 3.0 + 5.0 * (eta - eta0) ** 2 + 0.8 * (eta - eta0) ** 3
    fit = fit_birch_murnaghan(v, e)
    assert abs(fit["v0_a3"] - 17.0) < 1e-6
    assert abs(fit["e0_ev"] - 3.0) < 1e-9
    assert fit["b0_gpa"] > 0
    assert fit["fit_rms_ev"] < 1e-10


@pytest.fixture(scope="module")
def lj_eos():
    # Scan around the LJ minimum (near a = 3.19 A), not the Nb Vegard guess.
    return eos_scan(lj(), "Nb", reps=2, seed=0, span=0.12, n_points=9, a=3.19)


def test_eos_scan_recenters_from_a_bad_starting_guess():
    """Start far from the LJ minimum; the walk must still find and fit it."""
    res = eos_scan(lj(), "Nb", reps=2, seed=0, span=0.10, n_points=9, a=3.9)
    assert res["n_scans"] > 1
    assert abs(res["a0_angstrom"] - 3.187) < 0.02
    v = np.array(res["volumes_a3"])
    assert v.min() < res["v0_a3"] < v.max()


def test_eos_scan_finds_interior_minimum(lj_eos):
    factors = np.array(lj_eos["volume_factors"])
    v = np.array(lj_eos["volumes_a3"])
    assert v.min() < lj_eos["v0_a3"] < v.max()
    assert lj_eos["b0_gpa"] > 0
    assert len(factors) == 9


def test_cauchy_relation_for_pair_potential(lj_eos):
    """C12 = C44 for central pair forces at equilibrium (Cauchy relation)."""
    res = elastic_constants(
        lj(),
        "Nb",
        a0=lj_eos["a0_angstrom"],
        b0_gpa=lj_eos["b0_gpa"],
        reps=2,
        seed=0,
    )
    assert res["c44_gpa"] > 0
    assert abs(res["c12_gpa"] - res["c44_gpa"]) / res["c44_gpa"] < 0.15
    # Quadratic fits must actually describe the energies.
    assert res["fit_rms_ev"]["ortho"] < 1e-4
    assert res["fit_rms_ev"]["mono"] < 1e-4


def test_rdf_of_uniform_gas_is_flat():
    rng = np.random.default_rng(0)
    cell = np.eye(3) * 12.0
    snapshots = [(rng.uniform(0, 12.0, (200, 3)), cell) for _ in range(20)]
    r, g = compute_rdf(snapshots, r_max=5.0, n_bins=40)
    # Ignore the smallest bins (few counts); the rest must hover around 1.
    assert abs(np.mean(g[10:]) - 1.0) < 0.05


def test_rdf_of_bcc_crystal_peaks_at_first_shell():
    from alloybench.lattice import bcc_supercell

    fr = bcc_supercell("Nb", reps=3, seed=0)
    r, g = compute_rdf([(fr.positions, fr.cell)], r_max=4.8, n_bins=160)
    peak = first_peak(r, g, lo=2.0, hi=3.5)
    expected = 3.31 * np.sqrt(3.0) / 2.0
    assert abs(peak["position_angstrom"] - expected) < 0.1


def test_vacancy_formation_bookkeeping(lj_eos):
    res = vacancy_formation(
        lj(),
        "Nb",
        a0=lj_eos["a0_angstrom"],
        reservoir_e_atom={"Nb": lj_eos["e0_per_atom_ev"]},
        reps=2,
        seed=0,
        n_sites=1,
        relax_steps=60,
    )
    site = res["sites"][0]
    assert site["element"] == "Nb"
    assert np.isfinite(site["ef_ev"])
    assert res["ef_mean_ev"] == pytest.approx(site["ef_ev"])
    # Removing an atom from a bound crystal must cost energy.
    assert res["ef_mean_ev"] > 0


def test_nve_drift_reports_gate(lj_eos):
    res = nve_drift(
        lj(),
        "Nb",
        a0=lj_eos["a0_angstrom"],
        temperature_k=80.0,
        reps=2,
        seed=0,
        steps=120,
        record_every=4,
        settle_steps=20,
    )
    assert res["drift_mev_atom_ps"] >= 0
    assert isinstance(res["gate_passed"], bool)
    assert len(res["times_ps"]) == len(res["e_tot_ev_atom"])


def test_speed_benchmark_returns_positive_throughput(lj_eos):
    res = speed_benchmark(
        lj(), "Nb", a0=lj_eos["a0_angstrom"], reps=2, seed=0, warmup_steps=2, timed_steps=10
    )
    assert res["ms_per_atom_step"] > 0
    assert res["n_atoms"] == 16


def test_expansion_coefficient_linear_case():
    alpha = expansion_coefficient([300.0, 600.0, 900.0], [3.30, 3.31, 3.32])
    assert alpha == pytest.approx((0.01 / 300.0) / 3.30, rel=1e-6)

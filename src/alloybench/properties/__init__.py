"""Property benchmarks consuming only the ASE calculator interface."""

from alloybench.properties.elastic import elastic_constants
from alloybench.properties.eos import eos_scan, fit_birch_murnaghan
from alloybench.properties.rdf import compute_rdf, first_peak, rdf_benchmark
from alloybench.properties.relax import relax_positions
from alloybench.properties.speed import speed_benchmark
from alloybench.properties.stability import nve_drift
from alloybench.properties.thermal import expansion_coefficient, npt_lattice_parameter
from alloybench.properties.vacancy import vacancy_formation

__all__ = [
    "elastic_constants",
    "eos_scan",
    "fit_birch_murnaghan",
    "compute_rdf",
    "first_peak",
    "rdf_benchmark",
    "relax_positions",
    "speed_benchmark",
    "nve_drift",
    "expansion_coefficient",
    "npt_lattice_parameter",
    "vacancy_formation",
]

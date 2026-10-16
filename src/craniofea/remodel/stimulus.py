"""Daily mechanical stimulus from a loading history.

Per element e and load case i the effective strain is

    eps_ie = sqrt(2 U_ie / E_eff,e)

with U the strain energy density and E_eff the isotropic-equivalent
(Voigt-Reuss-Hill) Young's modulus of the element's current stiffness C_e
(`craniofea.materials.hill_modulus`; equal to E for isotropic elements). For
uniaxial stress this is exactly the axial strain; in general it is the strain
of an isotropic element of modulus E_eff storing the same energy. It is the
strain analogue of the effective stress sigma_bar = sqrt(2 E U) of Carter,
Fyhrie & Whalen (1987) and Beaupré, Orr & Carter (1990).

The daily stimulus combines the events of a day with the cycle-number
weighting of Carter et al. (1987) / Beaupré et al. (1990):

    Psi_e = ( sum_i n_i eps_ie^m )^(1/m)

m ~ 4 weights magnitude much more than cycle number (Carter et al. 1987;
Turner 1998 for the general principle that dynamic, not static, loading
drives adaptation). n_i is the daily cycle count of cyclic events; for a
sustained (static) event n_i = `static_equivalent_cycles` (default 1 per day,
the convention of a sustained load counted as a single cycle). How strongly a
sustained load should count is not known; static loads are weakly osteogenic
in animal experiments (Rubin & Lanyon 1984; Lanyon & Rubin 1984), so the study
reports results as a function of this parameter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from craniofea.config import StimulusConfig
from craniofea.materials import hill_modulus
from craniofea.model import Material, Mesh, Problem, Solution, Solver
from craniofea.remodel.loading import DailyLoad


def effective_strain(sed: np.ndarray, E_eff: np.ndarray) -> np.ndarray:
    """eps = sqrt(2 U / E_eff); negative round-off in U is clipped to 0."""
    return np.sqrt(2.0 * np.maximum(np.asarray(sed, float), 0.0) / np.asarray(E_eff, float))


def cycle_weights(history: list[DailyLoad], static_equivalent_cycles: float) -> np.ndarray:
    """(L,) daily cycle weight n_i of each event."""
    return np.array([static_equivalent_cycles if d.static else d.case.cycles_per_day
                     for d in history], dtype=float)


def daily_stimulus(strains: np.ndarray, n: np.ndarray, m: float) -> np.ndarray:
    """Psi = (sum_i n_i eps_i^m)^(1/m) for strains (L, M) and weights (L,).

    Evaluated as s * (sum_i n_i (eps_i/s)^m)^(1/m) with s = max_i eps_i to
    avoid under/overflow for large m.
    """
    eps = np.atleast_2d(np.asarray(strains, float))
    n = np.asarray(n, float)
    s = eps.max(axis=0)
    safe = np.where(s > 0, s, 1.0)
    total = np.einsum("i,im->m", n, (eps / safe) ** m)
    return np.where(s > 0, safe * total ** (1.0 / m), 0.0)


@dataclass
class StimulusResult:
    psi: np.ndarray                    # (M,) daily stimulus
    strains: dict[str, np.ndarray]     # case name -> (M,) effective strain
    weights: dict[str, float]          # case name -> daily cycle weight n_i
    m: float
    info: dict

    def shares(self, elems: np.ndarray | None = None,
               volume: np.ndarray | None = None) -> dict[str, float]:
        """Fraction of sum_e V_e Psi_e^m contributed by each case (over `elems`)."""
        sel = slice(None) if elems is None else np.asarray(elems)
        v = np.ones_like(self.psi)[sel] if volume is None else np.asarray(volume)[sel]
        terms = {k: float(np.sum(v * self.weights[k] * self.strains[k][sel] ** self.m))
                 for k in self.strains}
        tot = sum(terms.values())
        return {k: (t / tot if tot > 0 else 0.0) for k, t in terms.items()}


def compute_stimulus(mesh: Mesh, material: Material, history: list[DailyLoad],
                     solver: Solver, cfg: StimulusConfig,
                     fixed_nodes: np.ndarray | None = None) -> StimulusResult:
    """Solve all events of `history` with ONE Problem and return Psi.

    All load cases share the stiffness matrix, so a direct or preconditioned
    solver factorises/sets up once per call.
    """
    fixed = mesh.node_sets["fixed"] if fixed_nodes is None else fixed_nodes
    problem = Problem(mesh, material, fixed_nodes=np.asarray(fixed, dtype=np.int64),
                      load_cases=[d.case for d in history])
    sols: dict[str, Solution] = solver.solve(problem)
    failed = [k for k, s in sols.items() if s.info.get("converged") is False]
    if failed:
        raise RuntimeError(f"linear solve did not converge for load case(s) {failed}: "
                           + ", ".join(f"{k}: residual {sols[k].info.get('relative_residual')}"
                                       for k in failed))
    E_eff = hill_modulus(material.C)
    strains = {d.name: effective_strain(sols[d.name].sed, E_eff) for d in history}
    n = cycle_weights(history, cfg.static_equivalent_cycles)
    psi = daily_stimulus(np.stack([strains[d.name] for d in history]), n, cfg.m)
    info = {k: dict(s.info) for k, s in sols.items()}
    return StimulusResult(psi=psi, strains=strains,
                          weights={d.name: float(w) for d, w in zip(history, n, strict=True)},
                          m=cfg.m, info=info)

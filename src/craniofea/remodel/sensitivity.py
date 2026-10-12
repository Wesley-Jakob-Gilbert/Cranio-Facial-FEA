"""Sensitivity analyses of the habit contrast.

Response (default): final volume-weighted mean delta-rho of the palatal
vault, mewing minus mouth_breathing (``cfg.sensitivity.region`` /
``cfg.sensitivity.contrast``).

(a) `sweep_static_cycles`: the response as a function of the daily cycle
    weight given to sustained loads - the central unknown of the hypothesis.
(b) `morris`: elementary-effects screening (Morris 1991) with the mu*
    statistic of Campolongo, Cariboni & Saltelli (2007). Each of r
    trajectories starts at a random point of a p-level grid in the unit
    hypercube and moves one factor at a time by Delta = p / (2 (p - 1)).
    Elementary effect of factor j:  EE_j = [f(x + Delta e_j) - f(x)] / Delta
    in unit-cube coordinates (so effects are comparable across factors and
    equal (hi - lo) * df/dx for a linear response in a linear-scaled factor).
    Reported per factor: mu (mean EE), mu* (mean |EE|), sigma (std of EE).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from craniofea.config import ParamRange, StudyConfig
from craniofea.model import Mesh, Solver
from craniofea.remodel.simulate import run_study


def contrast_response(mesh: Mesh, cfg: StudyConfig, solver: Solver) -> dict:
    """Run the two contrast scenarios and return the response and region details."""
    a, b = cfg.sensitivity.contrast
    res = run_study(mesh, cfg, solver, scenarios=[a, b])
    regions = sorted({r["region"] for r in res.summary})
    return {
        "response": res.contrast(a, b, cfg.sensitivity.region),
        "by_region": {r: res.contrast(a, b, r) for r in regions},
        a: {r: res.final_mean_drho(a, r) for r in regions},
        b: {r: res.final_mean_drho(b, r) for r in regions},
        "shares": res.shares,
    }


def sweep_static_cycles(mesh: Mesh, cfg: StudyConfig, solver: Solver,
                        values: list[float] | None = None,
                        progress: Callable[[str], None] | None = None) -> list[dict]:
    """Response for each static_equivalent_cycles value (setpoint recomputed each time)."""
    vals = cfg.sensitivity.static_cycles if values is None else values
    rows = []
    for v in vals:
        c = cfg.with_overrides({"stimulus.static_equivalent_cycles": float(v)})
        out = contrast_response(mesh, c, solver)
        rows.append({"static_equivalent_cycles": float(v), **out})
        if progress:
            progress(f"static_equivalent_cycles={v:g}: response={out['response']:.4g}")
    return rows


def critical_static_cycles(mesh: Mesh, cfg: StudyConfig, solver: Solver,
                           scenario: str = "mewing") -> dict:
    """Smallest static_equivalent_cycles at which `scenario` leaves the lazy zone.

    At rho = 1 split each daily stimulus into cyclic and sustained parts,
    Psi^m = A + n B  (A = sum over cyclic events n_i eps_i^m, B = sum over
    sustained events eps_i^m, n = static_equivalent_cycles). Formation starts
    where Psi_s / Psi_ref = 1 + w, i.e.

        n_crit = [(1 + w)^m A_ref - A_s] / [B_s - (1 + w)^m B_ref]

    (finite only where the denominator is positive). This is the first-step,
    unsmoothed threshold; it does not depend on rates or on time.
    Returns per-element n_crit (inf where never reached) and region minima.
    """
    from craniofea.materials import build_material
    from craniofea.remodel.loading import build_daily_loading
    from craniofea.remodel.simulate import REGIONS
    from craniofea.remodel.stimulus import compute_stimulus

    m, w = cfg.stimulus.m, cfg.remodeling.lazy_zone
    mat = build_material(mesh, 1.0, cfg.materials)
    parts = {}
    for s in ("reference", scenario):
        hist = build_daily_loading(mesh, cfg.loading, s)
        st = compute_stimulus(mesh, mat, hist, solver, cfg.stimulus)
        A, B = np.zeros(mesh.n_elems), np.zeros(mesh.n_elems)
        for d in hist:
            if d.static:
                B += st.strains[d.name] ** m
            else:
                A += d.case.cycles_per_day * st.strains[d.name] ** m
        parts[s] = (A, B)
    (Ar, Br), (As, Bs) = parts["reference"], parts[scenario]
    k = (1.0 + w) ** m
    num, den = k * Ar - As, Bs - k * Br
    bone = np.ones(mesh.n_elems, dtype=bool)
    bone[np.asarray(mesh.elem_sets["suture_mid"], dtype=np.int64)] = False   # not remodeled
    with np.errstate(divide="ignore", invalid="ignore"):
        n_crit = np.where((den > 0) & bone, np.maximum(num, 0.0) / den, np.inf)
    out = {"scenario": scenario, "lazy_zone": w, "m": m, "n_crit": n_crit}
    for r in REGIONS:
        ids = np.asarray(mesh.elem_sets[r], dtype=np.int64)
        v = n_crit[ids]
        out[r] = {"min": float(v.min()), "median": float(np.median(v)),
                  "frac_finite": float(np.isfinite(v).mean())}
    return out


# --------------------------------------------------------------------------- #
# Morris elementary effects
# --------------------------------------------------------------------------- #

@dataclass
class MorrisResult:
    names: list[str]
    mu: np.ndarray
    mu_star: np.ndarray
    sigma: np.ndarray
    effects: np.ndarray          # (r, k) elementary effects
    X: np.ndarray                # (r, k+1, k) unit-cube design
    Y: np.ndarray                # (r, k+1) responses
    meta: dict = field(default_factory=dict)

    def table(self) -> list[dict]:
        order = np.argsort(-self.mu_star)
        return [{"parameter": self.names[i], "mu": float(self.mu[i]),
                 "mu_star": float(self.mu_star[i]), "sigma": float(self.sigma[i])}
                for i in order]


def morris_design(k: int, r: int, levels: int, rng: np.random.Generator):
    """(r, k+1, k) trajectories in [0, 1]^k and (r, k) factor order / step signs."""
    delta = levels / (2.0 * (levels - 1))
    jump = levels // 2
    X = np.empty((r, k + 1, k))
    order = np.empty((r, k), dtype=np.int64)
    sign = np.empty((r, k))
    for t in range(r):
        lvl = rng.integers(0, levels, size=k)
        d = rng.choice([-1, 1], size=k)
        d = np.where((lvl + d * jump < 0) | (lvl + d * jump > levels - 1), -d, d)
        x = lvl / (levels - 1)
        perm = rng.permutation(k)
        X[t, 0] = x
        for s, j in enumerate(perm):
            x = x.copy()
            x[j] += d[j] * delta
            X[t, s + 1] = x
        order[t], sign[t] = perm, d[perm]
    return X, order, sign, delta


def scale_unit(x: np.ndarray, ranges: list[ParamRange]) -> np.ndarray:
    """Map unit-cube coordinates to physical values (log-uniform where requested)."""
    out = np.empty_like(x, dtype=float)
    for j, p in enumerate(ranges):
        if p.log:
            out[..., j] = np.exp(np.log(p.lo) + x[..., j] * (np.log(p.hi) - np.log(p.lo)))
        else:
            out[..., j] = p.lo + x[..., j] * (p.hi - p.lo)
    return out


def morris(func: Callable[[dict[str, float]], float], params: dict[str, ParamRange],
           trajectories: int = 10, levels: int = 4, seed: int = 0,
           progress: Callable[[str], None] | None = None) -> MorrisResult:
    """Morris screening of a scalar function of named parameters."""
    names = list(params)
    ranges = [params[n] for n in names]
    k = len(names)
    rng = np.random.default_rng(seed)
    X, order, sign, delta = morris_design(k, trajectories, levels, rng)
    P = scale_unit(X, ranges)
    Y = np.empty((trajectories, k + 1))
    for t in range(trajectories):
        for s in range(k + 1):
            Y[t, s] = func(dict(zip(names, P[t, s].tolist(), strict=True)))
        if progress:
            progress(f"morris trajectory {t + 1}/{trajectories}")
    EE = np.empty((trajectories, k))
    for t in range(trajectories):
        dY = np.diff(Y[t])
        EE[t, order[t]] = dY * sign[t] / delta
    return MorrisResult(names=names, mu=EE.mean(axis=0), mu_star=np.abs(EE).mean(axis=0),
                        sigma=EE.std(axis=0, ddof=1) if trajectories > 1 else np.zeros(k),
                        effects=EE, X=X, Y=Y,
                        meta={"trajectories": trajectories, "levels": levels, "seed": seed,
                              "delta": delta, "evaluations": int(Y.size)})


def morris_study(mesh: Mesh, cfg: StudyConfig, solver: Solver,
                 progress: Callable[[str], None] | None = None) -> MorrisResult:
    """Morris screening of the study response over cfg.sensitivity.morris.parameters."""
    mc = cfg.sensitivity.morris

    def f(values: dict[str, float]) -> float:
        return contrast_response(mesh, cfg.with_overrides(values), solver)["response"]

    return morris(f, mc.parameters, mc.trajectories, mc.levels, mc.seed, progress)

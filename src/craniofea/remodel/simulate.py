"""Study runner: setpoint, per-scenario density evolution, summaries, provenance.

Algorithm (see docs/METHODS.md)::

    Psi* = Psi(reference habit, rho = 1)
    for each scenario:
        rho^0 = 1
        for k = 0 .. n_steps-1:
            Psi^k = Psi(scenario habit, rho^k)      # one FE Problem, all load cases
            S^k   = smooth(Psi^k / Psi* - 1)
            rho^{k+1} = clip(rho^k + dt * rate(S^k))
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from craniofea.config import MeshConfig, StudyConfig
from craniofea.materials import LAYERS, build_material, layer_index
from craniofea.mesh.primitives import tet_signed_volumes
from craniofea.model import Mesh, Solver
from craniofea.remodel.law import nodal_smoother, remodel_rate, stimulus_error, update_density
from craniofea.remodel.loading import build_daily_loading
from craniofea.remodel.stimulus import StimulusResult, compute_stimulus

REGIONS = ("region_palatal_vault", "region_alveolar_anterior",
           "region_alveolar_posterior", "region_basal")
REQUIRED_ELEM_SETS = (*LAYERS, *REGIONS)
_TOL = 1e-9


@dataclass
class StudyResult:
    config: StudyConfig
    times: np.ndarray                         # (K+1,) days
    setpoint: np.ndarray                      # (M,) Psi*
    rho: dict[str, np.ndarray]                # scenario -> (K+1, M)
    error: dict[str, np.ndarray]              # scenario -> (K, M) smoothed S
    summary: list[dict]                       # one row per scenario x step x region
    shares: dict[str, dict[str, dict[str, float]]]   # scenario -> region -> case -> share at k=0
    manifest: dict = field(default_factory=dict)

    def final_mean_drho(self, scenario: str, region: str) -> float:
        rows = [r for r in self.summary if r["scenario"] == scenario and r["region"] == region]
        return float(max(rows, key=lambda r: r["step"])["mean_drho"])

    def contrast(self, a: str, b: str, region: str = "region_palatal_vault") -> float:
        """Final volume-weighted mean delta-rho difference a - b in `region`."""
        return self.final_mean_drho(a, region) - self.final_mean_drho(b, region)

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "setpoint.npz", psi_star=self.setpoint)
        for s, r in self.rho.items():
            np.savez_compressed(out / f"density_{s}.npz", times=self.times, rho=r,
                                error=self.error[s])
        if self.summary:
            with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.summary[0]))
                w.writeheader()
                w.writerows(self.summary)
        (out / "stimulus_shares.json").write_text(json.dumps(self.shares, indent=2))
        (out / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=str))
        return out


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def _git_commit() -> dict:
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10).stdout.strip() != ""
        return {"commit": sha, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("craniofea", "numpy", "scipy", "jax", "pyamg", "pyyaml"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    import craniofea

    out["craniofea_source"] = craniofea.__version__
    return out


def provenance(cfg: StudyConfig, mesh: Mesh, solver: Solver, extra: dict | None = None) -> dict:
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": cfg.hash(),
        "config": cfg.to_dict(),
        "git": _git_commit(),
        "versions": _versions(),
        "platform": platform.platform(),
        "solver": getattr(solver, "name", type(solver).__name__),
        "mesh": {"n_nodes": mesh.n_nodes, "n_elems": mesh.n_elems, "meta": mesh.meta},
        **(extra or {}),
    }


# --------------------------------------------------------------------------- #
# Study
# --------------------------------------------------------------------------- #

def load_study_mesh(mcfg: MeshConfig,
                    log: Callable[[str], None] | None = None) -> tuple[Mesh, str]:
    """Mesh for a study: `mcfg.source`, then each `mcfg.fallback` in order.

    Returns (mesh, source actually used). Every fallback is reported through
    `log` and recorded in the mesh meta, so a substitution is never silent.
    """
    log = log or (lambda msg: None)
    errors = []
    for src in (mcfg.source, *mcfg.fallback):
        try:
            if src == "file":
                if not mcfg.path or not Path(mcfg.path).is_file():
                    raise FileNotFoundError(f"mesh file not found: {mcfg.path}")
                mesh = Mesh.load(mcfg.path)
            elif src == "template":
                from craniofea.mesh.template import build_template_mesh

                mesh = build_template_mesh(**mcfg.template)
            else:
                from craniofea.remodel.demo import labeled_box_mesh

                mesh = labeled_box_mesh(**mcfg.box)
        except (ImportError, OSError, ValueError, KeyError) as e:
            errors.append(f"{src}: {e}")
            log(f"mesh source '{src}' unavailable ({e}); trying next")
            continue
        if errors:
            mesh.meta = {**mesh.meta, "mesh_fallback_from": errors}
        return mesh, src
    raise RuntimeError("no mesh source succeeded: " + "; ".join(errors))


def check_mesh(mesh: Mesh) -> None:
    """Raise if the mesh lacks a canonical label required by the study."""
    missing = [s for s in REQUIRED_ELEM_SETS if s not in mesh.elem_sets]
    missing += [f"surface:{s}" for s in ("palate", "occlusal") if s not in mesh.surface_sets]
    if "fixed" not in mesh.node_sets:
        missing.append("node:fixed")
    if mesh.frame is None:
        missing.append("frame")
    if missing:
        raise ValueError(f"mesh lacks canonical labels: {missing}")


class _Layered:
    """Per-element arrays of the remodeling parameters."""

    def __init__(self, mesh: Mesh, cfg: StudyConfig):
        lid = layer_index(mesh)
        self.bone = lid != LAYERS.index("suture_mid")
        r = cfg.remodeling

        def per_layer(v, suture):
            return np.array([v.cortical, v.cancellous, suture])[lid]

        self.c_form = per_layer(r.formation_rate, 0.0)
        self.c_res = per_layer(r.resorption_rate, 0.0)
        self.rho_min = per_layer(r.rho_min, 1.0)
        self.rho_max = per_layer(r.rho_max, 1.0)


def compute_setpoint(mesh: Mesh, cfg: StudyConfig, solver: Solver) -> StimulusResult:
    """Daily stimulus of the reference habit at rho = 1 (the homeostatic setpoint)."""
    material = build_material(mesh, np.ones(mesh.n_elems), cfg.materials)
    history = build_daily_loading(mesh, cfg.loading, "reference")
    return compute_stimulus(mesh, material, history, solver, cfg.stimulus)


def run_study(mesh: Mesh, cfg: StudyConfig, solver: Solver,
              scenarios: list[str] | None = None,
              progress: Callable[[str], None] | None = None) -> StudyResult:
    """Run every scenario (default: all in `cfg.loading.scenarios`) for cfg.remodeling.days."""
    check_mesh(mesh)
    t0 = time.perf_counter()
    log = progress or (lambda msg: None)
    scen = list(cfg.loading.scenarios) if scenarios is None else list(scenarios)
    for s in scen:
        if s not in cfg.loading.scenarios:
            raise KeyError(f"unknown scenario {s!r}")

    rc = cfg.remodeling
    K, dt = rc.n_steps, rc.dt
    vol = tet_signed_volumes(mesh.nodes, mesh.tets)
    lay = _Layered(mesh, cfg)
    smoother = nodal_smoother(mesh, vol, lay.bone) if rc.smoothing else None

    ref = compute_setpoint(mesh, cfg, solver)
    psi_star = ref.psi
    log(f"setpoint computed ({len(ref.strains)} load cases)")

    regions = {r: np.asarray(mesh.elem_sets[r], dtype=np.int64) for r in REGIONS}
    region_bone = {r: ids[lay.bone[ids]] for r, ids in regions.items()}

    rho_hist: dict[str, np.ndarray] = {}
    err_hist: dict[str, np.ndarray] = {}
    shares: dict[str, dict[str, dict[str, float]]] = {}
    summary: list[dict] = []
    solver_info: dict = {}
    for s in scen:
        history = build_daily_loading(mesh, cfg.loading, s)
        rho = np.ones(mesh.n_elems)
        R = np.empty((K + 1, mesh.n_elems))
        E = np.empty((K, mesh.n_elems))
        R[0] = rho
        for k in range(K):
            material = build_material(mesh, rho, cfg.materials)
            stim = compute_stimulus(mesh, material, history, solver, cfg.stimulus)
            if k == 0:
                shares[s] = {r: stim.shares(ids, vol) for r, ids in region_bone.items()}
            S = stimulus_error(stim.psi, psi_star, cfg.stimulus.setpoint_floor)
            S = np.where(lay.bone, S, 0.0)
            if smoother is not None:
                S = np.where(lay.bone, smoother @ S, 0.0)
            rate = remodel_rate(S, rc.lazy_zone, lay.c_form, lay.c_res)
            rho = update_density(rho, rate, dt, lay.rho_min, lay.rho_max)
            E[k], R[k + 1] = S, rho
            solver_info = next(iter(stim.info.values()), {})
        rho_hist[s], err_hist[s] = R, E
        summary += _summarise(s, R, rc, region_bone, vol)
        log(f"scenario {s}: done")

    result = StudyResult(config=cfg, times=np.arange(K + 1) * dt, setpoint=psi_star,
                         rho=rho_hist, error=err_hist, summary=summary, shares=shares)
    result.manifest = provenance(cfg, mesh, solver, {
        "scenarios": scen, "wall_time_s": time.perf_counter() - t0,
        "solver_info_last": solver_info,
        "load_cases_reference": sorted(ref.strains),
    })
    return result


def _summarise(scenario: str, R: np.ndarray, rc, regions: dict[str, np.ndarray],
               vol: np.ndarray) -> list[dict]:
    rows = []
    for k in range(R.shape[0]):
        d = R[k] - R[0]
        for name, ids in regions.items():
            v = vol[ids]
            V = v.sum()
            if V <= 0:
                continue
            rows.append({
                "scenario": scenario, "step": k, "day": k * rc.dt, "region": name,
                "mean_rho": float(np.sum(v * R[k, ids]) / V),
                "mean_drho": float(np.sum(v * d[ids]) / V),
                "frac_forming": float(v[d[ids] > _TOL].sum() / V),
                "frac_resorbing": float(v[d[ids] < -_TOL].sum() / V),
            })
    return rows

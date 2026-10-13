#!/usr/bin/env python3
"""Regenerate every result reported in the README.

1. CT study (Open-Full-Jaw patient 3) at the literature default
   static_equivalent_cycles = 1, and at 1e6 / 1e7 to show what weighting of
   sustained loads the mewing hypothesis would require.
2. Per-element critical static_equivalent_cycles on the CT mesh (analytic).
3. Static-cycles sweep + Morris screening on the parametric template mesh.

Usage:  python scripts/reproduce_results.py [--skip-ct] [--skip-sensitivity]
Outputs go to results/ (gitignored); figures are made by scripts/make_figures.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from craniofea.config import load_config
from craniofea.remodel.sensitivity import (
    critical_static_cycles,
    morris_study,
    sweep_static_cycles,
)
from craniofea.remodel.simulate import load_study_mesh, provenance, run_study
from craniofea.solvers import get_solver

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MORRIS_TRAJECTORIES = 6        # 6 x (14 + 1) = 90 template studies
CT_STATIC_CYCLES = {"ct_default": 1.0, "ct_static_1e6": 1.0e6, "ct_static_1e7": 1.0e7}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ct_studies() -> None:
    cfg = load_config(ROOT / "configs" / "study.yaml")
    mesh, src = load_study_mesh(cfg.mesh, log)
    if "template" in src:
        raise SystemExit("CT mesh not found; run `python scripts/build_meshes.py ct` first")
    solver = get_solver(cfg.solver.name, **cfg.solver.options)
    region = cfg.sensitivity.region
    summary = {}
    for name, n_static in CT_STATIC_CYCLES.items():
        out = RESULTS / name
        if (out / "summary.csv").exists():
            log(f"{name}: exists, skipping")
            continue
        t0 = time.perf_counter()
        c = cfg.with_overrides({"stimulus.static_equivalent_cycles": n_static})
        # The reference habit is homeostatic by construction at any weighting;
        # it is simulated once (default run) as a check and skipped afterwards.
        scen = None if name == "ct_default" else list(cfg.sensitivity.contrast)
        res = run_study(mesh, c, solver, scenarios=scen, progress=log)
        res.save(out)
        summary[name] = {
            "static_equivalent_cycles": n_static,
            "final_mean_drho": {s: res.final_mean_drho(s, region) for s in res.rho},
            "contrast": res.contrast(*cfg.sensitivity.contrast, region),
            "wall_time_s": time.perf_counter() - t0,
        }
        log(f"{name}: {json.dumps(summary[name])}")
    if summary:
        path = RESULTS / "ct_summary.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**old, **summary}, indent=2))

    out = RESULTS / "ct_critical_static_cycles"
    if not (out / "critical_static_cycles.json").exists():
        out.mkdir(parents=True, exist_ok=True)
        crit = critical_static_cycles(mesh, cfg, solver, scenario=cfg.sensitivity.contrast[0])
        np.savez_compressed(out / "critical_static_cycles.npz", n_crit=crit.pop("n_crit"))
        (out / "critical_static_cycles.json").write_text(json.dumps(crit, indent=2))
        log(f"critical static cycles: {json.dumps(crit)}")


def template_sensitivity() -> None:
    cfg = load_config(ROOT / "configs" / "study.yaml").with_overrides(
        {"mesh.source": "template", "mesh.path": None,
         "sensitivity.morris.trajectories": MORRIS_TRAJECTORIES})
    mesh, src = load_study_mesh(cfg.mesh, log)
    solver = get_solver(cfg.solver.name, **cfg.solver.options)
    out = RESULTS / "template_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(provenance(cfg, mesh, solver, {"analysis": "sweep+morris"}), indent=2,
                   default=str))
    rows = sweep_static_cycles(mesh, cfg, solver, progress=log)
    (out / "static_cycles_sweep.json").write_text(json.dumps(rows, indent=2))
    mr = morris_study(mesh, cfg, solver, progress=log)
    (out / "morris.json").write_text(json.dumps({"meta": mr.meta, "table": mr.table()},
                                                indent=2))
    log("template sensitivity done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-ct", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)                      # config paths are relative to the repo root
    RESULTS.mkdir(exist_ok=True)
    if not args.skip_ct:
        ct_studies()
    if not args.skip_sensitivity:
        template_sensitivity()


if __name__ == "__main__":
    main()

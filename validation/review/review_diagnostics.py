"""Small implementation diagnostics, distinct from independent_calibration.py.

This file intentionally imports production functions to reproduce behavior.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from craniofea.config import load_config
from craniofea.fem.jax_solver import JaxSolver
from craniofea.mesh.primitives import box_mesh, tet_signed_volumes
from craniofea.mesh.template import build_template_mesh
from craniofea.remodel.law import nodal_smoother, stimulus_error, remodel_rate
from craniofea.remodel.simulate import run_study


def main():
    # All six tets share some nodes. An inactive setpoint-floor element must
    # remain fixed per METHODS.md, even next to positively stimulated elements.
    mesh = box_mesh(1, 1, 1, 1, 1, 1)
    vol = tet_signed_volumes(mesh.nodes, mesh.tets)
    active = np.ones(mesh.n_elems, dtype=bool)
    target = np.ones(mesh.n_elems)
    target[0] = 0
    psi = np.full(mesh.n_elems, 2.0)
    psi[0] = 0
    raw = stimulus_error(psi, target)
    smoothed = nodal_smoother(mesh, vol, active) @ raw
    rate = remodel_rate(smoothed, .1, np.full(mesh.n_elems, 1.1e-4),
                        np.full(mesh.n_elems, 1.1e-4))
    result = {"setpoint_floor_smoothing": {
        "floor_element": 0, "raw_error": float(raw[0]),
        "smoothed_error": float(smoothed[0]), "density_rate_per_day": float(rate[0]),
        "documented_fixed_behavior_refuted": bool(rate[0] != 0)}}
    cfg = load_config(ROOT / "configs/study.yaml")
    mesh = build_template_mesh()
    result["template"] = []
    for n_static, steps in [(1.0, 12), (1e7, 12), (1e7, 24)]:
        c = cfg.with_overrides({"stimulus.static_equivalent_cycles": n_static,
                                "remodeling.n_steps": steps})
        study = run_study(mesh, c, JaxSolver(), scenarios=["mewing", "mouth_breathing"])
        region = "region_palatal_vault"
        row = {"n_static": n_static, "steps": steps,
               "mewing_mean_drho": study.final_mean_drho("mewing", region),
               "mouth_breathing_mean_drho": study.final_mean_drho("mouth_breathing", region),
               "contrast": study.contrast("mewing", "mouth_breathing", region),
               "shares_mewing": study.shares["mewing"][region],
               "max_abs_error_mewing": float(np.max(np.abs(study.error["mewing"]))),
               "manifest": study.manifest}
        result["template"].append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "manifest"}), flush=True)
        Path(__file__).with_name("review_diagnostics.json").write_text(json.dumps(result, indent=2))
    print("Floor+smoothing diagnostic:", result["setpoint_floor_smoothing"])


if __name__ == "__main__":
    main()

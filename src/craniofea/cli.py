"""``craniofea`` command-line interface.

    craniofea study CONFIG [--solver NAME] [--out DIR]
    craniofea sensitivity CONFIG [--solver NAME] [--out DIR] [--what sweep|morris|all]
    craniofea info MESH
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _log(msg: str) -> None:
    print(f"[craniofea] {msg}", file=sys.stderr, flush=True)


def _setup(args):
    from craniofea.config import load_config
    from craniofea.remodel.simulate import load_study_mesh
    from craniofea.solvers import get_solver

    cfg = load_config(args.config)
    mesh, src = load_study_mesh(cfg.mesh, _log)
    _log(f"mesh: {src} ({mesh.n_nodes} nodes, {mesh.n_elems} tets)")
    name = args.solver or cfg.solver.name
    opts = cfg.solver.options if name == cfg.solver.name else {}
    solver = get_solver(name, **opts)
    out = Path(args.out) if args.out else Path(cfg.output_dir)
    return cfg, mesh, solver, out


def cmd_study(args) -> int:
    from craniofea.remodel.simulate import run_study

    cfg, mesh, solver, out = _setup(args)
    res = run_study(mesh, cfg, solver, progress=_log)
    res.save(out)
    region = cfg.sensitivity.region
    print(f"final mean delta-rho after {cfg.remodeling.days:g} days ({region}):")
    for s in res.rho:
        print(f"  {s:<18s} {res.final_mean_drho(s, region):+.5f}")
    a, b = cfg.sensitivity.contrast
    print(f"  contrast {a} - {b}: {res.contrast(a, b, region):+.5f}")
    print(f"results written to {out}")
    return 0


def cmd_sensitivity(args) -> int:
    from craniofea.remodel.sensitivity import (
        critical_static_cycles,
        morris_study,
        sweep_static_cycles,
    )

    cfg, mesh, solver, out = _setup(args)
    if args.trajectories:
        cfg = cfg.with_overrides({"sensitivity.morris.trajectories": args.trajectories})
    out.mkdir(parents=True, exist_ok=True)
    from craniofea.remodel.simulate import provenance

    manifest = provenance(cfg, mesh, solver, {"analysis": args.what})
    (out / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if args.what in ("sweep", "all"):
        a = cfg.sensitivity.contrast[0]
        crit = critical_static_cycles(mesh, cfg, solver, scenario=a)
        np.savez_compressed(out / "critical_static_cycles.npz", n_crit=crit.pop("n_crit"))
        (out / "critical_static_cycles.json").write_text(json.dumps(crit, indent=2))
        print(f"static_equivalent_cycles at which '{a}' first leaves the lazy zone (rho = 1):")
        for r, v in crit.items():
            if isinstance(v, dict):
                print(f"  {r:<28s} min {v['min']:.3g}  median {v['median']:.3g}  "
                      f"(finite in {100 * v['frac_finite']:.0f}% of elements)")
        rows = sweep_static_cycles(mesh, cfg, solver, progress=_log)
        (out / "static_cycles_sweep.json").write_text(json.dumps(rows, indent=2))
        print("static_equivalent_cycles  response")
        for r in rows:
            print(f"  {r['static_equivalent_cycles']:>12.3g}  {r['response']:+.5f}")
    if args.what in ("morris", "all"):
        mr = morris_study(mesh, cfg, solver, progress=_log)
        table = mr.table()
        (out / "morris.json").write_text(json.dumps({"meta": mr.meta, "table": table}, indent=2))
        np.savez_compressed(out / "morris_samples.npz", X=mr.X, Y=mr.Y, effects=mr.effects,
                            names=np.array(mr.names))
        print(f"{'parameter':<48s} {'mu*':>10s} {'mu':>10s} {'sigma':>10s}")
        for r in table:
            print(f"  {r['parameter']:<46s} {r['mu_star']:10.3g} {r['mu']:+10.3g} "
                  f"{r['sigma']:10.3g}")
    print(f"results written to {out}")
    return 0


def cmd_info(args) -> int:
    from craniofea.model import Mesh

    m = Mesh.load(args.mesh)
    print(f"{args.mesh}: {m.n_nodes} nodes, {m.n_elems} tets")
    lo, hi = m.nodes.min(axis=0), m.nodes.max(axis=0)
    print(f"  bounding box (m): {np.round(lo, 4).tolist()} .. {np.round(hi, 4).tolist()}")
    for title, sets in (("node sets", m.node_sets), ("elem sets", m.elem_sets),
                        ("surface sets", m.surface_sets)):
        print(f"  {title}: " + ", ".join(f"{k}({len(v)})" for k, v in sorted(sets.items())))
    print("  elem fields: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in m.elem_fields.items()))
    if m.frame is not None:
        f = m.frame
        print(f"  frame: lr={np.round(f.lr, 3).tolist()} ap={np.round(f.ap, 3).tolist()} "
              f"si={np.round(f.si, 3).tolist()}")
    else:
        print("  frame: none")
    if m.meta:
        print(f"  meta: {json.dumps(m.meta)[:400]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="craniofea",
                                description="FE + strain-adaptive remodeling of the maxilla")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("study", help="run all habit scenarios of a study config")
    s.add_argument("config")
    s.add_argument("--solver", help="override solver.name (reference, jax, calculix, febio)")
    s.add_argument("--out", help="output directory (default: config output_dir)")
    s.set_defaults(func=cmd_study)

    s = sub.add_parser("sensitivity", help="static-load sweep and Morris screening")
    s.add_argument("config")
    s.add_argument("--solver")
    s.add_argument("--out")
    s.add_argument("--what", choices=("sweep", "morris", "all"), default="all")
    s.add_argument("--trajectories", type=int, help="override Morris trajectory count")
    s.set_defaults(func=cmd_sensitivity)

    s = sub.add_parser("info", help="summarise a mesh .npz")
    s.add_argument("mesh")
    s.set_defaults(func=cmd_info)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as e:  # report config/mesh errors without a traceback
        from craniofea.config import ConfigError

        if isinstance(e, (ConfigError, FileNotFoundError)):
            print(f"craniofea: error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    sys.exit(main())

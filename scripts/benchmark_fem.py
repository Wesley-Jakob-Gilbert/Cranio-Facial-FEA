"""Benchmark the production FE solver on high-contrast box models.

The model is a box with a stiff outer shell (cortical-like, 17 GPa), a soft
core (cancellous-like, 0.5 GPa) and a compliant mid-plane slab (suture-like,
50 MPa) that cuts through shell and core. The x = 0 face is clamped; two
load cases (pressure on top + gravity, pressure on one side) share one
preconditioner.

Each (size, method) combination runs in a fresh subprocess so that the
reported peak resident memory belongs to that case alone.

    python scripts/benchmark_fem.py                       # 100k and 850k tets, AMG
    python scripts/benchmark_fem.py --sizes 100k --methods amg jax-cg
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# hexes per axis; each hex is split into 6 tetrahedra
SIZES = {"10k": (20, 14, 12), "100k": (35, 25, 20), "300k": (50, 35, 28), "850k": (70, 50, 40)}


def build_problem(divisions: tuple[int, int, int]):
    from craniofea.fem.verification import layered_box

    return layered_box(divisions)


def run_single(size: str, method: str) -> dict:
    from craniofea.fem.jax_solver import JaxSolver

    t0 = time.perf_counter()
    prob = build_problem(SIZES[size])
    t_mesh = time.perf_counter() - t0
    solver = JaxSolver(method=method, rtol=1e-10)
    t0 = time.perf_counter()
    sols = solver.solve(prob)
    total = time.perf_counter() - t0
    info = [s.info for s in sols.values()]
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    tip = {k: float(abs(s.u).max()) for k, s in sols.items()}
    return {
        "size": size, "method": method, "n_elems": prob.mesh.n_elems,
        "n_nodes": prob.mesh.n_nodes, "n_dofs": 3 * prob.mesh.n_nodes,
        "mesh_s": t_mesh, "total_s": total,
        "assembly_s": info[0].get("assembly_s", float("nan")),
        "setup_s": info[0].get("setup_s", float("nan")),
        "solve_s": [i["solve_s"] for i in info],
        "iterations": [i["iterations"] for i in info],
        "relative_residual": [i["relative_residual"] for i in info],
        "amg_levels": info[0].get("amg_levels"),
        "operator_complexity": info[0].get("operator_complexity"),
        "device": info[0].get("device", "cpu"),
        "peak_rss_gb": rss_gb, "max_abs_u": tip,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", nargs="+", default=["100k", "850k"], choices=list(SIZES))
    ap.add_argument("--methods", nargs="+", default=["amg"], choices=["amg", "jax-cg"])
    ap.add_argument("--json", type=Path, help="write results to this file")
    ap.add_argument("--single", nargs=2, metavar=("SIZE", "METHOD"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.single:
        print(json.dumps(run_single(*args.single)))
        return

    results = []
    for size in args.sizes:
        for method in args.methods:
            env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false")
            if method == "amg":
                env.setdefault("JAX_PLATFORMS", "cpu")
            proc = subprocess.run([sys.executable, __file__, "--single", size, method],
                                  capture_output=True, text=True, env=env, check=False)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                continue
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            results.append(r)
            its = "/".join(str(i) for i in r["iterations"])
            print(f"{r['size']:>5} {r['method']:>7}  tets={r['n_elems']:>8,d} "
                  f"dofs={r['n_dofs']:>8,d}  total={r['total_s']:7.1f} s  "
                  f"assembly={r['assembly_s']:6.1f} s  setup={r['setup_s']:6.1f} s  "
                  f"solve={'/'.join(f'{s:.1f}' for s in r['solve_s'])} s  it={its}  "
                  f"res={max(r['relative_residual']):.1e}  RSS={r['peak_rss_gb']:.2f} GB  "
                  f"[{r['device']}]", flush=True)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

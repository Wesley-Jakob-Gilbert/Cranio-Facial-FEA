"""Run the FE verification suite and write the convergence figures.

Cases (closed-form references, see craniofea.fem.verification):
  * Lame thick-walled cylinder, plane strain, quarter annulus with symmetry
  * hanging column under self-weight
  * end-loaded cantilever against Timoshenko beam theory
plus agreement with the direct reference solver and an adjoint-gradient
check against central finite differences.

    python scripts/verify_fem.py            # tables + docs/figures/verification_*.png
    python scripts/verify_fem.py --quick    # coarser levels, for a smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from craniofea.fem import verification as V  # noqa: E402
from craniofea.fem.jax_solver import JaxSolver, make_differentiable_solver  # noqa: E402
from craniofea.fem.reference import ReferenceSolver  # noqa: E402

FIG_DIR = ROOT / "docs" / "figures"

# Reference categorical slots 1-3 (validated all-pairs), neutral ink, light surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def convergence(builder, levels, case, rtol=1e-11):
    solver = JaxSolver(rtol=rtol)
    rows = []
    for k in levels:
        c = builder(k)
        t0 = time.perf_counter()
        s = solver.solve(c.problem)[case]
        m = c.problem.mesh
        rows.append({
            "level": k, "h": c.h, "n_elems": m.n_elems, "n_dofs": 3 * m.n_nodes,
            "u_l2": V.l2_error(m, s.u, c.u_exact)[1],
            "stress_l2": V.stress_l2_error(m, s.stress, c.stress_exact)[1],
            "iterations": s.info["iterations"], "time_s": time.perf_counter() - t0,
        })
    return rows


def cantilever_study(levels):
    solver = JaxSolver(rtol=1e-10)
    rows = []
    for k in levels:
        c = V.cantilever(k)
        s = solver.solve(c.problem)["tip"]
        d = -s.u[c.meta["tip_nodes"], 2].mean()
        rows.append({"level": k, "h": c.h, "n_elems": c.problem.mesh.n_elems,
                     "delta": float(d), "delta_timoshenko": c.meta["delta_timoshenko"],
                     "ratio": float(d / c.meta["delta_timoshenko"]),
                     "iterations": s.info["iterations"]})
    # Richardson extrapolation from the three finest meshes (observed order)
    h = np.array([r["h"] for r in rows[-3:]])
    d = np.array([r["delta"] for r in rows[-3:]])
    ratio = h[0] / h[1]
    if np.isclose(h[1] / h[2], ratio):
        q = np.log((d[1] - d[0]) / (d[2] - d[1])) / np.log(ratio)
        d_inf = d[2] + (d[2] - d[1]) / (ratio**q - 1)
    else:
        q, d_inf = float("nan"), float("nan")
    return rows, {"order": float(q), "delta_extrapolated": float(d_inf)}


def reference_agreement():
    prob = V.heterogeneous_box()
    ref = ReferenceSolver().solve(prob)
    out = {}
    for method in ("amg", "jax-cg"):
        sol = JaxSolver(method=method, rtol=1e-13).solve(prob)
        out[method] = {name: {
            "u": float(np.linalg.norm(sol[name].u - ref[name].u) / np.linalg.norm(ref[name].u)),
            "stress": float(np.linalg.norm(sol[name].stress - ref[name].stress)
                            / np.linalg.norm(ref[name].stress)),
            "iterations": sol[name].info["iterations"]} for name in ref}
    return out


def patch_test():
    m = V.jittered_box(4, amplitude=0.3, seed=1)
    rng = np.random.default_rng(3)
    from craniofea.fem.constitutive import orthotropic_C, rotate_C
    from craniofea.model import LoadCase, Material, Problem

    C = rotate_C(orthotropic_C(12e9, 8e9, 20e9, 3e9, 4.5e9, 5e9, 0.35, 0.2, 0.25),
                 V.random_rotations(1, rng)[0])
    mat = Material(C=np.broadcast_to(C, (m.n_elems, 6, 6)).copy(), mass_density=np.ones(m.n_elems))
    G = rng.normal(size=(3, 3)) * 1e-3
    u_aff = m.nodes @ G.T + rng.normal(size=3) * 1e-4
    prob = Problem(m, mat, fixed_nodes=m.node_sets["boundary"], load_cases=[LoadCase("patch")])
    E = 0.5 * (G + G.T)
    eps = np.array([E[0, 0], E[1, 1], E[2, 2], 2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]])
    out = {}
    for method in ("amg", "jax-cg"):
        s = JaxSolver(method=method, rtol=1e-13).solve(prob, prescribed=u_aff)["patch"]
        out[method] = {"u": float(np.abs(s.u - u_aff).max() / np.abs(u_aff).max()),
                       "strain": float(np.abs(s.strain - eps).max() / np.abs(eps).max())}
    return out


def gradient_check():
    import jax
    import jax.numpy as jnp

    prob = V.heterogeneous_box((4, 3, 3), seed=7)
    rng = np.random.default_rng(11)
    w = jnp.asarray(rng.normal(size=(3, prob.mesh.n_nodes, 3)))
    out = {}
    for method in ("amg", "jax-cg"):
        fem = make_differentiable_solver(prob, method=method, rtol=1e-14)

        def L(s, fem=fem):
            r = fem(s)
            return 1e5 * jnp.sum(w * r["u"]) + 1e-3 * jnp.sum(r["sed"])

        s0 = jnp.asarray(rng.uniform(0.5, 1.5, prob.mesh.n_elems))
        g = jax.grad(L)(s0)
        d = jnp.asarray(rng.normal(size=s0.shape))
        h = 1e-4
        fd = (L(s0 + h * d) - L(s0 - h * d)) / (2 * h)
        out[method] = float(abs(g @ d - fd) / abs(fd))
    return out


def rate(rows, key):
    return V.observed_rate(np.array([r["h"] for r in rows]), np.array([r[key] for r in rows]))


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, which="both", labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _series(ax, h, e, color, label, marker="o", fit_last=None):
    h, e = np.asarray(h), np.asarray(e)
    ax.loglog(h, e, color=color, lw=2, marker=marker, ms=6, mec=SURFACE, mew=1.2, label=label)
    n = len(h) if fit_last is None else fit_last
    p = np.polyfit(np.log(h[-n:]), np.log(e[-n:]), 1)
    note = "" if fit_last is None else f"\n({n} finest)"
    ax.annotate(f"slope {p[0]:.2f}{note}", xy=(h[-1], e[-1]), xytext=(6, -2),
                textcoords="offset points", fontsize=9, color=INK2, va="center")


def figure(lame, column, cant, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor=SURFACE)
    titles = ["Lamé cylinder (plane strain)", "Hanging column (self-weight)",
              "Cantilever vs. Timoshenko"]
    b_a = 0.01
    for ax, rows, scale in ((axes[0], lame, b_a), (axes[1], column, 1.0)):
        h = [r["h"] / scale for r in rows]
        _series(ax, h, [r["u_l2"] for r in rows], BLUE, "displacement, L2")
        _series(ax, h, [r["stress_l2"] for r in rows], ORANGE, "stress, L2", marker="s")
    axes[0].set_xlabel("h / (b − a)", color=INK, fontsize=10)
    axes[1].set_xlabel("h / L", color=INK, fontsize=10)
    h = [r["h"] / 0.1 for r in cant]
    _series(axes[2], h, [abs(1 - r["ratio"]) for r in cant], AQUA, "tip deflection", marker="D",
            fit_last=3)
    axes[2].set_xlabel("h / H", color=INK, fontsize=10)
    for ax, t in zip(axes, titles, strict=True):
        _style(ax)
        ax.set_title(t, color=INK, fontsize=11, loc="left")
        ax.set_ylabel("relative error", color=INK, fontsize=10)
        ax.invert_xaxis()
        ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left")
        ax.margins(x=0.15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def lame_profile_figure(level: int, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = V.lame_cylinder(level)
    s = JaxSolver(rtol=1e-11).solve(c.problem)["pressure"]
    m = c.problem.mesh
    xc = m.nodes[m.tets].mean(axis=1)
    r, th = np.hypot(xc[:, 0], xc[:, 1]), np.arctan2(xc[:, 1], xc[:, 0])
    cs, sn = np.cos(th), np.sin(th)
    sx, sy, sxy = s.stress[:, 0], s.stress[:, 1], s.stress[:, 5]
    srr = sx * cs**2 + sy * sn**2 + 2 * sxy * sn * cs
    stt = sx * sn**2 + sy * cs**2 - 2 * sxy * sn * cs
    a, b, p = c.meta["a"], c.meta["b"], c.meta["p"]
    A = p * a**2 / (b**2 - a**2)
    rr = np.linspace(a, b, 200)

    fig, ax = plt.subplots(figsize=(6, 3.8), facecolor=SURFACE)
    ax.scatter(r * 1e3, stt / p, s=3, color=BLUE, alpha=0.35, lw=0, label="σθθ, elements")
    ax.scatter(r * 1e3, srr / p, s=3, color=ORANGE, alpha=0.35, lw=0, label="σrr, elements")
    ax.plot(rr * 1e3, A * (1 + b**2 / rr**2) / p, color=INK, lw=1.5, label="exact")
    ax.plot(rr * 1e3, A * (1 - b**2 / rr**2) / p, color=INK, lw=1.5)
    ax.annotate("σθθ", xy=(b * 1e3, A * 2 / p), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
    ax.annotate("σrr", xy=(b * 1e3, 0), xytext=(6, 0), textcoords="offset points",
                color=INK2, fontsize=9, va="center")
    _style(ax)
    ax.set_xlabel("radius r (mm)", color=INK, fontsize=10)
    ax.set_ylabel("stress / internal pressure", color=INK, fontsize=10)
    ax.set_title(f"Lamé cylinder, {m.n_elems:,} tets: element stresses vs exact",
                 color=INK, fontsize=11, loc="left")
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK2, markerscale=4)
    for hnd in leg.legend_handles:
        hnd.set_alpha(1)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="coarser levels")
    ap.add_argument("--json", type=Path, help="write all numbers to this file")
    args = ap.parse_args()

    lame_levels = (1, 2, 4, 8) if args.quick else (1, 2, 4, 8, 16, 32)
    col_levels = (1, 2, 4) if args.quick else (1, 2, 4, 8)
    cant_levels = (1, 2, 4) if args.quick else (1, 2, 4, 6, 9)

    results = {}
    results["lame"] = lame = convergence(V.lame_cylinder, lame_levels, "pressure")
    results["column"] = column = convergence(V.hanging_column, col_levels, "gravity")
    results["cantilever"], results["cantilever_extrapolation"] = cant, extr = \
        cantilever_study(cant_levels)
    results["reference"] = reference_agreement()
    results["patch"] = patch_test()
    results["gradient"] = gradient_check()

    for name, rows in (("Lame cylinder", lame), ("Hanging column", column)):
        print(f"\n{name}: rate u {rate(rows, 'u_l2'):.2f}, stress {rate(rows, 'stress_l2'):.2f}")
        print("| level | tets | dofs | h | L2 error u | L2 error stress | CG it |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['level']} | {r['n_elems']:,} | {r['n_dofs']:,} | {r['h']:.3e} | "
                  f"{r['u_l2']:.3e} | {r['stress_l2']:.3e} | {r['iterations']} |")
    print("\nCantilever (L/H = 10, nu = 0.3)")
    print("| level | tets | elements through depth | delta / delta_Timoshenko |")
    print("|---|---|---|---|")
    for r in cant:
        print(f"| {r['level']} | {r['n_elems']:,} | {r['level']} | {r['ratio']:.4f} |")
    print(f"Richardson: order {extr['order']:.2f}, extrapolated delta / delta_T = "
          f"{extr['delta_extrapolated'] / cant[0]['delta_timoshenko']:.4f}")
    print("\nReference-solver agreement (relative L2):")
    for method, cases in results["reference"].items():
        for case, v in cases.items():
            print(f"  {method:7s} {case:9s} u {v['u']:.1e} stress {v['stress']:.1e} "
                  f"(it {v['iterations']})")
    print("Patch test (max relative error):", json.dumps(results["patch"]))
    print("Adjoint gradient vs central FD (relative):", json.dumps(results["gradient"]))

    figure(lame, column, cant, FIG_DIR / "verification_convergence.png")
    lame_profile_figure(16 if args.quick else 32, FIG_DIR / "verification_lame_stress.png")
    print(f"\nfigures written to {FIG_DIR.relative_to(ROOT)}")
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

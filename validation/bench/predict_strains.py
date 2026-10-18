"""Order-of-magnitude strain predictions for the pig-palate bench test.

Runs WITHOUT modifying the craniofea package: it only imports it from ../../src.
Outputs: validation/bench/predicted_strains.json and a printed table.

Three estimates, all linear elastic, so strain scales exactly with F/E:

1. Analytic clamped-clamped strip (palatal shelf spanning between two potted
   alveolar processes), uniform load over the full span:  M_edge = q L^2/12,
   M_mid = q L^2/24 (per unit width), surface strain eps = 6 M / (E' t^2),
   E' = E/(1-nu^2) (plate, plane strain across the width).
2. FE "bench analog": craniofea box_mesh plate L (span, x) x B (length, y) x t,
   nodes within a potting band of each x edge fully fixed (rigid potting),
   pressure on a central oral-surface patch of ~10 cm^2.  Also a full-span
   uniform load case for comparison with (1), which verifies the tet4 mesh in
   bending.
3. FE on the repo's parametric adolescent maxilla template (human geometry, not
   pig), pressure on the `palate` set with (a) the repo's default `fixed` set and
   (b) bench-like fixation (fixed + occlusal nodes = potted alveolar processes).

Usage:  python validation/bench/predict_strains.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from craniofea.fem.constitutive import isotropic_C  # noqa: E402
from craniofea.fem.reference import ReferenceSolver  # noqa: E402
from craniofea.mesh.primitives import boundary_faces, box_mesh, face_normals_areas  # noqa: E402
from craniofea.model import LoadCase, Material, Problem  # noqa: E402

FORCES = [2.0, 20.0, 50.0, 100.0, 200.0]          # N, total resultant on the pad
OUT = Path(__file__).with_name("predicted_strains.json")


# --------------------------------------------------------------------------- helpers
def surface_principal(mesh, sol, faces):
    """In-plane principal strains (microstrain) of the elements owning `faces`."""
    tets = mesh.tets
    # map each face to its owning element
    key_t = {}
    for e, t in enumerate(tets):
        for f in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            key_t[tuple(sorted(t[list(f)]))] = e
    elems = np.array([key_t[tuple(sorted(f))] for f in faces])
    n, _ = face_normals_areas(mesh.nodes, faces)
    eps = sol.strain[elems]                      # [xx yy zz yz xz xy], engineering shear
    E = np.zeros((len(elems), 3, 3))
    E[:, 0, 0], E[:, 1, 1], E[:, 2, 2] = eps[:, 0], eps[:, 1], eps[:, 2]
    E[:, 1, 2] = E[:, 2, 1] = eps[:, 3] / 2
    E[:, 0, 2] = E[:, 2, 0] = eps[:, 4] / 2
    E[:, 0, 1] = E[:, 1, 0] = eps[:, 5] / 2
    # tangent basis
    a = np.where(np.abs(n[:, [0]]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t1 = np.cross(n, a)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    T = np.stack([t1, t2], axis=2)               # (F, 3, 2)
    E2 = np.einsum("fai,fab,fbj->fij", T, E, T)
    w = np.linalg.eigvalsh(E2)                   # ascending
    return w[:, 1] * 1e6, w[:, 0] * 1e6, elems


def summarize(e1, e2, label):
    mag = np.maximum(np.abs(e1), np.abs(e2))
    return {"case": label,
            "max_principal_p99_ue": float(np.percentile(e1, 99)),
            "min_principal_p1_ue": float(np.percentile(e2, 1)),
            "abs_max_ue": float(mag.max()),
            "abs_median_ue": float(np.median(mag)),
            "abs_p90_ue": float(np.percentile(mag, 90))}


# --------------------------------------------------------------------------- 1 analytic
def analytic(span_mm, t_mm, F, B_mm, E=12.5e9, nu=0.35):
    """Clamped-clamped strip, load F spread uniformly over span x B."""
    L, t, B = span_mm * 1e-3, t_mm * 1e-3, B_mm * 1e-3
    q = F / B / L                                # N/m per unit width... -> N/m^2 * width
    Ep = E / (1 - nu ** 2)
    M_edge, M_mid = q * L ** 2 / 12, q * L ** 2 / 24
    return {"edge_ue": 6 * M_edge / (Ep * t ** 2) * 1e6, "mid_ue": 6 * M_mid / (Ep * t ** 2) * 1e6}


# --------------------------------------------------------------------------- 2 bench analog
def bench_analog(span=35.0, length=50.0, t=4.0, pot=5.0, patch=(25.0, 40.0), E=12.5e9,
                 nu=0.35, h=1.0, nz=6):
    """Plate with rigid potting bands along both long (y) edges."""
    lx, ly = span + 2 * pot, length
    m = box_mesh(lx * 1e-3, ly * 1e-3, t * 1e-3, int(round(lx / h)), int(round(ly / h)), nz)
    x = m.nodes[:, 0] * 1e3
    fixed = np.flatnonzero((x <= pot + 1e-6) | (x >= lx - pot - 1e-6))
    faces = m.surface_sets["zmin"]                       # oral surface (z = 0), load pushes +z
    fc = m.nodes[faces].mean(axis=1) * 1e3
    in_patch = (np.abs(fc[:, 0] - lx / 2) <= patch[0] / 2) & (np.abs(fc[:, 1] - ly / 2) <= patch[1] / 2)
    in_span = (fc[:, 0] > pot) & (fc[:, 0] < lx - pot)
    _, area = face_normals_areas(m.nodes, faces)
    m.surface_sets["patch"] = faces[in_patch]
    m.surface_sets["span"] = faces[in_span]
    A_patch, A_span = area[in_patch].sum(), area[in_span].sum()
    mat = Material(C=isotropic_C(np.full(m.n_elems, E), np.full(m.n_elems, nu)),
                   mass_density=np.full(m.n_elems, 1850.0))
    F = 1.0                                              # unit load, scale later
    cases = [LoadCase("patch", pressures={"patch": F / A_patch}),
             LoadCase("span", pressures={"span": F / A_span})]
    t0 = time.perf_counter()
    sol = ReferenceSolver().solve(Problem(m, mat, fixed, cases))
    dt = time.perf_counter() - t0
    # gauge-reading surface: the oral (z=0) faces in the free span AND the nasal (z=t) faces
    res = {"geometry_mm": {"span": span, "length": length, "t": t, "pot_band": pot,
                           "patch": patch, "patch_area_mm2": A_patch * 1e6},
           "E_GPa": E / 1e9, "nodes": m.n_nodes, "tets": m.n_elems, "solve_s": dt}
    for cname in ("patch", "span"):
        s = sol[cname]
        e1, e2, el = surface_principal(m, s, faces[in_span])
        # gauge sites: oral surface, mid-span centre and 3 mm from the potting edge
        c = m.nodes[faces[in_span]].mean(axis=1) * 1e3
        mid = np.argmin(np.hypot(c[:, 0] - lx / 2, c[:, 1] - ly / 2))
        edge = np.argmin(np.hypot(c[:, 0] - (pot + 3.0), c[:, 1] - ly / 2))
        res[cname] = {**summarize(e1, e2, cname),
                      "oral_midspan_ue_per_N": [float(e1[mid]), float(e2[mid])],
                      "oral_3mm_from_pot_ue_per_N": [float(e1[edge]), float(e2[edge])],
                      "max_u_um_per_N": float(np.abs(s.u).max() * 1e6)}
    res["analytic_span_ue_per_N"] = analytic(span, t, 1.0, length, E, nu)
    return res


# --------------------------------------------------------------------------- 3 template
def template_runs():
    from craniofea.materials import LayerProps, build_material
    from craniofea.mesh.template import build_template_from_config

    mesh = build_template_from_config(str(REPO / "configs" / "mesh_template.yaml"))
    layers = {"cortical": LayerProps(15e9, 0.3, 3, 1850.0),
              "cancellous": LayerProps(1e9, 0.3, 2, 1000.0),
              "suture_mid": LayerProps(10e6, 0.45, 0, 1100.0)}
    mat = build_material(mesh, 1.0, layers)
    pal = mesh.surface_sets["palate"]
    _, a = face_normals_areas(mesh.nodes, pal)
    A = a.sum()
    case = LoadCase("palate_1N", pressures={"palate": 1.0 / A})
    out = {"palate_area_mm2": A * 1e6, "force_at_2kPa_N": 2000 * A,
           "nodes": mesh.n_nodes, "tets": mesh.n_elems}
    bcs = {"repo_fixed": mesh.node_sets["fixed"],
           "bench_potted": np.unique(np.concatenate([mesh.node_sets["fixed"],
                                                     np.unique(mesh.surface_sets["occlusal"])]))}
    sut = set(np.asarray(mesh.elem_sets["suture_mid"]).tolist())
    for name, fixed in bcs.items():
        t0 = time.perf_counter()
        s = ReferenceSolver().solve(Problem(mesh, mat, fixed, [case]))["palate_1N"]
        e1, e2, el = surface_principal(mesh, s, pal)
        bone = np.array([e not in sut for e in el])
        out[name] = {**summarize(e1[bone], e2[bone], name + " (bone, excl. suture)"),
                     "suture_abs_max_ue_per_N": float(np.maximum(np.abs(e1[~bone]),
                                                                 np.abs(e2[~bone])).max()),
                     "solve_s": time.perf_counter() - t0}
    return out


def main():
    res = {"note": "all strains in microstrain PER NEWTON of total pad force; linear, scale by F",
           "forces_N": FORCES}
    res["analytic"] = {f"span{L}_t{t}": analytic(L, t, 1.0, 40.0)
                       for L in (25.0, 35.0) for t in (2.0, 3.0, 4.0, 6.0)}
    res["bench_analog"] = [bench_analog(t=t) for t in (3.0, 4.0, 6.0)]
    # mesh check: refine the 4 mm plate
    res["bench_analog_refined_t4"] = bench_analog(t=4.0, h=0.7, nz=8)
    res["template"] = template_runs()
    OUT.write_text(json.dumps(res, indent=1))

    print("Analytic clamped strip (ue per N, 40 mm long, E=12.5 GPa):")
    for k, v in res["analytic"].items():
        print(f"  {k:14s} edge {v['edge_ue']:7.1f}  mid {v['mid_ue']:7.1f}")
    print("\nFE bench analog (ue per N):")
    for b in res["bench_analog"] + [res["bench_analog_refined_t4"]]:
        g = b["geometry_mm"]
        print(f"  t={g['t']} mm ({b['tets']} tets): patch absmax {b['patch']['abs_max_ue']:.1f}, "
              f"midspan {b['patch']['oral_midspan_ue_per_N']}, near-pot "
              f"{b['patch']['oral_3mm_from_pot_ue_per_N']}; span absmax "
              f"{b['span']['abs_max_ue']:.1f} vs analytic edge {b['analytic_span_ue_per_N']['edge_ue']:.1f}")
    tp = res["template"]
    print(f"\nTemplate (palate {tp['palate_area_mm2']:.0f} mm2, 2 kPa = {tp['force_at_2kPa_N']:.2f} N):")
    for k in ("repo_fixed", "bench_potted"):
        v = tp[k]
        print(f"  {k}: absmax {v['abs_max_ue']:.2f}, p90 {v['abs_p90_ue']:.2f}, "
              f"median {v['abs_median_ue']:.2f} ue/N; suture max {v['suture_abs_max_ue_per_N']:.1f}")
    print("\nScaled to forces (FE bench analog t=4 mm, oral mid-span |max|):")
    b4 = res["bench_analog"][1]["patch"]
    mid = max(abs(x) for x in b4["oral_midspan_ue_per_N"])
    for F in FORCES:
        print(f"  F={F:6.1f} N: mid-span {mid*F:8.1f} ue, peak {b4['abs_max_ue']*F:8.1f} ue")


if __name__ == "__main__":
    main()

"""Specimen-specific FE model of a potted pig-palate bench specimen + virtual gauges.

Uses the craniofea package read-only (data model, materials, solvers, remesher).
Nothing in src/ is modified.

Pipeline (see docs/validation/BENCH_TEST_PROTOCOL.md, section 6):

  CT -> 3D Slicer segmentation -> bone.stl (+ pot_left.stl, pot_right.stl)
     -> tet mesh  (fTetWild via craniofea.mesh.remesh.remesh_surface_ftetwild,
                   or any gmsh/meshio tet mesh)
     -> labels    (fixed = bone nodes inside the potting volumes,
                   pad   = oral boundary faces inside the pressure-cup aperture)
     -> material  (M1: craniofea layered default; M2: CT-density mapped; M0: homogeneous)
     -> solve     (ReferenceSolver for < ~150k nodes, CalculiX for larger meshes)
     -> virtual gauges: strain tensor averaged over the surface elements under the
        grid footprint, projected on the grid axis  eps_g = d . eps . d
     -> predicted microstrain PER kPa of cup pressure (and per N of measured force)

Spec file (JSON), all coordinates in CT millimetres:
{
  "mesh": "path/to/bone_tets.msh"            (or "surface": "bone.stl", "size_mm": 0.6),
  "pots": [ {"type": "halfspace", "point": [x,y,z], "normal": [nx,ny,nz]} , ...],
           # bone nodes on the +normal side of any halfspace are fixed (rigid potting);
           # alternatively {"type": "box", "min": [...], "max": [...]}
  "pad": {"center": [x,y,z], "normal": [nx,ny,nz]  (pointing from bone toward cup),
          "u_axis": [ux,uy,uz], "half_u": 12.5, "half_v": 20.0, "rim_mm": 1.0},
  "material": {"model": "M0", "E_GPa": 12.5, "nu": 0.35},
  "gauges": [ {"id": "S1-R1-a", "center": [x,y,z], "dir": [dx,dy,dz], "grid_mm": 1.57}, ...]
}
Usage:
  python validation/bench/bench_fe_model.py spec.json out.csv
  python validation/bench/bench_fe_model.py --selftest      (synthetic plate check)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from craniofea.fem.constitutive import isotropic_C  # noqa: E402
from craniofea.mesh.primitives import boundary_faces, face_normals_areas, orient_tets  # noqa: E402
from craniofea.model import LoadCase, Material, Mesh, Problem  # noqa: E402

MM = 1e-3


# --------------------------------------------------------------------------- mesh
def load_mesh(spec: dict) -> Mesh:
    if "mesh" in spec:
        from craniofea.mesh.ct import read_gmsh_tets
        pts, tets, _ = read_gmsh_tets(spec["mesh"])
    else:
        import meshio
        from craniofea.mesh.remesh import remesh_surface_ftetwild
        s = meshio.read(spec["surface"])
        tri = np.concatenate([c.data for c in s.cells if c.type == "triangle"])
        pts, tets = remesh_surface_ftetwild(s.points, tri, size=spec.get("size_mm", 0.6))
    nodes = np.asarray(pts, float) * MM
    tets = orient_tets(nodes, np.asarray(tets, np.int64))
    faces, _ = boundary_faces(nodes, tets)
    return Mesh(nodes=nodes, tets=tets, surface_sets={"all": faces},
                meta={"source": "bench_specimen"})


# --------------------------------------------------------------------------- labels
def fixed_nodes(mesh: Mesh, pots: list[dict]) -> np.ndarray:
    x = mesh.nodes / MM
    sel = np.zeros(len(x), bool)
    for p in pots:
        if p["type"] == "halfspace":
            n = np.asarray(p["normal"], float)
            n /= np.linalg.norm(n)
            sel |= (x - np.asarray(p["point"], float)) @ n >= 0
        elif p["type"] == "box":
            sel |= np.all((x >= p["min"]) & (x <= p["max"]), axis=1)
        else:
            raise ValueError(p["type"])
    return np.flatnonzero(sel)


def pad_faces(mesh: Mesh, pad: dict) -> np.ndarray:
    """Boundary faces facing the cup and inside its aperture (minus the rim band)."""
    faces = mesh.surface_sets["all"]
    n_f, _ = face_normals_areas(mesh.nodes, faces)
    c = mesh.nodes[faces].mean(axis=1) / MM
    n = np.asarray(pad["normal"], float)
    n /= np.linalg.norm(n)
    u = np.asarray(pad["u_axis"], float)
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    d = c - np.asarray(pad["center"], float)
    rim = pad.get("rim_mm", 0.0)
    inside = (np.abs(d @ u) <= pad["half_u"] - rim) & (np.abs(d @ v) <= pad["half_v"] - rim)
    facing = n_f @ n > np.cos(np.deg2rad(pad.get("max_angle_deg", 75.0)))
    # keep only the surface nearest the cup (exclude the nasal floor under it)
    near = (d @ n) > np.percentile(d[inside & facing] @ n, 5) - 1.0 if np.any(inside & facing) else inside
    return faces[inside & facing & near]


# --------------------------------------------------------------------------- material
def material(mesh: Mesh, mcfg: dict) -> Material:
    M = mesh.n_elems
    model = mcfg.get("model", "M0")
    if model == "M0":
        E = np.full(M, mcfg.get("E_GPa", 12.5) * 1e9)
        nu = np.full(M, mcfg.get("nu", 0.35))
        return Material(C=isotropic_C(E, nu), mass_density=np.full(M, 1850.0), E=E, nu=nu)
    if model == "M2":
        # per-element apparent density rho_app (g/cm^3) from the calibrated CT (HA phantom),
        # supplied as a .npy aligned with the tets; E = a * rho^b (pre-registered a, b)
        rho = np.load(mcfg["rho_npy"])
        E = mcfg.get("a_GPa", 10.5) * 1e9 * np.clip(rho, 0.05, None) ** mcfg.get("b", 2.29)
        nu = np.full(M, mcfg.get("nu", 0.3))
        return Material(C=isotropic_C(E, nu), mass_density=1000 * rho, E=E, nu=nu)
    if model == "M1":
        # craniofea layered default. TODO (not implemented, not tested): the mesh must
        # carry elem sets "cortical" (elements within 1.5 mm of the surface),
        # "cancellous" (the rest) and "suture_mid" (elements within ~1 mm of the
        # midpalatal suture plane picked on CT) before build_material is called.
        from craniofea.materials import LayerProps, build_material
        layers = {"cortical": LayerProps(15e9, 0.3, 3, 1850.0),
                  "cancellous": LayerProps(1e9, 0.3, 2, 1000.0),
                  "suture_mid": LayerProps(10e6, 0.45, 0, 1100.0)}
        return build_material(mesh, 1.0, layers)
    raise ValueError(model)


# --------------------------------------------------------------------------- gauges
def virtual_gauges(mesh: Mesh, strain: np.ndarray, gauges: list[dict]) -> list[dict]:
    """Grid-axis normal strain averaged over surface-adjacent elements under each grid."""
    faces = mesh.surface_sets["all"]
    fc = mesh.nodes[faces].mean(axis=1) / MM
    n_f, _ = face_normals_areas(mesh.nodes, faces)
    cen = mesh.nodes[mesh.tets].mean(axis=1) / MM
    out = []
    for g in gauges:
        p = np.asarray(g["center"], float)
        k = int(np.argmin(np.linalg.norm(fc - p, axis=1)))
        n = n_f[k]
        d = np.asarray(g["dir"], float)
        d = d - (d @ n) * n
        d /= np.linalg.norm(d)
        w = np.cross(n, d)
        r = cen - fc[k]
        half = g.get("grid_mm", 1.57) / 2
        depth = g.get("depth_mm", 0.6)
        sel = (np.abs(r @ d) <= half) & (np.abs(r @ w) <= half) & (-(r @ n) <= depth) & ((r @ n) <= 0.2)
        if sel.sum() == 0:
            sel = np.zeros(len(cen), bool)
            sel[np.argsort(np.linalg.norm(r, axis=1))[:4]] = True
        e = strain[sel]
        T = np.array([d[0] ** 2, d[1] ** 2, d[2] ** 2, d[1] * d[2], d[0] * d[2], d[0] * d[1]])
        eg = e @ T                                  # engineering shear in Voigt -> exact
        out.append({"id": g["id"], "n_elems": int(sel.sum()),
                    "eps_ue": float(eg.mean() * 1e6), "eps_sd_ue": float(eg.std() * 1e6),
                    "surface_offset_mm": float(np.linalg.norm(fc[k] - p))})
    return out


def solve(mesh, mat, fixed, pad, p_kpa=1.0, solver="reference"):
    from craniofea.solvers import get_solver
    mesh.surface_sets["pad"] = pad
    _, a = face_normals_areas(mesh.nodes, pad)
    case = LoadCase("cup_1kPa", pressures={"pad": p_kpa * 1e3})
    sol = get_solver(solver).solve(Problem(mesh, mat, fixed, [case]))["cup_1kPa"]
    return sol, float(a.sum() / MM ** 2)


def run(spec_path: str, out_csv: str, solver: str = "reference"):
    spec = json.loads(Path(spec_path).read_text())
    t0 = time.time()
    mesh = load_mesh(spec)
    fixed = fixed_nodes(mesh, spec["pots"])
    pad = pad_faces(mesh, spec["pad"])
    mat = material(mesh, spec.get("material", {}))
    sol, area = solve(mesh, mat, fixed, pad, solver=solver)
    vg = virtual_gauges(mesh, sol.strain, spec["gauges"])
    force_per_kpa = area * 1e-3                     # mm^2 * kPa -> N  (1 kPa * 1 mm^2 = 1e-3 N)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gauge_id", "pred_ue_per_kPa", "pred_ue_per_N", "n_elems", "sd_ue_per_kPa",
                    "surface_offset_mm"])
        for g in vg:
            w.writerow([g["id"], f"{g['eps_ue']:.5f}", f"{g['eps_ue'] / force_per_kpa:.5f}",
                        g["n_elems"], f"{g['eps_sd_ue']:.5f}", f"{g['surface_offset_mm']:.3f}"])
    print(f"{mesh.n_nodes} nodes, {mesh.n_elems} tets, {len(fixed)} fixed nodes, pad "
          f"{area:.1f} mm^2 ({force_per_kpa:.3f} N/kPa); {time.time() - t0:.1f} s -> {out_csv}")


# --------------------------------------------------------------------------- self test
def selftest():
    """Write a synthetic potted plate as .msh, run the full pipeline, compare with the
    direct box computation of predict_pad_sensitivity (N_mid ~0.77, N_edge ~-0.96 ue/N)."""
    import meshio

    from craniofea.mesh.primitives import box_mesh
    work = Path(__file__).with_name("selftest")
    work.mkdir(exist_ok=True)
    m = box_mesh(45e-3, 50e-3, 4e-3, 45, 50, 6)
    meshio.write(work / "plate.msh", meshio.Mesh(m.nodes / MM, [("tetra", m.tets)]),
                 file_format="gmsh22", binary=False)
    spec = {"mesh": str(work / "plate.msh"),
            "pots": [{"type": "halfspace", "point": [5, 0, 0], "normal": [-1, 0, 0]},
                     {"type": "halfspace", "point": [40, 0, 0], "normal": [1, 0, 0]}],
            "pad": {"center": [22.5, 25, 0], "normal": [0, 0, -1], "u_axis": [1, 0, 0],
                    "half_u": 12.5, "half_v": 20.0, "rim_mm": 0.0},
            "material": {"model": "M0", "E_GPa": 12.5, "nu": 0.35},
            "gauges": [{"id": "N_mid", "center": [25.5, 25, 4], "dir": [1, 0, 0]},
                       {"id": "N_edge", "center": [8, 25, 4], "dir": [1, 0, 0]},
                       {"id": "O_edge", "center": [8, 25, 0], "dir": [1, 0, 0]},
                       {"id": "N_mid_long", "center": [25.5, 25, 4], "dir": [0, 1, 0]}]}
    (work / "spec.json").write_text(json.dumps(spec, indent=1))
    run(str(work / "spec.json"), str(work / "pred.csv"))
    print((work / "pred.csv").read_text())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--solver", default="reference")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a.spec, a.out, a.solver)

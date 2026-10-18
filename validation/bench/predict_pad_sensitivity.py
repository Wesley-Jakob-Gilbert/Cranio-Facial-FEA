"""Sensitivity of predicted gauge strains to pad pressure distribution, pad
misalignment, potting compliance and plate thickness (bench analog plate).

Gauge sites (all outside the pad footprint or on the opposite face):
  N_mid : nasal (top, z = t) surface, mid-span, 3 mm lateral of centre line
  N_edge: nasal surface, 3 mm from the potting edge
  O_edge: oral (loaded, z = 0) surface, 3 mm from the potting edge (outside pad)
Strain = normal strain along the transverse (x) axis, ue per N, averaged over
the elements under a 1.6 x 1.6 mm virtual grid.

Usage: python validation/bench/predict_pad_sensitivity.py   (~5 min on 2 CPUs)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from craniofea.fem.constitutive import isotropic_C  # noqa: E402
from craniofea.fem.reference import ReferenceSolver  # noqa: E402
from craniofea.mesh.primitives import box_mesh, face_normals_areas  # noqa: E402
from craniofea.model import LoadCase, Material, Problem  # noqa: E402

OUT = Path(__file__).with_name("pad_sensitivity.json")


def run(t=4.0, span=35.0, length=50.0, pot=5.0, patch=(25.0, 40.0), dist="uniform",
        offset=(0.0, 0.0), pot_mode="rigid", E=12.5e9, nu=0.35, h=1.0, nz=6):
    lx, ly = span + 2 * pot, length
    m = box_mesh(lx * 1e-3, ly * 1e-3, t * 1e-3, int(round(lx / h)), int(round(ly / h)), nz)
    x, z = m.nodes[:, 0] * 1e3, m.nodes[:, 2] * 1e3
    in_pot = (x <= pot + 1e-6) | (x >= lx - pot - 1e-6)
    if pot_mode == "rigid":
        fixed = np.flatnonzero(in_pot)
    elif pot_mode == "outer_only":           # only the outer 2 mm band of each pot fixed
        fixed = np.flatnonzero((x <= pot - 3 + 1e-6) | (x >= lx - pot + 3 - 1e-6))
    else:
        raise ValueError(pot_mode)
    faces = m.surface_sets["zmin"]
    fc = m.nodes[faces].mean(axis=1) * 1e3
    cx, cy = lx / 2 + offset[0], ly / 2 + offset[1]
    u, v = (fc[:, 0] - cx) / (patch[0] / 2), (fc[:, 1] - cy) / (patch[1] / 2)
    sel = (np.abs(u) <= 1) & (np.abs(v) <= 1)
    _, a = face_normals_areas(m.nodes, faces[sel])
    if dist == "uniform":
        w = np.ones(sel.sum())
    elif dist == "parabolic":                 # peak at centre, zero at the rim
        w = (1 - u[sel] ** 2) * (1 - v[sel] ** 2)
    elif dist == "rim":                       # stiff-rim pad: load concentrated near rim
        w = np.maximum(np.abs(u[sel]), np.abs(v[sel])) ** 4
    else:
        raise ValueError(dist)
    p = w / (w * a).sum()                     # resultant = 1 N
    m.surface_sets["pad"] = faces[sel]
    mat = Material(C=isotropic_C(np.full(m.n_elems, E), np.full(m.n_elems, nu)),
                   mass_density=np.full(m.n_elems, 1850.0))
    s = ReferenceSolver().solve(Problem(m, mat, fixed, [LoadCase("pad", pressures={"pad": p})]))["pad"]
    cen = m.nodes[m.tets].mean(axis=1) * 1e3
    dz = t / nz

    def gauge(xg, yg, top):
        zsel = cen[:, 2] > t - dz if top else cen[:, 2] < dz
        g = zsel & (np.abs(cen[:, 0] - xg) <= 0.8) & (np.abs(cen[:, 1] - yg) <= 0.8)
        return float(s.strain[g, 0].mean() * 1e6)

    return {"N_mid": gauge(lx / 2 + 3, ly / 2, True),
            "N_edge": gauge(pot + 3, ly / 2, True),
            "O_edge": gauge(pot + 3, ly / 2, False)}


def main():
    cases = {
        "base_t4_uniform": {},
        "parabolic": {"dist": "parabolic"},
        "rim": {"dist": "rim"},
        "offset_x+3mm": {"offset": (3.0, 0.0)},
        "offset_y+5mm": {"offset": (0.0, 5.0)},
        "pot_outer_only": {"pot_mode": "outer_only"},
        "t3": {"t": 3.0},
        "t5": {"t": 5.0},
        "span30": {"span": 30.0},
        "span40": {"span": 40.0},
    }
    res = {}
    for k, kw in cases.items():
        res[k] = run(**kw)
        print(k, {a: round(b, 3) for a, b in res[k].items()}, flush=True)
    base = res["base_t4_uniform"]
    rel = {k: {g: (v[g] / base[g] - 1) * 100 for g in base} for k, v in res.items()}
    OUT.write_text(json.dumps({"ue_per_N": res, "percent_vs_base": rel}, indent=1))
    print("\n% change vs base:")
    for k, v in rel.items():
        print(f"  {k:16s} " + "  ".join(f"{g} {x:+6.1f}%" for g, x in v.items()))


if __name__ == "__main__":
    main()

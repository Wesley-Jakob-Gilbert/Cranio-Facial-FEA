"""Element quality of linear tetrahedra.

Metrics (all equal to 1, or 70.53 deg, for the regular tetrahedron):

- radius ratio  rho = 3 r_in / R_circ, in (0, 1]
- minimum / maximum dihedral angle, deg (regular: 70.53 deg)
- aspect ratio  L_max / (2 sqrt(6) r_in), >= 1

Linear tets lose accuracy and conditioning as rho -> 0; the minimum dihedral
angle detects slivers that edge-length measures miss (Shewchuk 2002, "What
is a good linear finite element?"). Typical acceptance limits for structural
analysis are rho > 0.1-0.2 and dihedral angles > 5-10 deg.
"""
from __future__ import annotations

import numpy as np

from craniofea.mesh.primitives import TET_FACES, tet_signed_volumes
from craniofea.model import Mesh

MM = 1e-3


def tet_quality(nodes: np.ndarray, tets: np.ndarray) -> dict[str, np.ndarray]:
    """Per-element volume, radius ratio, dihedral extremes, aspect ratio and edge lengths."""
    x = nodes[tets]
    vol = tet_signed_volumes(nodes, tets)
    av = np.abs(vol)
    a, b, c = x[:, 1] - x[:, 0], x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]

    f = x[:, TET_FACES]                                            # (M, 4, 3, 3)
    fn = np.cross(f[:, :, 1] - f[:, :, 0], f[:, :, 2] - f[:, :, 0])
    farea = 0.5 * np.linalg.norm(fn, axis=2)
    fn = fn / np.maximum(2 * farea[..., None], 1e-300)
    fn *= np.sign(vol)[:, None, None]                             # outward for either sign
    r_in = 3.0 * av / np.maximum(farea.sum(axis=1), 1e-300)
    num = (np.einsum("ij,ij->i", a, a)[:, None] * np.cross(b, c)
           + np.einsum("ij,ij->i", b, b)[:, None] * np.cross(c, a)
           + np.einsum("ij,ij->i", c, c)[:, None] * np.cross(a, b))
    r_circ = np.linalg.norm(num, axis=1) / np.maximum(12.0 * av, 1e-300)

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    cosang = np.stack([np.einsum("ij,ij->i", fn[:, i], fn[:, j]) for i, j in pairs], axis=1)
    dihedral = 180.0 - np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

    edges = np.stack([np.linalg.norm(x[:, i] - x[:, j], axis=1) for i, j in pairs], axis=1)
    return {
        "volume": vol,
        "radius_ratio": 3.0 * r_in / np.maximum(r_circ, 1e-300),
        "min_dihedral": dihedral.min(axis=1),
        "max_dihedral": dihedral.max(axis=1),
        "aspect_ratio": edges.max(axis=1) / np.maximum(2.0 * np.sqrt(6.0) * r_in, 1e-300),
        "edge_mean": edges.mean(axis=1),
    }


def quality_report(mesh: Mesh, chunk: int = 200_000) -> dict[str, float]:
    """Summary statistics of `tet_quality` (lengths in mm, volumes in mm^3)."""
    parts = [tet_quality(mesh.nodes, mesh.tets[s:s + chunk])
             for s in range(0, mesh.n_elems, chunk)]
    q = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rho, dmin, ar = q["radius_ratio"], q["min_dihedral"], q["aspect_ratio"]
    v = q["volume"] / MM**3
    return {
        "n_nodes": mesh.n_nodes,
        "n_tets": mesh.n_elems,
        "volume_mm3": float(v.sum()),
        "negative_volumes": int((v <= 0).sum()),
        "edge_mean_mm": float(q["edge_mean"].mean() / MM),
        "edge_p5_mm": float(np.percentile(q["edge_mean"], 5) / MM),
        "edge_p95_mm": float(np.percentile(q["edge_mean"], 95) / MM),
        "tet_volume_min_mm3": float(v.min()),
        "tet_volume_median_mm3": float(np.median(v)),
        "radius_ratio_min": float(rho.min()),
        "radius_ratio_p1": float(np.percentile(rho, 1)),
        "radius_ratio_median": float(np.median(rho)),
        "min_dihedral_min_deg": float(dmin.min()),
        "min_dihedral_p1_deg": float(np.percentile(dmin, 1)),
        "min_dihedral_median_deg": float(np.median(dmin)),
        "max_dihedral_max_deg": float(q["max_dihedral"].max()),
        "aspect_ratio_median": float(np.median(ar)),
        "aspect_ratio_p99": float(np.percentile(ar, 99)),
        "aspect_ratio_max": float(ar.max()),
        "frac_radius_ratio_below_0.1": float((rho < 0.1).mean()),
        "frac_min_dihedral_below_10deg": float((dmin < 10.0).mean()),
    }


def format_report(rows: dict[str, dict[str, float]]) -> str:
    """Markdown table with one column per mesh."""
    names = list(rows)
    keys = list(next(iter(rows.values())))
    out = ["| metric | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for k in keys:
        vals = []
        for n in names:
            v = rows[n][k]
            vals.append(f"{v:d}" if isinstance(v, int) else
                        (f"{v:.3g}" if abs(v) < 1e4 else f"{v:.0f}"))
        out.append(f"| {k} | " + " | ".join(vals) + " |")
    return "\n".join(out)

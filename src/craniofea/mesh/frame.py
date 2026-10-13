"""Automatic anatomical frame of a maxilla mesh.

The frame (``lr`` patient left -> right, ``ap`` posterior -> anterior, ``si``
inferior -> superior) is recovered from shape alone, in three steps that each
rest on a robust anatomical property of the maxilla:

1. **si axis (unsigned).** The maxilla, palate and dental arch form a flat,
   plate-like body: its smallest principal axis of inertia is the normal to
   the palatal/occlusal plane. (This is the bone's own plane, which is tilted
   by a few degrees relative to the Frankfort horizontal.)
2. **lr axis (unsigned).** The maxilla is bilaterally symmetric about the
   midsagittal plane. Within the principal plane we scan the plane normal
   and keep the one whose reflection best maps the shape onto itself
   (mean nearest-neighbour residual).
3. **Signs.** The dental arch is a U that is *closed anteriorly* (incisors on
   the midline) and *open posteriorly* (two tuberosities, nothing on the
   midline), so the end of the ap axis whose extreme slice lies closest to the
   midline is anterior. The alveolar processes hang *below* the palatal
   vault, so the lateral rim of the bone protrudes beyond the central
   (palatal) region much further on the inferior side than on the superior
   side.
   Finally ``lr = ap x si`` makes the frame right-handed (patient RAS), which
   is correct for any non-mirrored scanner/patient coordinate system such as
   DICOM LPS.

Any axis can be overridden from configuration (`frame_from_override`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from craniofea.mesh.primitives import tet_signed_volumes
from craniofea.model import AnatomicalFrame, Mesh

_AXES = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}


@dataclass
class FrameEstimate:
    frame: AnatomicalFrame
    diagnostics: dict = field(default_factory=dict)


def _weighted_sample(points: np.ndarray, weights: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = rng.choice(len(points), size=n, replace=False, p=weights / weights.sum())
    return points[idx]


def _symmetry_residual(x: np.ndarray, tree: cKDTree, normal: np.ndarray) -> float:
    """Mean distance from the mirror image (about x.normal = 0) back to the shape."""
    xr = x - 2.0 * np.outer(x @ normal, normal)
    d, _ = tree.query(xr, k=1)
    return float(np.mean(d))


def sample_volume(nodes: np.ndarray, tets: np.ndarray, n: int,
                  rng: np.random.Generator) -> np.ndarray:
    """`n` points uniformly distributed in the volume of a tet mesh."""
    vol = np.abs(tet_signed_volumes(nodes, tets))
    e = rng.choice(len(tets), size=n, p=vol / vol.sum())
    b = -np.log(rng.random((n, 4)))          # Dirichlet(1,1,1,1): uniform in a simplex
    b /= b.sum(axis=1, keepdims=True)
    return np.einsum("pk,pki->pi", b, nodes[tets[e]])


def estimate_frame(points: np.ndarray, weights: np.ndarray | None = None,
                   n_sample: int = 30_000, seed: int = 0) -> FrameEstimate:
    """Estimate the anatomical frame from a (volume-weighted) point cloud of the bone.

    `points` should sample the bone volume uniformly (see `sample_volume`);
    alternatively pass element centroids with element volumes as `weights`.
    """
    points = np.asarray(points, dtype=float)
    w = np.ones(len(points)) if weights is None else np.abs(np.asarray(weights, float))
    origin = (w[:, None] * points).sum(0) / w.sum()
    x0 = points - origin
    cov = (w[:, None] * x0).T @ x0 / w.sum()
    evals, evecs = np.linalg.eigh(cov)                    # ascending
    si_axis, e_mid, e_big = evecs[:, 0], evecs[:, 1], evecs[:, 2]

    rng = np.random.default_rng(seed)
    xs = _weighted_sample(x0, w, n_sample, rng)
    tree = cKDTree(xs)

    def normal(theta):
        return np.cos(theta) * e_big + np.sin(theta) * e_mid

    # Coarse scan on a subsample, refinement on the full sample.
    xc = xs[: min(len(xs), 8000)]
    tree_c = cKDTree(xc)
    thetas = np.deg2rad(np.arange(0.0, 180.0, 3.0))
    res = np.array([_symmetry_residual(xc, tree_c, normal(t)) for t in thetas])
    t_best = thetas[np.argmin(res)]
    fine = t_best + np.deg2rad(np.arange(-3.0, 3.01, 0.25))
    res_f = np.array([_symmetry_residual(xs, tree, normal(t)) for t in fine])
    t_best = fine[np.argmin(res_f)]
    lr_axis = normal(t_best)
    sym_best = float(res_f.min())
    sym_ortho = _symmetry_residual(xs, tree, normal(t_best + np.pi / 2))

    ap_axis = np.cross(si_axis, lr_axis)
    ap_axis /= np.linalg.norm(ap_axis)

    # ap sign: the anterior end of the arch is closed (bone on the midline).
    t = x0 @ ap_axis
    r = np.abs(x0 @ lr_axis)
    lo, hi = np.percentile(t, [1, 99])
    band = 0.12 * (hi - lo)
    r_pos = float(np.median(r[t > hi - band]))
    r_neg = float(np.median(r[t < lo + band]))
    if r_neg < r_pos:
        ap_axis = -ap_axis
        r_pos, r_neg = r_neg, r_pos

    # si sign: the alveolar crest (lateral rim) is the most inferior bone.
    h = x0 @ si_axis
    half_width = np.percentile(r, 98)
    t = x0 @ ap_axis
    lo, hi = np.percentile(t, [2, 98])
    central = (r < 0.3 * half_width) & (t > lo + 0.25 * (hi - lo)) & (t < hi - 0.25 * (hi - lo))
    rim = r > 0.65 * half_width
    if central.sum() < 10 or rim.sum() < 10:
        raise ValueError("cannot separate palatal and alveolar regions for the si sign test")
    # How far the rim protrudes beyond the central region on either side.
    rim_lo, rim_hi = np.percentile(h[rim], [5, 95])
    cen_lo, cen_hi = np.percentile(h[central], [5, 95])
    below, above = float(cen_lo - rim_lo), float(rim_hi - cen_hi)
    si = si_axis if below > above else -si_axis
    lr = np.cross(ap_axis, si)

    diag = {
        "principal_variances": evals.tolist(),
        "flatness": float(evals[0] / evals[1]),
        "symmetry_residual": sym_best,
        "symmetry_residual_orthogonal": sym_ortho,
        "midline_spread_anterior": r_pos,
        "midline_spread_posterior": r_neg,
        "alveolar_protrusion": max(below, above),
        "opposite_protrusion": min(below, above),
    }
    return FrameEstimate(AnatomicalFrame(origin=origin, lr=lr, ap=ap_axis, si=si), diag)


def parse_axis(spec) -> np.ndarray:
    """'+x', '-y', 'z' or a 3-vector -> unit vector."""
    if isinstance(spec, str):
        s = spec.strip().lower()
        sign = -1.0 if s.startswith("-") else 1.0
        return sign * _AXES[s.lstrip("+-")]
    v = np.asarray(spec, dtype=float)
    return v / np.linalg.norm(v)


def frame_from_override(override: dict, origin: np.ndarray) -> AnatomicalFrame:
    """Frame from configuration, e.g. ``{"ap": "-y", "si": "+z"}`` (lr = ap x si).

    Any two of lr/ap/si determine the third (right-handed); if all three are
    given they must be orthonormal and right-handed.
    """
    ax = {k: parse_axis(override[k]) for k in ("lr", "ap", "si") if override.get(k) is not None}
    if len(ax) < 2:
        raise ValueError("frame override needs at least two of lr/ap/si")
    if "lr" not in ax:
        ax["lr"] = np.cross(ax["ap"], ax["si"])
    elif "ap" not in ax:
        ax["ap"] = np.cross(ax["si"], ax["lr"])
    elif "si" not in ax:
        ax["si"] = np.cross(ax["lr"], ax["ap"])
    R = np.stack([ax["lr"], ax["ap"], ax["si"]])
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
        raise ValueError("frame override must be orthonormal and right-handed")
    o = np.asarray(override.get("origin", origin), dtype=float)
    return AnatomicalFrame(origin=o, lr=ax["lr"], ap=ax["ap"], si=ax["si"])


def detect_frame(mesh: Mesh, override: dict | None = None, n_sample: int = 30_000,
                 seed: int = 0) -> AnatomicalFrame:
    """Anatomical frame of `mesh`: automatic unless `override` gives the axes.

    The automatic estimate uses points sampled uniformly in the volume, so it
    does not depend on the mesh density. Its diagnostics are stored in
    ``mesh.meta["frame"]``.
    When an override is given, the automatic estimate is still computed and
    the angle between the two is recorded (a large angle flags a wrong
    override or an unusual geometry).
    """
    pts = sample_volume(mesh.nodes, mesh.tets, n_sample, np.random.default_rng(seed))
    est = estimate_frame(pts, n_sample=n_sample, seed=seed)
    info = {"method": "auto", **est.diagnostics}
    frame = est.frame
    if override:
        frame = frame_from_override(override, est.frame.origin)
        info["method"] = "override"
        info["override"] = {k: (v if isinstance(v, str) else list(map(float, v)))
                            for k, v in override.items()}
        info["auto_vs_override_deg"] = {
            k: float(np.degrees(np.arccos(np.clip(getattr(frame, k) @ getattr(est.frame, k),
                                                  -1, 1))))
            for k in ("lr", "ap", "si")}
    mesh.meta["frame"] = info
    return frame

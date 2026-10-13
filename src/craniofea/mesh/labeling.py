"""Anatomical labeling of maxilla meshes (canonical sets of docs/ARCHITECTURE.md).

All parameters are lengths in millimetres or angles in degrees, collected
in `LabelConfig` (defaults below, overridden from ``configs/mesh_*.yaml``).
Geometric tests are done in the anatomical frame (lr, ap, si), so the
labeling is independent of scanner orientation.

Surface sets
------------
``fixed``
    Where the segmented maxilla would continue into the skull. A CBCT
    segmentation of the maxilla ends at a Le Fort I-like level: planar
    crops where the field of view or segmentation box ends, and free upper
    rims where thin sinus and nasal walls were not segmented. Masticatory
    load leaves the maxilla through three vertical buttresses
    (nasomaxillary, zygomaticomaxillary, pterygomaxillary; Gruss &
    Mackinnon 1986, *Plast. Reconstr. Surg.* 78:9-22), so ``fixed`` is the
    union of (1) planar crops at the configured extremes of the bone
    (`detect_planar_cuts`), minus the part of a crop that passes through the
    thin palatal process (clamping it would clamp the palate), and (2) the
    superior rims of the buccal walls and tuberosities outside the dental
    arch (`detect_buttress_rims`). Teeth sockets, alveolar ridge and palate
    are never part of it.
``palate``
    Oral surface of the palatal vault, i.e. tongue contact: inferior-facing
    faces, medial to the dental arch (inside the plan hull of the tooth
    sockets, extended posteriorly to include the palatine shelves), at least
    ``palate.min_socket_distance_mm`` from any socket so that the alveolar
    crest and the gingival margin are excluded. The nasal floor faces
    superiorly and is excluded by the normal test.
``occlusal``
    Where bite forces enter the bone: the socket walls (bone faces shared
    with the PDL; Open-Full-Jaw provides them exactly) plus the alveolar
    crest, i.e. inferior-facing faces within ``occlusal.crest_distance_mm``
    of a socket.

Material layers (disjoint, covering)
------------------------------------
``suture_mid``
    Midsagittal slab |lr| < ``suture_half_width_mm`` limited superiorly to
    the palatal process: si <= (highest palatal vault point on the midline)
    + ``suture_nasal_margin_mm``. The midpalatal suture is a palatal
    structure; it continues anteriorly as the intermaxillary suture between
    the central incisors, which the slab also covers, but above the nasal
    floor the midline belongs to the nasal septum (vomer, septal cartilage),
    not to the maxilla, so the slab must not extend up the whole height.
``cortical``
    Elements whose centroid lies within ``cortical_thickness_mm`` of the
    periosteal surface (all boundary faces except the ``fixed`` cuts, across
    which the bone continues, and except the sockets), or within
    ``socket_lining_mm`` of a socket wall (lamina dura). Using distance
    rather than "owns a boundary face" makes the shell thickness independent
    of mesh size.
``cancellous``
    All remaining elements.

The previous models' "lateral suture" (an anterior-lateral block of
elements) is not represented: it did not correspond to an anatomical
suture of the maxilla. The relevant lateral articulations (zygomaticomaxillary,
frontomaxillary, pterygomaxillary) lie at or beyond the cut surfaces and are
represented by the ``fixed`` boundary condition, not by a material layer.

Reporting regions (disjoint, covering)
--------------------------------------
``region_palatal_vault``  closer to the palate surface than to the occlusal
surface and within ``vault_max_depth_mm`` of it.
``region_alveolar_anterior`` / ``region_alveolar_posterior``  tooth-bearing
bone (below the root-apex level) split at the canine/premolar transition,
taken as ``anterior_fraction`` of the arch depth behind the incisors.
``region_basal``  everything else (maxillary body above the alveolar
process, toward the cut surfaces).

Element fields
--------------
``depth`` (M,) distance from the centroid to the natural surface, m;
``surface_normal`` (M,3) unit outward normal of that nearest face.
"""
from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay

from craniofea.mesh.primitives import face_keys, face_normals_areas, tet_signed_volumes
from craniofea.mesh.surface import SurfaceQuery, keep_large_components
from craniofea.model import AnatomicalFrame, Mesh

log = logging.getLogger(__name__)

MM = 1e-3

DEFAULTS: dict = {
    "fixed": {
        "directions": ["superior"],
        "angle_tol_deg": 12.0,
        "plane_tol_mm": 0.2,
        "search_depth_mm": 3.0,
        "min_area_fraction": 0.01,
        "palatal_exclusion_radius_mm": 2.0,
        "palatal_max_thickness_mm": 5.0,
        "rims": True,
        "rim_max_angle_deg": 50.0,
        "rim_buccal_offset_mm": 1.5,
        "rim_min_height_mm": 8.0,
        "rim_min_component_fraction": 0.02,
    },
    "palate": {
        "max_angle_deg": 75.0,
        "min_socket_distance_mm": 2.0,
        "posterior_extension_mm": 15.0,
        "last_tooth_band_mm": 8.0,
        "posterior_wedge_slope": 1.0,
        "pns_margin_mm": 1.0,
        "min_component_fraction": 0.05,
    },
    "occlusal": {
        "crest_distance_mm": 1.0,
        "crest_max_angle_deg": 70.0,
    },
    "layers": {
        "cortical_thickness_mm": 1.5,
        "socket_lining_mm": 0.4,
        "suture_half_width_mm": 0.5,
        "suture_nasal_margin_mm": 4.0,
    },
    "regions": {
        "anterior_fraction": 0.30,
        "alveolar_min_height_mm": 12.0,
        "vault_max_depth_mm": 8.0,
    },
    "template": {
        "crest_half_width_mm": 3.0,
        "palate_min_q_mm": 4.0,
    },
}

CANONICAL_SURFACES = ("all", "palate", "occlusal", "fixed")
LAYERS = ("cortical", "cancellous", "suture_mid")
REGIONS = ("region_palatal_vault", "region_alveolar_anterior", "region_alveolar_posterior",
           "region_basal")


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class LabelConfig:
    fixed: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["fixed"]))
    palate: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["palate"]))
    occlusal: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["occlusal"]))
    layers: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["layers"]))
    regions: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["regions"]))
    template: dict = field(default_factory=lambda: copy.deepcopy(DEFAULTS["template"]))

    @classmethod
    def from_dict(cls, d: dict | LabelConfig | None) -> LabelConfig:
        if isinstance(d, LabelConfig):
            return d
        unknown = set(d or {}) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown label config sections: {sorted(unknown)}")
        merged = _merge(DEFAULTS, d)
        return cls(**merged)

    def as_dict(self) -> dict:
        return {k: copy.deepcopy(getattr(self, k)) for k in DEFAULTS}


# --------------------------------------------------------------------------- surfaces


def _face_geometry(mesh: Mesh, faces: np.ndarray, frame: AnatomicalFrame):
    n, a = face_normals_areas(mesh.nodes, faces)
    c = mesh.nodes[faces].mean(axis=1)
    R = np.stack([frame.lr, frame.ap, frame.si])
    return n @ R.T, a, frame.to_local(c)


def _subset(faces: np.ndarray, of: np.ndarray) -> np.ndarray:
    """Boolean mask of rows of `faces` that also occur in `of` (any winding)."""
    if len(of) == 0:
        return np.zeros(len(faces), dtype=bool)
    return np.isin(face_keys(faces), face_keys(of))


def _direction(frame: AnatomicalFrame, name: str) -> np.ndarray:
    table = {"superior": frame.si, "inferior": -frame.si, "anterior": frame.ap,
             "posterior": -frame.ap, "right": frame.lr, "left": -frame.lr}
    try:
        return np.asarray(table[name], dtype=float)
    except KeyError as err:
        raise ValueError(f"unknown cut direction {name!r}; use one of {sorted(table)}") from err


def detect_planar_cuts(mesh: Mesh, frame: AnatomicalFrame, cfg: dict,
                       palate: np.ndarray | None = None) -> np.ndarray:
    """Planar cut surfaces at the extremes of the bone (see module docstring).

    For each direction d: candidate faces face along d (within angle_tol)
    and lie within search_depth of the extreme of the bone along d; the
    plane offset is seeded at the area-weighted mode of the candidates'
    height and refined by area-weighted least-squares plane fits (which
    allows a cut that is tilted relative to the frame). Faces within
    plane_tol of the plane and parallel to it form the cut. Cuts covering
    less than min_area_fraction of the boundary are discarded.

    If `palate` faces are given, cut faces lying within
    ``palatal_exclusion_radius_mm`` (in plan) of a palate face that is less
    than ``palatal_max_thickness_mm`` below them are dropped: there the crop
    passed through the thin palatal process just under the nasal floor,
    and clamping it would clamp the palate itself. Those faces remain free
    (a slightly lowered nasal floor).
    """
    faces = mesh.surface_sets["all"]
    n, a = face_normals_areas(mesh.nodes, faces)
    c = mesh.nodes[faces].mean(axis=1)
    cos_tol = np.cos(np.radians(cfg["angle_tol_deg"]))
    tol = cfg["plane_tol_mm"] * MM
    depth = cfg["search_depth_mm"] * MM
    chosen = np.zeros(len(faces), dtype=bool)
    report = {}
    for name in cfg["directions"]:
        d = _direction(frame, name)
        h = c @ d
        hmax = h.max()
        cand = (n @ d > cos_tol) & (h > hmax - depth)
        if not cand.any():
            report[name] = 0.0
            continue
        edges = np.arange(hmax - depth, hmax + tol, tol)
        hist, _ = np.histogram(h[cand], bins=edges, weights=a[cand])
        h0 = edges[np.argmax(hist)] + 0.5 * tol
        sel = cand & (np.abs(h - h0) < tol)
        normal = d
        for _ in range(4):
            w = a[sel]
            x0 = (w[:, None] * c[sel]).sum(0) / w.sum()
            xc = c[sel] - x0
            evals, evecs = np.linalg.eigh((w[:, None] * xc).T @ xc)
            nn = evecs[:, 0] * np.sign(evecs[:, 0] @ d)
            if nn @ d < cos_tol:
                break
            normal = nn
            sel = (np.abs((c - x0) @ normal) < tol) & (n @ normal > cos_tol)
        area = a[sel].sum()
        if area < cfg["min_area_fraction"] * a.sum():
            log.info("no planar cut toward %s (%.1f mm^2 below threshold)", name, area / MM**2)
            report[name] = 0.0
            continue
        sel[sel] = keep_large_components(faces[sel], a[sel], 0.01)
        chosen |= sel
        report[name] = float(a[sel].sum() / MM**2)
        log.info("planar cut toward %s: %d faces, %.1f mm^2, normal tilt %.2f deg", name,
                 sel.sum(), report[name], np.degrees(np.arccos(np.clip(normal @ d, -1, 1))))
    info = mesh.meta.setdefault("labels", {})
    if palate is not None and len(palate) and chosen.any():
        from scipy.spatial import cKDTree

        xp = frame.to_local(mesh.nodes[palate].mean(axis=1))
        xc = frame.to_local(c[chosen])
        r = cfg["palatal_exclusion_radius_mm"] * MM
        tmax = cfg["palatal_max_thickness_mm"] * MM
        near = cKDTree(xp[:, :2]).query_ball_point(xc[:, :2], r)
        over = np.array([bool(len(j)) and np.min(xc[i, 2] - xp[j, 2]) < tmax
                         for i, j in enumerate(near)], dtype=bool)
        idx = np.flatnonzero(chosen)
        chosen[idx[over]] = False
        info["fixed_released_over_palate_mm2"] = float(a[idx[over]].sum() / MM**2)
        log.info("released %d cut faces (%.1f mm^2) over the palatal process", over.sum(),
                 info["fixed_released_over_palate_mm2"])
    info["fixed_area_mm2"] = report
    return faces[chosen]


def detect_buttress_rims(mesh: Mesh, frame: AnatomicalFrame, cfg: dict,
                         occlusal: np.ndarray) -> np.ndarray:
    """Superior rims of the buccal walls (see module docstring, ``fixed``).

    Superior-facing faces (within rim_max_angle of si) that lie radially
    outside the dental arch (buccal/posterior to the nearest occlusal face, in
    plan, by more than rim_buccal_offset) and at least rim_min_height above
    the alveolar crest. These are the free upper edges of the lateral
    maxillary wall and the tuberosity, where the zygomaticomaxillary and
    pterygomaxillary buttresses continue beyond the segmented bone.
    """
    from scipy.spatial import cKDTree

    faces = mesh.surface_sets["all"]
    n_loc, a, x = _face_geometry(mesh, faces, frame)
    xo = frame.to_local(mesh.nodes[occlusal].mean(axis=1))
    centre = xo[:, :2].mean(axis=0)
    _, j = cKDTree(xo[:, :2]).query(x[:, :2])
    radial = (np.linalg.norm(x[:, :2] - centre, axis=1)
              - np.linalg.norm(xo[j, :2] - centre, axis=1))
    crest = np.percentile(xo[:, 2], 5)
    sel = ((n_loc[:, 2] > np.cos(np.radians(cfg["rim_max_angle_deg"])))
           & (radial > cfg["rim_buccal_offset_mm"] * MM)
           & (x[:, 2] > crest + cfg["rim_min_height_mm"] * MM))
    sel[sel] = keep_large_components(faces[sel], a[sel], cfg["rim_min_component_fraction"])
    mesh.meta.setdefault("labels", {})["fixed_rim_area_mm2"] = float(a[sel].sum() / MM**2)
    log.info("buttress rims: %d faces, %.1f mm^2", sel.sum(), a[sel].sum() / MM**2)
    return faces[sel]


def label_fixed(mesh: Mesh, frame: AnatomicalFrame, cfg: dict, palate: np.ndarray,
                occlusal: np.ndarray) -> np.ndarray:
    """``fixed``: planar superior cuts (minus the palatal process) + buttress rims."""
    cuts = detect_planar_cuts(mesh, frame, cfg, palate)
    if not cfg.get("rims", True):
        return cuts
    rims = detect_buttress_rims(mesh, frame, cfg, occlusal)
    both = np.vstack([cuts, rims])
    _, first = np.unique(face_keys(both), return_index=True)
    return both[np.sort(first)]


def _plan_hull_contains(hull_pts: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return Delaunay(hull_pts).find_simplex(pts) >= 0


def label_palate_occlusal(mesh: Mesh, frame: AnatomicalFrame, cfg: LabelConfig,
                          socket: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`palate` and `occlusal` surface sets of a mesh with known tooth-socket faces."""
    faces = mesh.surface_sets["all"]
    n_loc, a, x_loc = _face_geometry(mesh, faces, frame)
    is_socket = _subset(faces, socket)
    d_sock, _ = SurfaceQuery(mesh.nodes, socket).query(mesh.nodes[faces].mean(axis=1))

    # Occlusal: socket walls + alveolar crest (inferior-facing rim bone).
    oc = cfg.occlusal
    crest = (~is_socket & (d_sock < oc["crest_distance_mm"] * MM)
             & (-n_loc[:, 2] > np.cos(np.radians(oc["crest_max_angle_deg"]))))
    occlusal = is_socket | crest

    # Palate: inferior-facing, medial to the arch, away from the sockets.
    pc = cfg.palate
    s_loc = frame.to_local(mesh.nodes[socket].reshape(-1, 3))[:, :2]
    ext = s_loc - [0.0, pc["posterior_extension_mm"] * MM]
    inside = _plan_hull_contains(np.vstack([s_loc, ext]), x_loc[:, :2])
    # Behind the last tooth on each side the hull extension would reach the
    # tuberosity; there the palate must stay medial to the last sockets and
    # narrows towards the posterior nasal spine (wedge of the given slope).
    s_full = frame.to_local(mesh.nodes[socket].reshape(-1, 3))
    for side in (-1.0, 1.0):
        on = np.sign(s_full[:, 0]) == side
        if not on.any():
            continue
        back = s_full[on, 1].min()
        last = on & (s_full[:, 1] < back + pc["last_tooth_band_mm"] * MM)
        lr_in = np.abs(s_full[last, 0]).min()
        limit = lr_in - pc["posterior_wedge_slope"] * (back - x_loc[:, 1])
        inside &= ~((np.sign(x_loc[:, 0]) == side) & (x_loc[:, 1] < back)
                    & (np.abs(x_loc[:, 0]) > limit))
    # The hard palate ends posteriorly at the posterior nasal spine, the most
    # posterior bone on the midline.
    mid = np.abs(x_loc[:, 0]) < 2 * MM
    if mid.any():
        inside &= x_loc[:, 1] > x_loc[mid, 1].min() - pc["pns_margin_mm"] * MM
    pal = (~occlusal & inside & (d_sock > pc["min_socket_distance_mm"] * MM)
           & (-n_loc[:, 2] > np.cos(np.radians(pc["max_angle_deg"]))))
    pal[pal] = keep_large_components(faces[pal], a[pal], pc["min_component_fraction"])
    return faces[pal], faces[occlusal]


# --------------------------------------------------------------------------- volume


def label_volume(mesh: Mesh, cfg: LabelConfig | dict | None = None) -> Mesh:
    """Material layers, reporting regions, `depth`/`surface_normal` and node set `fixed`.

    Requires ``mesh.frame`` and the surface sets all/palate/occlusal/fixed.
    """
    cfg = LabelConfig.from_dict(cfg)
    frame = mesh.frame
    if frame is None:
        raise ValueError("label_volume needs mesh.frame")
    missing = [s for s in CANONICAL_SURFACES if s not in mesh.surface_sets]
    if missing:
        raise ValueError(f"label_volume needs surface sets {missing}")
    faces = mesh.surface_sets["all"]
    fixed = mesh.surface_sets["fixed"]
    natural = faces[~_subset(faces, fixed)]
    cen = mesh.nodes[mesh.tets].mean(axis=1)
    x = frame.to_local(cen)

    depth, idx = SurfaceQuery(mesh.nodes, natural).query(cen)
    nrm, _ = face_normals_areas(mesh.nodes, natural)
    mesh.elem_fields["depth"] = depth
    mesh.elem_fields["surface_normal"] = nrm[idx]

    # Periosteal cortex is measured from the external surface; the socket
    # lining (lamina dura) is a much thinner plate.
    lc = cfg.layers
    socket = mesh.surface_sets.get("socket", np.zeros((0, 3), dtype=np.int64))
    if len(socket):
        is_sock = _subset(natural, socket)
        d_out = (SurfaceQuery(mesh.nodes, natural[~is_sock]).query(cen)[0]
                 if (~is_sock).any() else np.full(len(cen), np.inf))
        d_sock = SurfaceQuery(mesh.nodes, natural[is_sock]).query(cen)[0]
        shell = ((d_out < lc["cortical_thickness_mm"] * MM)
                 | (d_sock < lc["socket_lining_mm"] * MM))
    else:
        shell = depth < lc["cortical_thickness_mm"] * MM

    # Layers.
    pal = mesh.surface_sets["palate"]
    pal_x = frame.to_local(mesh.nodes[pal].mean(axis=1)) if len(pal) else np.zeros((0, 3))
    near_mid = np.abs(pal_x[:, 0]) < 5 * MM
    si_vault = (pal_x[near_mid, 2].max() if near_mid.any()
                else (pal_x[:, 2].max() if len(pal_x) else x[:, 2].max()))
    si_cap = si_vault + lc["suture_nasal_margin_mm"] * MM
    suture = (np.abs(x[:, 0]) < lc["suture_half_width_mm"] * MM) & (x[:, 2] <= si_cap)
    cortical = ~suture & shell
    cancellous = ~suture & ~cortical

    # Regions.
    rc = cfg.regions
    occ = mesh.surface_sets["occlusal"]
    d_occ, _ = SurfaceQuery(mesh.nodes, occ).query(cen)
    d_pal = SurfaceQuery(mesh.nodes, pal).query(cen)[0] if len(pal) else np.full(len(cen), np.inf)
    occ_x = frame.to_local(mesh.nodes[occ].mean(axis=1))
    crest = np.percentile(occ_x[:, 2], 5)
    si_apex = max(np.percentile(occ_x[:, 2], 95), crest + rc["alveolar_min_height_mm"] * MM)
    ap_front, ap_back = np.percentile(occ_x[:, 1], [99.5, 0.5])
    ap_split = ap_front - rc["anterior_fraction"] * (ap_front - ap_back)
    vault = (d_pal <= d_occ) & (d_pal <= rc["vault_max_depth_mm"] * MM)
    alveolar = ~vault & (x[:, 2] <= si_apex)
    basal = ~vault & ~alveolar

    sets = {
        "cortical": cortical, "cancellous": cancellous, "suture_mid": suture,
        "region_palatal_vault": vault,
        "region_alveolar_anterior": alveolar & (x[:, 1] >= ap_split),
        "region_alveolar_posterior": alveolar & (x[:, 1] < ap_split),
        "region_basal": basal,
    }
    for k, m in sets.items():
        mesh.elem_sets[k] = np.flatnonzero(m).astype(np.int64)
    mesh.node_sets["fixed"] = np.unique(fixed).astype(np.int64)

    info = mesh.meta.setdefault("labels", {})
    info.update({
        "config": cfg.as_dict(),
        "suture_si_cap_mm": float(si_cap / MM),
        "alveolar_crest_si_mm": float(crest / MM),
        "alveolar_apex_si_mm": float(si_apex / MM),
        "anterior_split_ap_mm": float(ap_split / MM),
    })
    return mesh


def label_ct_mesh(mesh: Mesh, cfg: LabelConfig | dict | None = None) -> Mesh:
    """All canonical labels for an imported CT mesh with a ``socket`` surface set."""
    cfg = LabelConfig.from_dict(cfg)
    if "socket" not in mesh.surface_sets or len(mesh.surface_sets["socket"]) == 0:
        raise ValueError("CT labeling needs the tooth-socket surface set 'socket'")
    frame = mesh.frame
    palate, occlusal = label_palate_occlusal(mesh, frame, cfg, mesh.surface_sets["socket"])
    mesh.surface_sets["palate"] = palate
    mesh.surface_sets["occlusal"] = occlusal
    mesh.surface_sets["fixed"] = label_fixed(mesh, frame, cfg.fixed, palate, occlusal)
    return label_volume(mesh, cfg)


# --------------------------------------------------------------------------- checks & report


def validate_labels(mesh: Mesh) -> list[str]:
    """Return a list of contract violations (empty if the labels are consistent)."""
    problems = []
    M = mesh.n_elems
    for group in (LAYERS, REGIONS):
        count = np.zeros(M, dtype=np.int64)
        for name in group:
            if name not in mesh.elem_sets:
                problems.append(f"missing elem set {name}")
                continue
            np.add.at(count, mesh.elem_sets[name], 1)
        if np.any(count != 1):
            problems.append(f"{'/'.join(group)}: {int((count == 0).sum())} uncovered, "
                            f"{int((count > 1).sum())} multiply assigned elements")
    for name in CANONICAL_SURFACES:
        if name not in mesh.surface_sets or len(mesh.surface_sets[name]) == 0:
            problems.append(f"surface set {name} missing or empty")
    if "fixed" not in mesh.node_sets or len(mesh.node_sets["fixed"]) == 0:
        problems.append("node set fixed missing or empty")
    allf = mesh.surface_sets.get("all", np.zeros((0, 3), dtype=np.int64))
    for name in ("palate", "occlusal", "fixed"):
        s = mesh.surface_sets.get(name)
        if s is not None and len(s) and not _subset(s, allf).all():
            problems.append(f"surface set {name} contains non-boundary faces")
    for name, shape in (("depth", (M,)), ("surface_normal", (M, 3))):
        f = mesh.elem_fields.get(name)
        if f is None or f.shape != shape:
            problems.append(f"elem field {name} missing or wrong shape")
    if mesh.frame is None:
        problems.append("frame missing")
    return problems


def summary_rows(mesh: Mesh) -> list[tuple[str, str, int, float, str]]:
    """(kind, name, count, measure, unit) rows: surface areas in mm^2, volumes in mm^3."""
    rows = []
    vol = np.abs(tet_signed_volumes(mesh.nodes, mesh.tets)) / MM**3
    for k, f in mesh.surface_sets.items():
        _, a = face_normals_areas(mesh.nodes, f)
        rows.append(("surface", k, len(f), float(a.sum() / MM**2), "mm^2"))
    for k, e in mesh.elem_sets.items():
        rows.append(("elements", k, len(e), float(vol[e].sum()), "mm^3"))
    for k, nset in mesh.node_sets.items():
        rows.append(("nodes", k, len(nset), float("nan"), ""))
    return rows


def summary_table(mesh: Mesh) -> str:
    vol = np.abs(tet_signed_volumes(mesh.nodes, mesh.tets)).sum() / MM**3
    lines = [f"{mesh.meta.get('source', 'mesh')}: {mesh.n_nodes} nodes, {mesh.n_elems} tets, "
             f"{vol:.0f} mm^3",
             f"{'kind':<9} {'set':<27} {'count':>9} {'measure':>12}"]
    for kind, name, cnt, meas, unit in summary_rows(mesh):
        m = "" if np.isnan(meas) else f"{meas:12.1f} {unit}"
        lines.append(f"{kind:<9} {name:<27} {cnt:>9d} {m}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- pipeline


def load_config(config: str | Path | dict | None) -> dict:
    if config is None:
        return {}
    if isinstance(config, dict):
        return copy.deepcopy(config)
    import yaml

    with open(config) as fh:
        return yaml.safe_load(fh) or {}


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                 else str(o)))


def build_ct_mesh(config: str | Path | dict | None = "configs/mesh_ct.yaml",
                  root: str | Path = ".", save: bool = True) -> Mesh:
    """Import the Open-Full-Jaw maxilla, detect its frame, label it and save it.

    Paths in the configuration are relative to `root`. The output
    (``output`` key, default ``data/derived/patient3_maxilla.npz``) contains
    derived data of a CC BY-NC-SA dataset and must not be committed.
    """
    from craniofea.mesh.ct import load_open_full_jaw
    from craniofea.mesh.frame import detect_frame

    cfg = load_config(config)
    root = Path(root)
    src = cfg.get("source", {})
    msh = root / src.get("msh", "data/open_full_jaw/extracted/Patient_3/output/maxilla/"
                                "volumetric_meshes/maxilla_volumetric_mesh.msh")
    mesh = load_open_full_jaw(msh, scale_to_m=float(src.get("scale_to_m", 1e-3)),
                              keep=tuple(src.get("keep", ["bone"])), tags=src.get("tags"),
                              min_normalized_volume=float(src.get("min_normalized_volume",
                                                                  1e-6)))
    mesh.frame = detect_frame(mesh, (cfg.get("frame") or {}).get("override"))
    label_ct_mesh(mesh, cfg.get("labels"))
    mesh.meta["patient"] = src.get("patient", 3)
    mesh.meta["license"] = "Derived from Open-Full-Jaw (CC BY-NC-SA 4.0); do not redistribute."
    mesh.meta = _jsonable(mesh.meta)
    problems = validate_labels(mesh)
    if problems:
        raise RuntimeError("label validation failed: " + "; ".join(problems))
    log.info("\n%s", summary_table(mesh))
    if save:
        out = root / cfg.get("output", "data/derived/patient3_maxilla.npz")
        out.parent.mkdir(parents=True, exist_ok=True)
        mesh.save(out)
        log.info("saved %s", out)
    return mesh

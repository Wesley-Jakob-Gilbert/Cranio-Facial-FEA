"""Parametric adolescent maxilla template (used when the CT data are absent).

Geometry
--------
The planform is the region enclosed by the buccal outline of a U-shaped
alveolar ridge and a posterior palatal border. The ridge centreline (the
alveolar crest) is the arch curve of the earlier ``template_maxilla_v2``
model,

    ap(s) = D * S(s) * (1 - f + f S(s))
    lr(s) = -(W/2) C(s) - flare(s),     s in [0, 1],

with S = sin(pi s)^(2/p), C = sign(cos pi s) |cos pi s|^(2/p), running from
the left tuberosity (s=0) through the incisors (s=1/2) to the right
tuberosity (s=1); D is the arch depth, W the inter-crest width, f the legacy
shape term and p a superellipse exponent. p = 2 recovers the legacy
elliptic curve; p ~ 2.8 gives the ovoid arch with near-parallel posterior
segments typical of the adolescent maxilla. With q the signed plan distance from
the crest (q > 0 palatal), the inferior (oral) surface is

    z_inf = h_b (q/w_b)^2                       buccal slope, -w_b <= q <= 0
    z_inf = H * smoothstep(q / w_p)             palatal wall rising into the vault

so the crest is the lowest bone and the vault reaches height H above it.
The superior surface is the nasal floor (z = H + t_p) over the palate and
rises to a flat cut plane above the alveolar processes, where the maxillary
body would continue to the zygoma, nasal walls and cranial base.

Default dimensions are those of an adolescent (12-16 y) maxilla: alveolar
crest width ~46 mm across the molars and arch depth ~40 mm (longitudinal
norms of Bishara et al., *Am. J. Orthod. Dentofacial Orthop.* 111:401-409,
1997 and *Angle Orthod.* 68:69-74, 1998), palatal vault ~12 mm above the
crest and a palatal process ~3.5 mm thick; they are of the same order as
the Open-Full-Jaw patient-3 maxilla (vault ~12.5 mm above the crest).

Discretisation
--------------
The planform is triangulated (Delaunay on a hexagonal lattice plus
boundary, crest and suture rows), extruded in `layers` layers between
z_inf and z_sup, and every prism is split into three tets with the
diagonal of each side quad running from the top of its lower-numbered to
the bottom of its higher-numbered vertex. Because the rule depends only on
global vertex numbers, neighbouring prisms agree on shared faces and the
tet mesh is conforming. Two rows of plan nodes at lr = +/- `suture_half_width`
give the midpalatal suture its own element column.

Coordinates: x = lr (patient right +), y = ap (anterior +), z = si
(superior +), i.e. patient RAS, with the frame stored on the mesh.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, fields

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import Delaunay, cKDTree

from craniofea.mesh.primitives import boundary_faces, face_normals_areas, orient_tets
from craniofea.model import AnatomicalFrame, Mesh

log = logging.getLogger(__name__)


@dataclass
class TemplateParams:
    """Template dimensions in millimetres (converted to metres on output)."""

    arch_width: float = 46.0          # crest-to-crest width at the tuberosities
    arch_depth: float = 40.0          # posterior border -> incisor crest
    arch_flatness: float = 0.0        # legacy shape term (> 0 sharpens the incisor curve)
    arch_squareness: float = 2.8      # superellipse exponent (2 = legacy elliptic arch)
    posterior_flare_deg: float = 4.0  # tuberosities angle slightly outward
    buccal_half_width: float = 4.0    # crest -> buccal outline (plan)
    buccal_height: float = 5.0        # height of the buccal outline above the crest
    palatal_wall_width: float = 9.0   # crest -> start of the vault roof (plan)
    vault_height: float = 12.0        # palatal vault apex above the crest
    palate_thickness: float = 3.5     # palatal process thickness (oral -> nasal)
    body_height: float = 3.0          # cut plane above the nasal floor
    body_band: float = 0.0            # q below which the bone reaches the cut plane
    body_transition: float = 4.0      # lateral nasal wall width
    posterior_bow: float = 1.5        # posterior nasal spine projection
    plan_size: float = 2.4            # target plan edge length
    layers: int = 6                   # elements through the thickness
    suture_half_width: float = 0.5    # midpalatal suture column half-width

    @classmethod
    def from_dict(cls, d: dict | None) -> TemplateParams:
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown template parameters: {sorted(unknown)}")
        return cls(**d)


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def crest_curve(p: TemplateParams, n: int = 2001) -> np.ndarray:
    """(n, 2) alveolar-crest centreline in plan (lr, ap), left -> anterior -> right."""
    s = np.linspace(0.0, 1.0, n)
    e = 2.0 / p.arch_squareness
    sn = np.abs(np.sin(np.pi * s)) ** e
    cs = np.sign(np.cos(np.pi * s)) * np.abs(np.cos(np.pi * s)) ** e
    ap = p.arch_depth * sn * (1.0 - p.arch_flatness + p.arch_flatness * sn)
    smooth_sign = cs / (np.abs(cs) + 0.15)
    flare = (1.0 - sn) ** 2 * np.tan(np.radians(p.posterior_flare_deg)) * ap * smooth_sign
    lr = -(p.arch_width / 2.0) * cs - flare
    return np.column_stack([lr, ap])


def _offset(curve: np.ndarray, dist: float) -> np.ndarray:
    """Offset a plan polyline by `dist` to its outer (buccal) side."""
    t = np.gradient(curve, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    outward = np.column_stack([-t[:, 1], t[:, 0]])  # left normal of a left->right sweep
    return curve + dist * outward


def _resample(curve: np.ndarray, h: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(np.ceil(s[-1] / h)), 2)
    si = np.linspace(0.0, s[-1], n + 1)
    return np.column_stack([np.interp(si, s, curve[:, 0]), np.interp(si, s, curve[:, 1])])


class _Planform:
    """Plan geometry: crest distance field, outline polygon and surfaces."""

    def __init__(self, p: TemplateParams):
        self.p = p
        self.crest = crest_curve(p)
        self.outline = _offset(self.crest, p.buccal_half_width)
        self.outline[:, 1] = np.maximum(self.outline[:, 1], 0.0)
        a, b = self.outline[0], self.outline[-1]
        lr = np.linspace(b[0], a[0], 201)[1:-1]
        half = 0.5 * (b[0] - a[0])
        mid = 0.5 * (a[0] + b[0])
        border = np.column_stack([lr, -p.posterior_bow * (1.0 - ((lr - mid) / half) ** 2)])
        self.polygon = np.vstack([self.outline, border])
        self.path = MplPath(self.polygon)
        # Palatal side of the crest: the crest closed by rays running posteriorly.
        self.crest_path = MplPath(np.vstack([self.crest, [[self.crest[-1, 0], -1e3],
                                                          [self.crest[0, 0], -1e3]]]))
        self.crest_tree = cKDTree(self.crest)
        self.half = half
        self.mid = mid

    def border_ap(self, lr: np.ndarray) -> np.ndarray:
        """ap of the (bowed) posterior palatal border at plan position lr."""
        return -self.p.posterior_bow * (1.0 - ((lr - self.mid) / self.half) ** 2)

    def outline_ap(self, lr: float) -> float:
        """ap of the anterior buccal outline at lr (near the midline)."""
        o = self.outline[self.outline[:, 1] > 0.5 * self.outline[:, 1].max()]
        o = o[np.argsort(o[:, 0])]
        return float(np.interp(lr, o[:, 0], o[:, 1]))

    def q(self, xy: np.ndarray) -> np.ndarray:
        """Signed plan distance from the crest (mm), positive on the palatal side."""
        d, _ = self.crest_tree.query(xy)
        inside = self.crest_path.contains_points(xy)
        return np.where(inside, d, -d)

    def z_inf(self, xy: np.ndarray) -> np.ndarray:
        p, q = self.p, self.q(xy)
        buccal = p.buccal_height * np.clip(-q / p.buccal_half_width, 0.0, 1.0) ** 2
        palatal = p.vault_height * smoothstep(q / p.palatal_wall_width)
        return np.where(q < 0, buccal, palatal)

    def z_sup(self, xy: np.ndarray) -> np.ndarray:
        p, q = self.p, self.q(xy)
        floor = p.vault_height + p.palate_thickness
        cut = floor + p.body_height
        t = smoothstep((q - p.body_band) / p.body_transition)
        return cut + (floor - cut) * t


def _plan_points(pf: _Planform) -> np.ndarray:
    """Plan nodes: outline, border, crest, cut-edge and suture rows, then a hex lattice."""
    h, hw = pf.p.plan_size, pf.p.suture_half_width
    border = np.vstack([pf.outline[-1:], pf.polygon[len(pf.outline):], pf.outline[:1]])
    rows = [_resample(pf.outline, h), _resample(border, h)[1:-1],
            _resample(pf.crest, h)[1:-1]]
    # Edges of the cut plane and of the nasal floor (inward offsets of the crest).
    for q in (pf.p.body_band, pf.p.body_band + pf.p.body_transition):
        if q > 0.3 * h:
            r = _resample(_offset(pf.crest, -q), h)
            rows.append(r[r[:, 1] > pf.border_ap(r[:, 0]) + 0.4 * h])
    for side in (-hw, hw):
        a0, a1 = float(pf.border_ap(np.array(side))), pf.outline_ap(side)
        ap = np.linspace(a0, a1, int(np.ceil((a1 - a0) / h)) + 1)
        rows.append(np.column_stack([np.full_like(ap, side), ap]))

    # Accept feature rows in priority order, dropping points crowding earlier rows.
    pts = rows[0]
    for r in rows[1:]:
        d, _ = cKDTree(pts).query(r)
        pts = np.vstack([pts, r[d > 0.4 * h]])

    # Hexagonal lattice in the interior, kept clear of the feature rows.
    lo, hi = pf.polygon.min(0), pf.polygon.max(0)
    dy = h * np.sqrt(3) / 2
    ys = np.arange(lo[1], hi[1] + dy, dy)
    xs = np.arange(lo[0] - h, hi[0] + h, h)
    X, Y = np.meshgrid(xs, ys)
    X = X + 0.5 * h * (np.arange(len(ys))[:, None] % 2)
    lat = np.column_stack([X.ravel(), Y.ravel()])
    lat = lat[pf.path.contains_points(lat)]
    d, _ = cKDTree(pts).query(lat)
    lat = lat[(d > 0.55 * h) & (np.abs(lat[:, 0]) > hw + 0.45 * h)]
    return np.vstack([pts, lat])


def _triangulate(pf: _Planform, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = Delaunay(pts)
    t = tri.simplices
    c = pts[t].mean(axis=1)
    t = t[pf.path.contains_points(c)]
    e1, e2 = pts[t[:, 1]] - pts[t[:, 0]], pts[t[:, 2]] - pts[t[:, 0]]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    t[area < 0] = t[area < 0][:, [0, 2, 1]]
    if np.abs(area).min() < 1e-3 * pf.p.plan_size ** 2:
        raise RuntimeError("template planform triangulation has degenerate triangles")
    x, y = pf.polygon[:, 0], pf.polygon[:, 1]
    poly_area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if abs(np.abs(area).sum() - poly_area) > 0.01 * poly_area:
        raise RuntimeError("template planform triangulation does not cover the outline "
                           "(non-convex outline? reduce posterior_flare_deg)")
    used = np.unique(t)
    remap = np.full(len(pts), -1)
    remap[used] = np.arange(len(used))
    return pts[used], remap[t]


def _extrude(xy: np.ndarray, tri: np.ndarray, z_lo: np.ndarray, z_hi: np.ndarray,
             layers: int) -> tuple[np.ndarray, np.ndarray]:
    n2 = len(xy)
    frac = np.linspace(0.0, 1.0, layers + 1)
    z = z_lo[None, :] + frac[:, None] * (z_hi - z_lo)[None, :]
    nodes = np.column_stack([np.tile(xy, (layers + 1, 1)), z.ravel()])
    v = np.sort(tri, axis=1)                   # global-order rule for conformity
    tets = []
    for k in range(layers):
        b, t = v + k * n2, v + (k + 1) * n2
        tets += [np.column_stack([b[:, 0], b[:, 1], b[:, 2], t[:, 0]]),
                 np.column_stack([b[:, 1], b[:, 2], t[:, 0], t[:, 1]]),
                 np.column_stack([b[:, 2], t[:, 0], t[:, 1], t[:, 2]])]
    return nodes, np.concatenate(tets)


def build_template_mesh(params: TemplateParams | dict | None = None,
                        labels: dict | None = None) -> Mesh:
    """Build and label the parametric maxilla template (SI units).

    `labels` is the shared labeling configuration (see
    `craniofea.mesh.labeling.LabelConfig`); template-specific surface
    widths are read from ``labels["template"]``.
    """
    from craniofea.mesh.labeling import LabelConfig, label_fixed, label_volume

    p = params if isinstance(params, TemplateParams) else TemplateParams.from_dict(params)
    cfg = LabelConfig.from_dict(labels)
    pf = _Planform(p)
    xy, tri = _triangulate(pf, _plan_points(pf))
    z_lo, z_hi = pf.z_inf(xy), pf.z_sup(xy)
    nodes_mm, tets = _extrude(xy, tri, z_lo, z_hi, p.layers)
    nodes = nodes_mm * 1e-3
    tets = orient_tets(nodes, tets)
    faces, _ = boundary_faces(nodes, tets)

    n2 = len(xy)
    layer = np.arange(len(nodes)) // n2
    bottom = (layer[faces] == 0).all(axis=1)
    q_face = pf.q(xy[faces % n2].mean(axis=1))
    tcfg = cfg.template
    occlusal = faces[bottom & (np.abs(q_face) <= tcfg["crest_half_width_mm"])]
    palate = faces[bottom & (q_face >= tcfg["palate_min_q_mm"])]

    origin = nodes.mean(axis=0)
    origin[0] = 0.0                      # the template is symmetric about lr = 0
    frame = AnatomicalFrame(origin=origin, lr=np.array([1.0, 0, 0]),
                            ap=np.array([0, 1.0, 0]), si=np.array([0, 0, 1.0]))
    mesh = Mesh(nodes=nodes, tets=tets, frame=frame,
                surface_sets={"all": faces, "palate": palate, "occlusal": occlusal},
                meta={"source": "parametric_template", "params_mm": p.__dict__.copy(),
                      "frame": {"method": "constructed"}})
    mesh.surface_sets["fixed"] = label_fixed(mesh, frame, cfg.fixed, palate, occlusal)
    label_volume(mesh, cfg)
    _, a = face_normals_areas(nodes, palate)
    log.info("template: %d nodes, %d tets, palate %.0f mm^2", mesh.n_nodes, mesh.n_elems,
             a.sum() * 1e6)
    return mesh


def build_template_from_config(config: str | dict | None = "configs/mesh_template.yaml") -> Mesh:
    """`build_template_mesh` from a ``configs/mesh_template.yaml``-style mapping or file."""
    from craniofea.mesh.labeling import load_config

    cfg = load_config(config)
    return build_template_mesh(cfg.get("geometry"), cfg.get("labels"))

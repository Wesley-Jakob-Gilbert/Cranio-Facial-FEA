"""Matplotlib figures of labeled meshes (no OpenGL needed).

Surfaces are drawn with a painter's algorithm (triangles sorted far-to-near,
Lambert shading); sections are the exact intersection polygons of the tets
with a plane, so both work headless and scale to the 0.4 M-tet CT mesh.
"""
from __future__ import annotations

import numpy as np

from craniofea.mesh.primitives import face_keys, face_normals_areas
from craniofea.model import Mesh

MM = 1e-3

# Colour-blind-safe palette (Okabe & Ito 2008) for labels; greys for bone.
COLORS = {
    "bone": "#dcd6c8",
    "palate": "#0072B2",
    "occlusal": "#D55E00",
    "fixed": "#009E73",
    "socket": "#E69F00",
    "cortical": "#7f7f7f",
    "cancellous": "#ecdfbd",
    "suture_mid": "#CC79A7",
    "region_palatal_vault": "#56B4E9",
    "region_alveolar_anterior": "#E69F00",
    "region_alveolar_posterior": "#D55E00",
    "region_basal": "#a6a6a6",
}

# Named views: (azimuth from +lr towards +ap, elevation towards +si), degrees.
VIEWS = {
    "inferior": (0.0, -90.0), "superior": (0.0, 90.0), "right": (0.0, 0.0),
    "left": (180.0, 0.0), "anterior": (90.0, 0.0), "posterior": (-90.0, 0.0),
    "superolateral": (50.0, 40.0),
}


def _rgb(c: str) -> np.ndarray:
    from matplotlib.colors import to_rgb

    return np.asarray(to_rgb(c))


def view_axes(frame, view: str | tuple[float, float]) -> tuple[np.ndarray, ...]:
    """(screen-x, screen-y, towards-viewer) unit vectors; right-handed, so not mirrored."""
    az, el = VIEWS[view] if isinstance(view, str) else view
    az, el = np.radians(az), np.radians(el)
    ez = (np.cos(el) * (np.cos(az) * frame.lr + np.sin(az) * frame.ap)
          + np.sin(el) * frame.si)
    up = frame.ap if abs(np.cos(el)) < 1e-6 else frame.si
    ey = up - (up @ ez) * ez
    ey /= np.linalg.norm(ey)
    return np.cross(ey, ez), ey, ez


def draw_surface(ax, mesh: Mesh, view: str | tuple[float, float],
                 highlight: dict[str, str] | None = None, light=(-0.4, 0.5, 1.0)) -> None:
    """Shaded boundary of `mesh` seen from an anatomical direction.

    `highlight` maps surface-set names to colours, applied in order.
    """
    from matplotlib.collections import PolyCollection

    ex, ey, ez = view_axes(mesh.frame, view)
    faces = mesh.surface_sets["all"]
    x = (mesh.nodes - mesh.frame.origin) / MM
    n, _ = face_normals_areas(mesh.nodes, faces)
    facing = n @ ez > 0
    idx = np.flatnonzero(facing)
    tri, n = x[faces[idx]], n[idx]
    color = np.tile(_rgb(COLORS["bone"]), (len(tri), 1))
    keys = face_keys(faces[idx])
    for name, col in (highlight or {}).items():
        s = mesh.surface_sets.get(name)
        if s is not None and len(s):
            color[np.isin(keys, face_keys(s))] = _rgb(col)
    lvec = light[0] * ex + light[1] * ey + light[2] * ez
    lvec /= np.linalg.norm(lvec)
    shade = 0.30 + 0.70 * np.clip(n @ lvec, 0, 1)
    order = np.argsort(tri.mean(axis=1) @ ez)
    poly = np.stack([tri @ ex, tri @ ey], axis=-1)[order]
    rgb = np.clip(color[order] * shade[order, None], 0, 1)
    # Edges in the face colour hide anti-aliasing seams between triangles.
    ax.add_collection(PolyCollection(poly, facecolors=rgb, edgecolors=rgb, linewidths=0.3))
    lo, hi = poly.reshape(-1, 2).min(0), poly.reshape(-1, 2).max(0)
    pad = 0.04 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")


def scale_bar(ax, length_mm: float = 10.0, loc: str = "lower right") -> None:
    from matplotlib.font_manager import FontProperties
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    ax.add_artist(AnchoredSizeBar(ax.transData, length_mm, f"{length_mm:g} mm", loc,
                                  pad=0.4, frameon=False, size_vertical=0.4,
                                  fontproperties=FontProperties(size=8)))


def section_polygons(mesh: Mesh, axis: str, offset_mm: float
                     ) -> tuple[list[np.ndarray], np.ndarray]:
    """Exact intersection of the tets with the plane ``local[axis] = offset_mm``.

    axis: 'lr' (sagittal, drawn in (ap, si)), 'ap' (coronal, (lr, si)) or
    'si' (axial, (lr, ap)). Returns (polygons in local mm, element ids).
    """
    names = ("lr", "ap", "si")
    k = names.index(axis)
    others = [i for i in range(3) if i != k]
    xl = mesh.frame.to_local(mesh.nodes) / MM
    s = xl[:, k][mesh.tets] - offset_mm
    e = np.flatnonzero((s.min(axis=1) < 0) & (s.max(axis=1) > 0))
    T, S = mesh.tets[e], s[e]
    pts = np.full((len(e), 6, 2), np.nan)
    for j, (a, b) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        hit = (S[:, a] * S[:, b]) < 0
        t = S[hit, a] / (S[hit, a] - S[hit, b])
        pa, pb = xl[T[hit, a]][:, others], xl[T[hit, b]][:, others]
        pts[hit, j] = pa + t[:, None] * (pb - pa)
    c = np.nanmean(pts, axis=1)
    ang = np.arctan2(pts[..., 1] - c[:, None, 1], pts[..., 0] - c[:, None, 0])
    order = np.argsort(np.where(np.isnan(ang), np.inf, ang), axis=1)
    pts = np.take_along_axis(pts, order[..., None], axis=1)
    return [p[~np.isnan(p[:, 0])] for p in pts], e


def draw_section(ax, mesh: Mesh, axis: str, offset_mm: float, sets: dict[str, str]) -> None:
    """Section coloured by element sets (`sets` maps elem-set name -> colour)."""
    from matplotlib.collections import PolyCollection

    polys, e = section_polygons(mesh, axis, offset_mm)
    col = np.tile(_rgb("#ffffff"), (len(e), 1))
    for name, c in sets.items():
        col[np.isin(e, mesh.elem_sets.get(name, []))] = _rgb(c)
    ax.add_collection(PolyCollection(polys, facecolors=col, edgecolors=col, linewidths=0.15))
    allp = np.vstack(polys)
    ax.set_xlim(allp[:, 0].min() - 1, allp[:, 0].max() + 1)
    ax.set_ylim(allp[:, 1].min() - 1, allp[:, 1].max() + 1)
    ax.set_aspect("equal")


def _clean(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _legend(ax, keys, labels=None, ncol=2, below=-0.02):
    from matplotlib.patches import Patch

    labels = labels or keys
    ax.legend(handles=[Patch(facecolor=COLORS[k], edgecolor="none", label=lab)
                       for k, lab in zip(keys, labels, strict=True)],
              loc="upper center", bbox_to_anchor=(0.5, below), fontsize=8, frameon=False,
              ncol=ncol, handlelength=1.2, columnspacing=1.2)


def label_figure(mesh: Mesh, title: str | None = None, coronal_ap_mm: float = 0.0,
                 sagittal_lr_mm: float = 10.0):
    """Four panels: (a) inferior view with palate/occlusal, (b) superolateral
    view with the fixed cut surfaces, (c) coronal section with material
    layers, (d) parasagittal section with reporting regions."""
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(10.5, 8.6), constrained_layout=True,
                            gridspec_kw={"height_ratios": [1.25, 1.0]})
    ax = axs[0, 0]
    draw_surface(ax, mesh, "inferior", {"palate": COLORS["palate"],
                                        "occlusal": COLORS["occlusal"]})
    _clean(ax)
    scale_bar(ax)
    ax.set_title("(a) inferior (oral) view", loc="left", fontsize=10)
    t = ax.transAxes
    ax.text(0.5, 1.0, "anterior", transform=t, ha="center", va="bottom", fontsize=8)
    ax.text(0.0, 0.5, "R", transform=t, ha="right", va="center", fontsize=9)
    ax.text(1.0, 0.5, "L", transform=t, ha="left", va="center", fontsize=9)
    _legend(ax, ["palate", "occlusal"], ["palate", "occlusal (crest + sockets)"])

    ax = axs[0, 1]
    draw_surface(ax, mesh, "superolateral", {"fixed": COLORS["fixed"],
                                             "occlusal": COLORS["occlusal"]})
    _clean(ax)
    scale_bar(ax)
    ax.set_title("(b) superolateral view (right, anterior, superior)", loc="left",
                 fontsize=10)
    _legend(ax, ["fixed", "occlusal"], ["fixed (cut surfaces)", "occlusal"])

    ax = axs[1, 0]
    layers = ("cortical", "cancellous", "suture_mid")
    draw_section(ax, mesh, "ap", coronal_ap_mm, {k: COLORS[k] for k in layers})
    ax.set_title(f"(c) coronal section ap = {coronal_ap_mm:+.0f} mm: layers", loc="left",
                 fontsize=10)
    ax.set_xlabel("lr (mm)  (patient left  ->  right, seen from behind)", fontsize=8)
    ax.set_ylabel("si (mm)", fontsize=8)
    _legend(ax, list(layers), ncol=3, below=-0.3)

    ax = axs[1, 1]
    regs = ("region_palatal_vault", "region_alveolar_anterior", "region_alveolar_posterior",
            "region_basal")
    draw_section(ax, mesh, "lr", sagittal_lr_mm, {k: COLORS[k] for k in regs})
    side = "left" if sagittal_lr_mm < 0 else "right"
    ax.set_title(f"(d) parasagittal section {abs(sagittal_lr_mm):.0f} mm {side}: regions",
                 loc="left", fontsize=10)
    ax.set_xlabel("ap (mm)  (posterior  ->  anterior)", fontsize=8)
    ax.set_ylabel("si (mm)", fontsize=8)
    _legend(ax, list(regs), [r.replace("region_", "").replace("_", " ") for r in regs],
            ncol=4, below=-0.2)
    for a in axs[1]:
        a.tick_params(labelsize=8)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    if title:
        fig.suptitle(title, fontsize=11)
    return fig

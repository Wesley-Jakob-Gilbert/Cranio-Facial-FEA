#!/usr/bin/env python3
"""Create remodeling density evolution animations.

Uses Poly3DCollection to render the oral surface of the mesh as
solid filled polygons (quads for C3D8 hex, triangles for C3D4 tet).
Each face is coloured by its element's density delta from baseline.

Outputs:
- results/mewing/density_evolution.gif
- results/mouth_breathing/density_evolution.gif
- results/comparison_animation.gif
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCENARIOS = ("mewing", "mouth_breathing")


# ---------------------------------------------------------------------------
# Mesh loading — oral surface faces
# ---------------------------------------------------------------------------

_TET_FACE_LOCAL = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])


def _extract_boundary_tet(nodes_dict, elems_dict):
    """Vectorised boundary-face extraction for C3D4 tet meshes.

    Returns
    -------
    face_verts : (B, 3, 3) vertex coordinates per boundary face
    face_eids  : list[int] — 1-based element id per face
    normals    : (B, 3) unit outward normals
    """
    eid_list = sorted(elems_dict.keys())
    eid_arr = np.array(eid_list)
    tets = np.array([elems_dict[eid] for eid in eid_list])  # (M, 4) 1-based

    max_nid = max(nodes_dict.keys())
    pts = np.zeros((max_nid + 1, 3))
    for nid, coord in nodes_dict.items():
        pts[nid] = coord

    all_faces = tets[:, _TET_FACE_LOCAL].reshape(-1, 3)
    tet_idx = np.repeat(np.arange(len(eid_list)), 4)

    sf = np.sort(all_faces, axis=1)
    N = int(sf.max()) + 1
    keys = (sf[:, 0].astype(np.int64) * N * N
            + sf[:, 1].astype(np.int64) * N
            + sf[:, 2].astype(np.int64))
    _, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    bmask = cnt[inv] == 1

    bf = all_faces[bmask]
    bf_eids = eid_arr[tet_idx[bmask]]

    p0, p1, p2 = pts[bf[:, 0]], pts[bf[:, 1]], pts[bf[:, 2]]
    verts = np.stack([p0, p1, p2], axis=1)
    normals = np.cross(p1 - p0, p2 - p0)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    return verts, bf_eids.tolist(), normals


def load_oral_surface():
    """Return oral-surface face data for Poly3DCollection rendering.

    For C3D4 tet meshes: extracts all downward-facing boundary faces
    (full inferior surface including palate, teeth, and alveolar bone).
    For C3D8 hex meshes: uses NSET_PALATE membership (original logic).

    Returns
    -------
    face_verts : ndarray, shape (N_faces, V, 3) — V=4 for hex, 3 for tet
    face_eids  : list[int]
    suture_mid_mask : ndarray[bool]
    suture_lat_mask : ndarray[bool]
    """
    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    nodes = {int(k): v for k, v in mesh["nodes"].items()}
    elems = {int(k): v for k, v in mesh["elements"].items()}
    sets = mesh.get("sets", {})
    et = mesh.get("elem_type", "C3D8")

    s_mid = set(sets.get("ESET_SUTURE_MID", []))
    s_lat = set(sets.get("ESET_SUTURE_LAT", []))

    if et == "C3D4":
        print("[anim] Extracting boundary faces for tet mesh ...")
        face_verts, face_eids, normals = _extract_boundary_tet(nodes, elems)
        down = normals[:, 2] < -0.3
        face_verts = face_verts[down]
        face_eids = [face_eids[i] for i, m in enumerate(down) if m]
        mid_mask = np.array([eid in s_mid for eid in face_eids])
        lat_mask = np.array([eid in s_lat for eid in face_eids])
        return face_verts, face_eids, mid_mask, lat_mask

    # Hex mesh: original NSET_PALATE-based logic
    palate_set = set(sets.get("NSET_PALATE", []))
    face_verts_list = []
    face_eids = []
    mid_mask = []
    lat_mask = []

    for eid in sorted(elems.keys()):
        conn = elems[eid]
        bottom_face = conn[:4]
        if all(n in palate_set for n in bottom_face):
            face_verts_list.append([nodes[n] for n in bottom_face])
            face_eids.append(eid)
            mid_mask.append(eid in s_mid)
            lat_mask.append(eid in s_lat)
            continue
        top_face = conn[4:8]
        if all(n in palate_set for n in top_face):
            face_verts_list.append([nodes[n] for n in top_face])
            face_eids.append(eid)
            mid_mask.append(eid in s_mid)
            lat_mask.append(eid in s_lat)

    face_verts = np.array(face_verts_list, dtype=float)
    return face_verts, face_eids, np.array(mid_mask), np.array(lat_mask)


def _vertex_extents(face_verts):
    """Return (xs_flat, ys_flat, zs_flat) as 1-D arrays for axis-limit computation."""
    flat = face_verts.reshape(-1, 3)
    return flat[:, 0], flat[:, 1], flat[:, 2]


# ---------------------------------------------------------------------------
# Snapshot loading (unchanged logic)
# ---------------------------------------------------------------------------

def cycle_num(path: Path) -> int:
    m = re.search(r"cycle_(\d+)_density\.json$", path.name)
    return int(m.group(1)) if m else -1


def load_snapshots(scenario: str):
    snap_dir = RESULTS / scenario / "snapshots"
    files = sorted(snap_dir.glob("cycle_*_density.json"), key=cycle_num)
    if not files:
        raise SystemExit(f"No snapshots found for {scenario}: {snap_dir}")

    data = []
    cycles = []
    for p in files:
        payload = json.loads(p.read_text())
        data.append({int(k): float(v) for k, v in payload.items()})
        cycles.append(cycle_num(p))
    return cycles, data


def density_series_by_eids(face_eids, snap_dicts):
    """Build (n_frames, n_faces) array of density values."""
    return np.array(
        [[d.get(eid, 1.0) for eid in face_eids] for d in snap_dicts],
        dtype=float,
    )


def delta_from_baseline(values):
    """values: (n_frames, n_faces) array.  Returns same shape, baseline-subtracted."""
    return values - values[0][np.newaxis, :]


# ---------------------------------------------------------------------------
# Poly3DCollection helpers
# ---------------------------------------------------------------------------

def _build_edge_colors(mid_mask, lat_mask):
    """Per-face edge colors: suture-mid=black, suture-lat=dimgray, bone=0.5 gray."""
    n = len(mid_mask)
    ec = np.full((n, 4), 0.5)  # RGBA — init to medium gray
    # Set alpha channel
    ec[:, 3] = 1.0
    # Convert to list of RGBA tuples for mpl
    edge_colors = []
    for i in range(n):
        if mid_mask[i]:
            edge_colors.append("black")
        elif lat_mask[i]:
            edge_colors.append("dimgray")
        else:
            edge_colors.append((0.5, 0.5, 0.5, 1.0))
    return edge_colors


def _build_linewidths(mid_mask, lat_mask):
    """Per-face linewidths: suture faces thicker."""
    lw = np.full(len(mid_mask), 0.3)
    lw[mid_mask] = 0.8
    lw[lat_mask] = 0.8
    return lw


def _add_surface(ax, face_verts, mid_mask, lat_mask, sm, initial_vals):
    """Create a Poly3DCollection on *ax* and return it.

    Parameters
    ----------
    sm : ScalarMappable  – maps density delta values to RGBA colours.
    initial_vals : 1-D array of delta values for the first frame.
    """
    facecolors = sm.to_rgba(initial_vals)
    n_faces = len(face_verts)

    if n_faces > 5000:
        # Dense mesh: very thin semi-transparent edges give shape without
        # obscuring face colours on small triangles
        poly = Poly3DCollection(
            face_verts,
            facecolors=facecolors,
            edgecolors=(0.65, 0.65, 0.65, 0.15),
            linewidths=0.08,
        )
    else:
        edge_colors = _build_edge_colors(mid_mask, lat_mask)
        linewidths = _build_linewidths(mid_mask, lat_mask)
        poly = Poly3DCollection(
            face_verts,
            facecolors=facecolors,
            edgecolors=edge_colors,
            linewidths=linewidths,
        )
    ax.add_collection3d(poly)
    return poly


def _setup_axes(ax, face_verts):
    """Configure 3-D axes: limits, labels, camera."""
    xs, ys, zs = _vertex_extents(face_verts)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_min, z_max = zs.min(), zs.max()

    # Small padding
    pad_x = (x_max - x_min) * 0.05
    pad_y = (y_max - y_min) * 0.05
    pad_z = (z_max - z_min) * 0.3  # more z padding for near-planar view

    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_y, y_max + pad_y)
    ax.set_zlim(z_min - pad_z, z_max + pad_z)

    ax.set_facecolor((0.93, 0.93, 0.93))
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.set_zlabel("z (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=85, azim=0)


def _add_anatomical_labels(ax, face_verts):
    """Place anterior / posterior labels at mesh extents."""
    xs, ys, zs = _vertex_extents(face_verts)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    z_mid = (zs.min() + zs.max()) / 2.0

    ax.text(x_max * 0.95, 0.0, z_mid, "Anterior", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)
    ax.text(x_min * 0.3, y_max * 0.85, z_mid, "Post. L", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)
    ax.text(x_min * 0.3, y_min * 0.85, z_mid, "Post. R", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)


# ---------------------------------------------------------------------------
# GIF generators
# ---------------------------------------------------------------------------

def make_single_scenario_gif(scenario, cycles, delta_vals, face_verts,
                              mid_mask, lat_mask, vmin, vmax):
    """Render single-scenario density evolution GIF.

    Parameters
    ----------
    delta_vals : ndarray (n_frames, n_faces)
    """
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])

    poly = _add_surface(ax, face_verts, mid_mask, lat_mask, sm, delta_vals[0])
    _setup_axes(ax, face_verts)
    _add_anatomical_labels(ax, face_verts)

    cbar = fig.colorbar(sm, ax=ax, shrink=0.75,
                        label="\u0394\u03c1 from baseline")
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(f"{scenario}: \u0394\u03c1 evolution (cycle {cycles[0]})")

    def update(i):
        facecolors = sm.to_rgba(delta_vals[i])
        poly.set_facecolors(facecolors)
        ax.set_title(f"{scenario}: \u0394\u03c1 evolution (cycle {cycles[i]})")
        return (poly,)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / scenario / "density_evolution.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def make_comparison_gif(cycles, mewing_delta, mouth_delta, face_verts,
                         mid_mask, lat_mask, vmin, vmax):
    """Render side-by-side comparison GIF."""
    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    sm.set_array([])

    poly1 = _add_surface(ax1, face_verts, mid_mask, lat_mask, sm, mewing_delta[0])
    poly2 = _add_surface(ax2, face_verts, mid_mask, lat_mask, sm, mouth_delta[0])

    for ax, label in ((ax1, "mewing"), (ax2, "mouth_breathing")):
        _setup_axes(ax, face_verts)
        _add_anatomical_labels(ax, face_verts)
        ax.set_title(label, fontsize=10)

    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.75,
                        label="\u0394\u03c1 from baseline")
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f"Cranio FEA remodeling comparison \u2014 cycle {cycles[0]}",
                 fontsize=12)

    def update(i):
        poly1.set_facecolors(sm.to_rgba(mewing_delta[i]))
        poly2.set_facecolors(sm.to_rgba(mouth_delta[i]))
        fig.suptitle(f"Cranio FEA remodeling comparison \u2014 cycle {cycles[i]}",
                     fontsize=12)
        return (poly1, poly2)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / "comparison_animation.gif"
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    face_verts, face_eids, mid_mask, lat_mask = load_oral_surface()
    print(f"Oral surface: {len(face_eids)} faces "
          f"({mid_mask.sum()} suture-mid, {lat_mask.sum()} suture-lat)")

    cycles_m, snaps_m = load_snapshots("mewing")
    cycles_b, snaps_b = load_snapshots("mouth_breathing")

    if cycles_m != cycles_b:
        raise SystemExit(f"Cycle mismatch: mewing={cycles_m} mouth_breathing={cycles_b}")

    vals_m = density_series_by_eids(face_eids, snaps_m)   # (n_frames, n_faces)
    vals_b = density_series_by_eids(face_eids, snaps_b)

    delta_m = delta_from_baseline(vals_m)
    delta_b = delta_from_baseline(vals_b)

    all_delta = np.concatenate([delta_m.ravel(), delta_b.ravel()])
    vmax = float(np.max(np.abs(all_delta)))
    if vmax == 0.0:
        vmax = 1e-9
    vmin = -vmax

    out_m = make_single_scenario_gif("mewing", cycles_m, delta_m, face_verts,
                                      mid_mask, lat_mask, vmin, vmax)
    out_b = make_single_scenario_gif("mouth_breathing", cycles_b, delta_b, face_verts,
                                      mid_mask, lat_mask, vmin, vmax)
    out_c = make_comparison_gif(cycles_m, delta_m, delta_b, face_verts,
                                 mid_mask, lat_mask, vmin, vmax)

    print(f"Wrote {out_m}")
    print(f"Wrote {out_b}")
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()

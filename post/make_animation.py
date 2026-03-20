#!/usr/bin/env python3
"""Create remodeling density evolution animations.

Outputs:
- results/mewing/density_evolution.gif
- results/mouth_breathing/density_evolution.gif
- results/comparison_animation.gif
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCENARIOS = ("mewing", "mouth_breathing")


def load_mesh_centers():
    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    nodes = {int(k): v for k, v in mesh["nodes"].items()}
    elems = {int(k): v for k, v in mesh["elements"].items()}
    sets = mesh.get("sets", {})

    s_mid = set(sets.get("ESET_SUTURE_MID", []))
    s_lat = set(sets.get("ESET_SUTURE_LAT", []))

    centers = {}
    for eid, conn in elems.items():
        pts = [nodes[n] for n in conn]
        centers[eid] = (
            sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts),
            sum(p[2] for p in pts) / len(pts),
        )

    order = sorted(centers.keys())
    xs = [centers[eid][0] for eid in order]
    ys = [centers[eid][1] for eid in order]
    zs = [centers[eid][2] for eid in order]
    mask_suture_mid = [eid in s_mid for eid in order]
    mask_suture_lat = [eid in s_lat for eid in order]
    return order, xs, ys, zs, mask_suture_mid, mask_suture_lat


def _overlay_sutures(ax, xs, ys, zs, mask_mid, mask_lat):
    """Draw suture elements as hollow circle markers on top of density scatter."""
    mid_x = [x for x, m in zip(xs, mask_mid) if m]
    mid_y = [y for y, m in zip(ys, mask_mid) if m]
    mid_z = [z for z, m in zip(zs, mask_mid) if m]
    lat_x = [x for x, m in zip(xs, mask_lat) if m]
    lat_y = [y for y, m in zip(ys, mask_lat) if m]
    lat_z = [z for z, m in zip(zs, mask_lat) if m]
    if mid_x:
        ax.scatter(mid_x, mid_y, mid_z, facecolors="none", edgecolors="black",
                   s=12, linewidths=0.6, marker="o", zorder=6, label="suture mid")
    if lat_x:
        ax.scatter(lat_x, lat_y, lat_z, facecolors="none", edgecolors="dimgray",
                   s=12, linewidths=0.6, marker="o", zorder=6, label="suture lat")


def _add_anatomical_labels(ax, xs, ys, zs):
    """Place anterior/posterior labels at mesh extents."""
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_mid = (min(zs) + max(zs)) / 2.0
    ax.text(x_max * 0.95, 0.0, z_mid, "Anterior", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)
    ax.text(x_min * 0.3, y_max * 0.85, z_mid, "Post. L", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)
    ax.text(x_min * 0.3, y_min * 0.85, z_mid, "Post. R", fontsize=7,
            ha="center", va="bottom", color="0.3", zorder=10)


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


def density_series_by_order(order, snap_dicts):
    return [[d.get(eid, 1.0) for eid in order] for d in snap_dicts]


def delta_from_baseline(values):
    baseline = values[0]
    return [[frame[i] - baseline[i] for i in range(len(frame))] for frame in values]


def make_single_scenario_gif(scenario, cycles, values, xs, ys, zs,
                             vmin, vmax, mask_mid, mask_lat):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    pt_size = max(4, min(25, 1400 // max(1, len(xs))))
    sc = ax.scatter(xs, ys, zs, c=values[0], cmap="RdBu_r", s=pt_size,
                    vmin=vmin, vmax=vmax)
    _overlay_sutures(ax, xs, ys, zs, mask_mid, mask_lat)
    _add_anatomical_labels(ax, xs, ys, zs)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75,
                        label="delta density ratio (\u0394rho from baseline)")
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=85, azim=0)

    def update(i):
        sc.set_array(values[i])
        ax.set_title(f"{scenario}: \u0394rho evolution (cycle {cycles[i]})")
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / scenario / "density_evolution.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def make_comparison_gif(cycles, mewing_vals, mouth_vals, xs, ys, zs,
                        vmin, vmax, mask_mid, mask_lat):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    pt_size = max(4, min(25, 1400 // max(1, len(xs))))
    sc1 = ax1.scatter(xs, ys, zs, c=mewing_vals[0], cmap="RdBu_r",
                      s=pt_size, vmin=vmin, vmax=vmax)
    sc2 = ax2.scatter(xs, ys, zs, c=mouth_vals[0], cmap="RdBu_r",
                      s=pt_size, vmin=vmin, vmax=vmax)
    fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.75,
                 label="delta density ratio (\u0394rho from baseline)")

    for ax, label in ((ax1, "mewing"), (ax2, "mouth_breathing")):
        _overlay_sutures(ax, xs, ys, zs, mask_mid, mask_lat)
        _add_anatomical_labels(ax, xs, ys, zs)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=85, azim=0)
        ax.set_title(label)

    def update(i):
        sc1.set_array(mewing_vals[i])
        sc2.set_array(mouth_vals[i])
        fig.suptitle(f"Cranio FEA remodeling comparison \u2014 cycle {cycles[i]}",
                     fontsize=12)
        return (sc1, sc2)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / "comparison_animation.gif"
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def main():
    order, xs, ys, zs, mask_mid, mask_lat = load_mesh_centers()

    cycles_m, snaps_m = load_snapshots("mewing")
    cycles_b, snaps_b = load_snapshots("mouth_breathing")

    if cycles_m != cycles_b:
        raise SystemExit(f"Cycle mismatch: mewing={cycles_m} mouth_breathing={cycles_b}")

    vals_m = density_series_by_order(order, snaps_m)
    vals_b = density_series_by_order(order, snaps_b)

    delta_m = delta_from_baseline(vals_m)
    delta_b = delta_from_baseline(vals_b)

    all_delta = [v for frame in (delta_m + delta_b) for v in frame]
    vmax = max(abs(min(all_delta)), abs(max(all_delta)))
    if vmax == 0.0:
        vmax = 1e-9
    vmin = -vmax

    out_m = make_single_scenario_gif("mewing", cycles_m, delta_m, xs, ys, zs,
                                     vmin, vmax, mask_mid, mask_lat)
    out_b = make_single_scenario_gif("mouth_breathing", cycles_b, delta_b, xs, ys, zs,
                                     vmin, vmax, mask_mid, mask_lat)
    out_c = make_comparison_gif(cycles_m, delta_m, delta_b, xs, ys, zs,
                                vmin, vmax, mask_mid, mask_lat)

    print(f"Wrote {out_m}")
    print(f"Wrote {out_b}")
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()

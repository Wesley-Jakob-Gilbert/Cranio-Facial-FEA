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
    suture = s_mid | s_lat

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
    mask_suture = [eid in suture for eid in order]
    return order, xs, ys, zs, mask_suture


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


def make_single_scenario_gif(scenario, cycles, values, xs, ys, zs, vmin, vmax):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(xs, ys, zs, c=values[0], cmap="viridis", s=22, vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, label="density ratio (rho)")
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=20, azim=-62)

    def update(i):
        sc.set_array(values[i])
        ax.set_title(f"{scenario}: density evolution (cycle {cycles[i]})")
        return (sc,)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / scenario / "density_evolution.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def make_comparison_gif(cycles, mewing_vals, mouth_vals, xs, ys, zs, vmin, vmax):
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")

    sc1 = ax1.scatter(xs, ys, zs, c=mewing_vals[0], cmap="viridis", s=20, vmin=vmin, vmax=vmax)
    sc2 = ax2.scatter(xs, ys, zs, c=mouth_vals[0], cmap="viridis", s=20, vmin=vmin, vmax=vmax)
    fig.colorbar(sc2, ax=[ax1, ax2], shrink=0.75, label="density ratio (rho)")

    for ax, label in ((ax1, "mewing"), (ax2, "mouth_breathing")):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.view_init(elev=20, azim=-62)
        ax.set_title(label)

    def update(i):
        sc1.set_array(mewing_vals[i])
        sc2.set_array(mouth_vals[i])
        fig.suptitle(f"Cranio FEA remodeling comparison — cycle {cycles[i]}", fontsize=12)
        return (sc1, sc2)

    anim = FuncAnimation(fig, update, frames=len(cycles), interval=420, blit=False)
    out = RESULTS / "comparison_animation.gif"
    anim.save(out, writer=PillowWriter(fps=2))
    plt.close(fig)
    return out


def main():
    order, xs, ys, zs, _mask_suture = load_mesh_centers()

    cycles_m, snaps_m = load_snapshots("mewing")
    cycles_b, snaps_b = load_snapshots("mouth_breathing")

    if cycles_m != cycles_b:
        raise SystemExit(f"Cycle mismatch: mewing={cycles_m} mouth_breathing={cycles_b}")

    vals_m = density_series_by_order(order, snaps_m)
    vals_b = density_series_by_order(order, snaps_b)

    all_vals = [v for frame in (vals_m + vals_b) for v in frame]
    vmin, vmax = min(all_vals), max(all_vals)

    out_m = make_single_scenario_gif("mewing", cycles_m, vals_m, xs, ys, zs, vmin, vmax)
    out_b = make_single_scenario_gif("mouth_breathing", cycles_b, vals_b, xs, ys, zs, vmin, vmax)
    out_c = make_comparison_gif(cycles_m, vals_m, vals_b, xs, ys, zs, vmin, vmax)

    print(f"Wrote {out_m}")
    print(f"Wrote {out_b}")
    print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()

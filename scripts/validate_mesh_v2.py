#!/usr/bin/env python3
"""Visual validation of template_maxilla_v2 mesh geometry (N-13).

Produces three matplotlib views saved to results/mesh_v2_validation.png:
  - Occlusal view  (looking up at the oral surface, x-y plane)
  - Lateral view   (x-z plane, right side)
  - Anterior view  (y-z plane, from the front)

Nodes are coloured by z-coordinate (height) so the palatal vault,
alveolar ridges, and bone thickness are immediately visible.
Suture zones are overlaid with distinct markers.

Usage:
    python3 scripts/validate_mesh_v2.py

Requires: matplotlib (pip install matplotlib)
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except ImportError:
    (ROOT / "results" / "MESH_VALIDATION_TODO.txt").write_text(
        "Install matplotlib to generate mesh validation plots.\n"
    )
    print("matplotlib not installed — wrote MESH_VALIDATION_TODO.txt")
    raise SystemExit(0)

MESH_PATH = ROOT / "mesh" / "mesh_data.json"
OUT = ROOT / "results" / "mesh_v2_validation.png"

if not MESH_PATH.exists():
    print(f"mesh_data.json not found — run: python3 mesh/build_mesh.py first")
    raise SystemExit(1)

mesh = json.loads(MESH_PATH.read_text())
profile = mesh.get("profile", "unknown")

if "v2" not in profile:
    print(f"Warning: profile is '{profile}', not v2. Validation still runs but may not be meaningful.")

nodes = {int(k): v for k, v in mesh["nodes"].items()}
sets  = mesh["sets"]

suture_mid_nids = set(sets.get("ESET_SUTURE_MID", []))
suture_lat_nids = set(sets.get("ESET_SUTURE_LAT", []))
fixed_nids      = set(sets.get("NSET_FIXED", []))
palate_nids     = set(sets.get("NSET_PALATE", []))

# For element-centred coloring, compute centroids for suture elements
elems = {int(k): v for k, v in mesh["elements"].items()}


def elem_centroid(conn):
    xs = [nodes[n][0] for n in conn]
    ys = [nodes[n][1] for n in conn]
    zs = [nodes[n][2] for n in conn]
    return sum(xs) / 8, sum(ys) / 8, sum(zs) / 8


suture_mid_eids = set(sets.get("ESET_SUTURE_MID", []))
suture_lat_eids = set(sets.get("ESET_SUTURE_LAT", []))

smid_cx, smid_cy, smid_cz = [], [], []
for eid in suture_mid_eids:
    cx, cy, cz = elem_centroid(elems[eid])
    smid_cx.append(cx); smid_cy.append(cy); smid_cz.append(cz)

slat_cx, slat_cy, slat_cz = [], [], []
for eid in suture_lat_eids:
    cx, cy, cz = elem_centroid(elems[eid])
    slat_cx.append(cx); slat_cy.append(cy); slat_cz.append(cz)

# Node arrays
all_x = [nodes[n][0] for n in sorted(nodes)]
all_y = [nodes[n][1] for n in sorted(nodes)]
all_z = [nodes[n][2] for n in sorted(nodes)]

# Color by z
norm = mcolors.Normalize(vmin=min(all_z), vmax=max(all_z))
cmap = cm.RdYlBu_r

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"Mesh Validation — {profile}\n"
             f"{len(nodes)} nodes  |  {len(elems)} elements  |  "
             f"BONE:{len(elems)-len(suture_mid_eids)-len(slat_cx)}  "
             f"SUTURE_MID:{len(suture_mid_eids)}  SUTURE_LAT:{len(suture_lat_eids)}",
             fontsize=11)

dot_size = 1.5

# ── Occlusal view (x-y, looking superiorly at oral surface) ──────────────────
ax = axes[0]
sc = ax.scatter(all_x, all_y, c=all_z, cmap=cmap, norm=norm, s=dot_size, alpha=0.6)
ax.scatter(smid_cx, smid_cy, color="red",    s=8, marker="x", label="SUTURE_MID", zorder=5)
ax.scatter(slat_cx, slat_cy, color="orange", s=8, marker="^", label="SUTURE_LAT", zorder=5)
ax.set_xlabel("x (AP, posterior→anterior) [m]")
ax.set_ylabel("y (mediolateral) [m]")
ax.set_title("Occlusal view (x-y)\ncolour = z-height")
ax.set_aspect("equal")
ax.legend(fontsize=7, markerscale=2)
plt.colorbar(sc, ax=ax, label="z [m]", fraction=0.04)

# ── Lateral view (x-z, right side of arch) ───────────────────────────────────
ax = axes[1]
sc2 = ax.scatter(all_x, all_z, c=all_y, cmap="PiYG", s=dot_size, alpha=0.6)
ax.scatter(smid_cx, smid_cz, color="red",    s=8, marker="x", zorder=5)
ax.scatter(slat_cx, slat_cz, color="orange", s=8, marker="^", zorder=5)
ax.set_xlabel("x (AP) [m]")
ax.set_ylabel("z (superior) [m]")
ax.set_title("Lateral view (x-z)\ncolour = y-position")
ax.set_aspect("equal")
plt.colorbar(sc2, ax=ax, label="y [m]", fraction=0.04)

# ── Anterior view (y-z, from front) ──────────────────────────────────────────
ax = axes[2]
sc3 = ax.scatter(all_y, all_z, c=all_x, cmap="viridis", s=dot_size, alpha=0.6)
ax.scatter(smid_cy, smid_cz, color="red",    s=8, marker="x", label="SUTURE_MID", zorder=5)
ax.scatter(slat_cy, slat_cz, color="orange", s=8, marker="^", label="SUTURE_LAT", zorder=5)
ax.set_xlabel("y (mediolateral) [m]")
ax.set_ylabel("z (superior) [m]")
ax.set_title("Anterior view (y-z)\ncolour = x depth")
ax.set_aspect("equal")
ax.legend(fontsize=7, markerscale=2)
plt.colorbar(sc3, ax=ax, label="x [m]", fraction=0.04)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Wrote {OUT}")

# ── text summary ──────────────────────────────────────────────────────────────
xs, ys, zs = all_x, all_y, all_z
print(f"\nBounding box:")
print(f"  x: [{min(xs)*1000:.1f}, {max(xs)*1000:.1f}] mm  (AP depth {(max(xs)-min(xs))*1000:.1f} mm)")
print(f"  y: [{min(ys)*1000:.1f}, {max(ys)*1000:.1f}] mm  (total width {(max(ys)-min(ys))*1000:.1f} mm)")
print(f"  z: [{min(zs)*1000:.1f}, {max(zs)*1000:.1f}] mm  (height range {(max(zs)-min(zs))*1000:.1f} mm)")
print(f"\nFixed face nodes (posterior arms): {len(sets.get('NSET_FIXED', []))}")
print(f"Palate nodes (oral surface):        {len(sets.get('NSET_PALATE', []))}")
print(f"Muscle nodes (nasal, post-lat):     {len(sets.get('NSET_MUSCLE_ALL', []))}")

#!/usr/bin/env python3
"""Create a first toy morphology animation (north-star prototype).

Method:
- use medium_kpa nodal displacement field as a direction field
- scale displacements over pseudo-time steps to emulate gradual morphing
- visualize as 3D node cloud evolution

This is NOT a validated growth law; it's an intuitive prototype animation.
"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
mesh = json.loads((ROOT / 'mesh' / 'mesh_data.json').read_text())
nodes = {int(k): v for k,v in mesh['nodes'].items()}

case = 'medium_kpa'
node_csv = ROOT / 'results' / case / 'node_u.csv'
if not node_csv.exists():
    raise SystemExit('Run solves + extract_fields first')

order = sorted(nodes.keys())
coords = np.array([nodes[n] for n in order], dtype=float)
U = np.zeros_like(coords)

with node_csv.open('r', encoding='utf-8') as f:
    r = csv.DictReader(f)
    d = {int(row['node_id']):(float(row['ux']), float(row['uy']), float(row['uz'])) for row in r}

for i,nid in enumerate(order):
    ux,uy,uz = d.get(nid, (0.0,0.0,0.0))
    U[i] = [ux,uy,uz]

# amplify for visibility in toy animation
def scale_at_t(t):
    # t in [0,1]
    return 0.0 + 2500.0 * t

fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(coords[:,0], coords[:,1], coords[:,2], s=8, c='steelblue', alpha=0.8)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
ax.set_title('Toy morphology evolution (medium load direction field)')
ax.view_init(elev=20, azim=-65)

mins = coords.min(axis=0)
maxs = coords.max(axis=0)
pad = (maxs - mins) * 0.1
ax.set_xlim(mins[0]-pad[0], maxs[0]+pad[0])
ax.set_ylim(mins[1]-pad[1], maxs[1]+pad[1])
ax.set_zlim(mins[2]-pad[2], maxs[2]+pad[2])

frames = 48

def update(frame):
    t = frame/(frames-1)
    s = scale_at_t(t)
    pts = coords + s*U
    sc._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
    ax.set_title(f'Toy morphology evolution (t={t:.2f})')
    return (sc,)

anim = FuncAnimation(fig, update, frames=frames, interval=120, blit=False)
out = ROOT / 'results' / 'toy_morphology_evolution.gif'
anim.save(out, writer=PillowWriter(fps=10))
print(f'Wrote {out}')

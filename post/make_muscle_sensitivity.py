#!/usr/bin/env python3
"""Plot medium-pressure response sensitivity to muscle resultant magnitude."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
metrics = ROOT / "results" / "metrics_summary.csv"

rows = []
with metrics.open("r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        if row.get("status") != "ok":
            continue
        kpa = float(row["tongue_pressure_kpa"])
        muscle = float(row.get("muscle_force_n", 0.0))
        if abs(kpa - 2.0) > 1e-9:
            continue
        rows.append((muscle, float(row["max_displacement_m"]), float(row["peak_von_mises_pa"]), row["case"]))

rows.sort(key=lambda x: x[0])
if len(rows) < 2:
    raise SystemExit("Not enough medium-kPa rows found for sensitivity plot")

x = [r[0] for r in rows]
u = [r[1] for r in rows]
vm = [r[2] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(10,4))
ax[0].plot(x, u, marker='o')
ax[0].set_xlabel('Muscle resultant force (N)')
ax[0].set_ylabel('Max displacement (m)')
ax[0].set_title('Medium tongue pressure: displacement sensitivity')

ax[1].plot(x, vm, marker='o')
ax[1].set_xlabel('Muscle resultant force (N)')
ax[1].set_ylabel('Peak von Mises (Pa)')
ax[1].set_title('Medium tongue pressure: stress sensitivity')

fig.tight_layout()
out = ROOT / 'results' / 'muscle_sensitivity_medium_kpa.png'
fig.savefig(out, dpi=170)
print(f'Wrote {out}')

csv_out = ROOT / 'results' / 'muscle_sensitivity_medium_kpa.csv'
with csv_out.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['case','muscle_force_n','max_displacement_m','peak_von_mises_pa'])
    for muscle, uu, vv, case in rows:
        w.writerow([case, f'{muscle:.3f}', f'{uu:.6e}', f'{vv:.6e}'])
print(f'Wrote {csv_out}')

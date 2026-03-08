#!/usr/bin/env python3
"""Generate simple load-vs-response plots from metrics_summary.csv.

Default plot uses tongue-only (muscle_force_n == 0) cases.
"""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
metrics = ROOT / "results" / "metrics_summary.csv"
out_png = ROOT / "results" / "mvp_load_response.png"

rows = []
with metrics.open("r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if r.get("status") != "ok":
            continue
        if float(r.get("muscle_force_n", 0.0)) != 0.0:
            continue
        rows.append(
            (
                float(r["tongue_pressure_kpa"]),
                float(r["max_displacement_m"]),
                float(r["peak_von_mises_pa"]),
            )
        )

rows.sort(key=lambda x: x[0])

try:
    import matplotlib.pyplot as plt
except Exception:
    note = ROOT / "results" / "PLOTS_TODO.txt"
    note.write_text("matplotlib unavailable; install python3-matplotlib to generate plots\n")
    print(f"matplotlib not available. Wrote {note}")
    raise SystemExit(0)

if not rows:
    print("No valid tongue-only rows to plot.")
    raise SystemExit(0)

x = [r[0] for r in rows]
y1 = [r[1] for r in rows]
y2 = [r[2] for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(x, y1, marker="o")
ax[0].set_xlabel("Tongue pressure (kPa)")
ax[0].set_ylabel("Max displacement (m)")
ax[0].set_title("Tongue-only: load vs max displacement")

ax[1].plot(x, y2, marker="o")
ax[1].set_xlabel("Tongue pressure (kPa)")
ax[1].set_ylabel("Peak von Mises stress (Pa)")
ax[1].set_title("Tongue-only: load vs peak stress")

fig.tight_layout()
fig.savefig(out_png, dpi=150)
print(f"Wrote {out_png}")

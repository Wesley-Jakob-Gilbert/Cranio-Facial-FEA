#!/usr/bin/env python3
"""Generate a 2x2 sensitivity dashboard from parameter_sensitivity.csv.

Reads results/parameter_sensitivity.csv (produced by scripts/parameter_sensitivity.py)
and creates a four-panel figure showing how each swept remodeling parameter
affects density metrics and zone fractions.

Output: results/sensitivity_dashboard.png
"""
from __future__ import annotations

import csv
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "parameter_sensitivity.csv"
OUT_PNG = ROOT / "results" / "sensitivity_dashboard.png"

# Baseline values — must match configs/remodeling.yaml defaults
DEFAULTS = OrderedDict([
    ("psi_ref_pa",       5.0),
    ("alpha_apposition", 0.001),
    ("lazy_zone_s",      0.15),
    ("n_power",          2.0),
])

PARAM_LABELS = {
    "psi_ref_pa":       "psi_ref (Pa)",
    "alpha_apposition": "alpha_apposition",
    "lazy_zone_s":      "lazy_zone_s",
    "n_power":          "n_power",
}


def load_data() -> dict[str, list[dict]]:
    """Load CSV and group rows by parameter name."""
    groups: dict[str, list[dict]] = {p: [] for p in DEFAULTS}
    with CSV_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            param = row["parameter"]
            if param in groups:
                groups[param].append(row)
    # Sort each group by sweep value
    for param in groups:
        groups[param].sort(key=lambda r: float(r["value"]))
    return groups


def main() -> None:
    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found. Run scripts/parameter_sensitivity.py first.")
        sys.exit(1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not available. Install with: pip install matplotlib")
        sys.exit(1)

    groups = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Remodeling Parameter Sensitivity ({3}-cycle OFAT, mewing scenario)",
        fontsize=13,
        fontweight="bold",
        y=0.97,
    )

    for ax, (param, default_val) in zip(axes.flat, DEFAULTS.items()):
        rows = groups.get(param, [])
        if not rows:
            ax.set_title(PARAM_LABELS.get(param, param))
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            continue

        xvals = [float(r["value"]) for r in rows]
        mean_rho = [float(r["mean_rho"]) for r in rows]
        rho_spread = [float(r["rho_spread"]) for r in rows]
        app_frac = [float(r["apposition_frac"]) for r in rows]
        lazy_frac = [float(r["lazy_frac"]) for r in rows]

        # Left Y-axis: density metrics
        color_rho = "#1f77b4"
        color_spread = "#2ca02c"
        ax.plot(xvals, mean_rho, "o-", color=color_rho, linewidth=1.8,
                markersize=5, label="mean rho")
        ax.plot(xvals, rho_spread, "s--", color=color_spread, linewidth=1.4,
                markersize=4, label="rho spread")
        ax.set_xlabel(PARAM_LABELS.get(param, param))
        ax.set_ylabel("Density", color=color_rho)
        ax.tick_params(axis="y", labelcolor=color_rho)

        # Baseline vertical line
        ax.axvline(default_val, color="gray", linestyle=":", linewidth=1.0,
                   alpha=0.7, label=f"baseline ({default_val})")

        # Right Y-axis: zone fractions
        ax2 = ax.twinx()
        color_app = "#d62728"
        color_lazy = "#7f7f7f"
        ax2.plot(xvals, app_frac, "^-", color=color_app, linewidth=1.4,
                 markersize=4, label="apposition frac")
        ax2.plot(xvals, lazy_frac, "v-", color=color_lazy, linewidth=1.4,
                 markersize=4, label="lazy frac")
        ax2.set_ylabel("Fraction", color=color_app)
        ax2.tick_params(axis="y", labelcolor=color_app)
        ax2.set_ylim(-0.05, 1.05)

        ax.set_title(PARAM_LABELS.get(param, param))

        # Combined legend (both axes)
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2,
                  fontsize=7, loc="best", framealpha=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()

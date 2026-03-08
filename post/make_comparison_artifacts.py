#!/usr/bin/env python3
"""Generate comparative artifacts from metrics_summary.csv.

Outputs:
- results/case_comparison_table.csv
- results/case_pairwise_deltas.csv
- results/linearity_check.csv                (tongue-only subset)
- results/muscle_effect_summary.csv           (paired tongue vs tongue+muscle)
- results/normalized_response.png             (tongue-only)
- results/linearity_check.png                (tongue-only)
- results/muscle_effect.png                  (paired deltas)
"""
from __future__ import annotations
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = RESULTS / "metrics_summary.csv"


def write_csv(path: Path, header: list[str], rows: list[list]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def read_metrics() -> list[dict]:
    rows = []
    with METRICS.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("status") != "ok":
                continue
            rows.append(
                {
                    "case": row["case"],
                    "kpa": float(row["tongue_pressure_kpa"]),
                    "muscle_n": float(row.get("muscle_force_n", 0.0)),
                    "max_u": float(row["max_displacement_m"]),
                    "peak_vm": float(row["peak_von_mises_pa"]),
                }
            )
    rows.sort(key=lambda x: (x["muscle_n"], x["kpa"], x["case"]))
    return rows


def main():
    if not METRICS.exists():
        raise SystemExit(f"Missing metrics file: {METRICS}")

    data = read_metrics()
    if len(data) < 2:
        raise SystemExit("Need at least 2 valid cases to compare")

    # Global table
    low_any = min(data, key=lambda d: d["max_u"])
    comparison_rows = []
    for d in data:
        comparison_rows.append(
            [
                d["case"],
                f"{d['kpa']:.3f}",
                f"{d['muscle_n']:.3f}",
                f"{d['max_u']:.6e}",
                f"{d['peak_vm']:.6e}",
                f"{d['max_u'] / low_any['max_u']:.6f}",
                f"{d['peak_vm'] / low_any['peak_vm']:.6f}",
                f"{d['max_u'] / max(d['kpa'], 1e-12):.6e}",
                f"{d['peak_vm'] / max(d['kpa'], 1e-12):.6e}",
            ]
        )

    write_csv(
        RESULTS / "case_comparison_table.csv",
        [
            "case",
            "tongue_pressure_kpa",
            "muscle_force_n",
            "max_displacement_m",
            "peak_von_mises_pa",
            "u_vs_global_low_ratio",
            "vm_vs_global_low_ratio",
            "u_per_kpa",
            "vm_per_kpa",
        ],
        comparison_rows,
    )

    # Pairwise deltas within same muscle setting
    pair_rows = []
    for muscle_n in sorted(set(d["muscle_n"] for d in data)):
        subset = [d for d in data if d["muscle_n"] == muscle_n]
        subset.sort(key=lambda x: x["kpa"])
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                pair_rows.append(
                    [
                        f"{muscle_n:.3f}",
                        a["case"],
                        b["case"],
                        f"{b['kpa'] - a['kpa']:.3f}",
                        f"{(b['max_u'] - a['max_u']):.6e}",
                        f"{(b['peak_vm'] - a['peak_vm']):.6e}",
                        f"{(b['max_u'] / a['max_u']):.6f}",
                        f"{(b['peak_vm'] / a['peak_vm']):.6f}",
                    ]
                )

    write_csv(
        RESULTS / "case_pairwise_deltas.csv",
        [
            "muscle_force_n",
            "case_a",
            "case_b",
            "delta_pressure_kpa",
            "delta_max_displacement_m",
            "delta_peak_von_mises_pa",
            "u_ratio_b_over_a",
            "vm_ratio_b_over_a",
        ],
        pair_rows,
    )

    # Tongue-only subset linearity
    tongue_only = [d for d in data if d["muscle_n"] == 0.0]
    tongue_only.sort(key=lambda x: x["kpa"])
    if len(tongue_only) >= 2:
        low = tongue_only[0]
        lin_rows = []
        for d in tongue_only:
            p_scale = d["kpa"] / low["kpa"]
            u_pred = low["max_u"] * p_scale
            vm_pred = low["peak_vm"] * p_scale
            lin_rows.append(
                [
                    d["case"],
                    f"{d['kpa']:.3f}",
                    f"{d['max_u']:.6e}",
                    f"{u_pred:.6e}",
                    f"{(d['max_u'] - u_pred):.6e}",
                    f"{(d['max_u'] / u_pred):.6f}",
                    f"{d['peak_vm']:.6e}",
                    f"{vm_pred:.6e}",
                    f"{(d['peak_vm'] - vm_pred):.6e}",
                    f"{(d['peak_vm'] / vm_pred):.6f}",
                ]
            )

        write_csv(
            RESULTS / "linearity_check.csv",
            [
                "case",
                "tongue_pressure_kpa",
                "u_actual_m",
                "u_pred_linear_m",
                "u_residual_m",
                "u_actual_over_pred",
                "vm_actual_pa",
                "vm_pred_linear_pa",
                "vm_residual_pa",
                "vm_actual_over_pred",
            ],
            lin_rows,
        )

    # Muscle effect paired by pressure (muscle vs no-muscle)
    by_key = {}
    for d in data:
        by_key[(d["kpa"], d["muscle_n"])] = d

    muscle_levels = sorted(set(d["muscle_n"] for d in data if d["muscle_n"] > 0))
    muscle_rows = []
    for m in muscle_levels:
        for kpa in sorted(set(d["kpa"] for d in data)):
            base = by_key.get((kpa, 0.0))
            musc = by_key.get((kpa, m))
            if not base or not musc:
                continue
            muscle_rows.append(
                [
                    f"{kpa:.3f}",
                    f"{m:.3f}",
                    f"{base['max_u']:.6e}",
                    f"{musc['max_u']:.6e}",
                    f"{(musc['max_u'] - base['max_u']):.6e}",
                    f"{(musc['max_u'] / base['max_u']):.6f}",
                    f"{base['peak_vm']:.6e}",
                    f"{musc['peak_vm']:.6e}",
                    f"{(musc['peak_vm'] - base['peak_vm']):.6e}",
                    f"{(musc['peak_vm'] / base['peak_vm']):.6f}",
                ]
            )

    write_csv(
        RESULTS / "muscle_effect_summary.csv",
        [
            "tongue_pressure_kpa",
            "muscle_force_n",
            "u_tongue_only_m",
            "u_tongue_plus_muscle_m",
            "u_delta_m",
            "u_ratio_plus_over_only",
            "vm_tongue_only_pa",
            "vm_tongue_plus_muscle_pa",
            "vm_delta_pa",
            "vm_ratio_plus_over_only",
        ],
        muscle_rows,
    )

    # Optional figures
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable; CSV artifacts generated only")
        return

    if tongue_only:
        low = tongue_only[0]
        x = [d["kpa"] for d in tongue_only]
        u_norm = [d["max_u"] / low["max_u"] for d in tongue_only]
        vm_norm = [d["peak_vm"] / low["peak_vm"] for d in tongue_only]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(x, u_norm, marker="o", label="u/u_low")
        ax[0].plot(x, [xx / low["kpa"] for xx in x], "--", label="ideal linear")
        ax[0].set_xlabel("Tongue pressure (kPa)")
        ax[0].set_ylabel("Normalized displacement")
        ax[0].set_title("Tongue-only displacement normalization")
        ax[0].legend()

        ax[1].plot(x, vm_norm, marker="o", label="vm/vm_low")
        ax[1].plot(x, [xx / low["kpa"] for xx in x], "--", label="ideal linear")
        ax[1].set_xlabel("Tongue pressure (kPa)")
        ax[1].set_ylabel("Normalized stress")
        ax[1].set_title("Tongue-only stress normalization")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(RESULTS / "normalized_response.png", dpi=170)
        plt.close(fig)

        u_ratio = [d["max_u"] / (low["max_u"] * (d["kpa"] / low["kpa"])) for d in tongue_only]
        vm_ratio = [d["peak_vm"] / (low["peak_vm"] * (d["kpa"] / low["kpa"])) for d in tongue_only]

        plt.figure(figsize=(7, 4))
        plt.plot(x, u_ratio, marker="o", label="u_actual/u_pred")
        plt.plot(x, vm_ratio, marker="o", label="vm_actual/vm_pred")
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Tongue pressure (kPa)")
        plt.ylabel("Ratio to linear prediction")
        plt.title("Tongue-only linearity check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "linearity_check.png", dpi=170)
        plt.close()

    # muscle effect figure
    if muscle_rows:
        x = [float(r[0]) for r in muscle_rows]
        u_ratio = [float(r[5]) for r in muscle_rows]
        vm_ratio = [float(r[9]) for r in muscle_rows]
        plt.figure(figsize=(7, 4))
        plt.plot(x, u_ratio, marker="o", label="u plus/only")
        plt.plot(x, vm_ratio, marker="o", label="vm plus/only")
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Tongue pressure (kPa)")
        plt.ylabel("Ratio")
        plt.title("Effect of simplified jaw-muscle load")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "muscle_effect.png", dpi=170)
        plt.close()

    print("Wrote comparison CSVs and PNG artifacts")


if __name__ == "__main__":
    main()

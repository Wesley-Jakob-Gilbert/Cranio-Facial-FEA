#!/usr/bin/env python3
"""Mesh convergence study for the cranio_FEA pipeline.

Compares peak von Mises stress and max displacement between the current
(coarse) mesh and a refined mesh at ~4x element count.  Both meshes are
solved for the 'mewing' load case (2 kPa tongue pressure, no muscle).

The script is safe: it backs up and restores geometry.json, mesh_data.json,
and elem_density.json even if an error occurs.

Convergence criterion:
    ratio = |metric_refined - metric_coarse| / metric_refined
    < 5%  : mesh is adequate
    < 10% : acceptable
    > 10% : refinement needed
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_CFG = ROOT / "configs" / "geometry.json"
MESH_DATA = ROOT / "mesh" / "mesh_data.json"
DENSITY_FILE = ROOT / "solver" / "elem_density.json"
RESULTS_DIR = ROOT / "results"
SOLVER_DIR = ROOT / "solver"

# Backup paths
GEOMETRY_BAK = GEOMETRY_CFG.with_suffix(".json.convergence_bak")
MESH_BAK = MESH_DATA.with_suffix(".json.convergence_bak")
DENSITY_BAK = DENSITY_FILE.with_suffix(".json.convergence_bak")

# Convergence work directories
COARSE_WORK = RESULTS_DIR / "_convergence_coarse"
REFINED_WORK = RESULTS_DIR / "_convergence_refined"

CASE_NAME = "mewing"


def backup_files():
    """Back up original files that will be modified."""
    print("[backup] Saving original files...")
    shutil.copy2(GEOMETRY_CFG, GEOMETRY_BAK)
    shutil.copy2(MESH_DATA, MESH_BAK)
    if DENSITY_FILE.exists():
        shutil.copy2(DENSITY_FILE, DENSITY_BAK)
    print("[backup] Done.")


def restore_files():
    """Restore original files from backup."""
    print("[restore] Restoring original files...")
    if GEOMETRY_BAK.exists():
        shutil.copy2(GEOMETRY_BAK, GEOMETRY_CFG)
        GEOMETRY_BAK.unlink()
        print(f"  Restored {GEOMETRY_CFG}")
    if MESH_BAK.exists():
        shutil.copy2(MESH_BAK, MESH_DATA)
        MESH_BAK.unlink()
        print(f"  Restored {MESH_DATA}")
    if DENSITY_BAK.exists():
        shutil.copy2(DENSITY_BAK, DENSITY_FILE)
        DENSITY_BAK.unlink()
        print(f"  Restored {DENSITY_FILE}")
    elif DENSITY_FILE.exists():
        # Density file was created during the study but didn't exist before
        # Actually, if backup didn't exist, the original had one -- leave it
        pass
    print("[restore] Done.")


def remove_density_file():
    """Temporarily remove solver/elem_density.json so make_inp.py uses baseline mode."""
    if DENSITY_FILE.exists():
        os.remove(DENSITY_FILE)


def run_cmd(cmd: list[str], label: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a subprocess with error handling."""
    print(f"[{label}] Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[{label}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(f"  stdout: {result.stdout[-500:]}" if result.stdout else "")
        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
        raise RuntimeError(f"{label} failed")
    print(f"[{label}] OK ({elapsed:.1f}s)")
    return result


def generate_mesh(label: str):
    """Run build_mesh.py and return element/node counts."""
    result = run_cmd([sys.executable, str(ROOT / "mesh" / "build_mesh.py")], f"mesh-{label}")
    # Parse output for counts
    for line in result.stdout.splitlines():
        if "Nodes:" in line and "Elements:" in line:
            parts = line.split(",")
            n_nodes = int(parts[0].split(":")[1].strip())
            n_elems = int(parts[1].split(":")[1].strip())
            return n_nodes, n_elems
    return 0, 0


def generate_inp_mewing_only(label: str):
    """Run make_inp.py (generates all cases, but we only use mewing)."""
    run_cmd([sys.executable, str(SOLVER_DIR / "make_inp.py")], f"inp-{label}")


def solve_case(work_dir: Path, label: str, timeout: int = 300):
    """Copy .inp to work_dir and run ccx."""
    work_dir.mkdir(parents=True, exist_ok=True)
    inp_src = SOLVER_DIR / f"{CASE_NAME}.inp"
    inp_dst = work_dir / f"{CASE_NAME}.inp"
    shutil.copy2(inp_src, inp_dst)

    print(f"[solve-{label}] Running ccx {CASE_NAME} in {work_dir}...")
    t0 = time.time()
    result = subprocess.run(
        ["ccx", CASE_NAME],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(work_dir),
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[solve-{label}] FAILED (exit {result.returncode}, {elapsed:.1f}s)")
        print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
        # Check for log
        log = work_dir / f"{CASE_NAME}.sta"
        if log.exists():
            print(f"  .sta file tail:")
            print("  " + log.read_text()[-500:])
        raise RuntimeError(f"ccx solve failed for {label}")
    print(f"[solve-{label}] OK ({elapsed:.1f}s)")


def extract_metrics_from_dat(dat_path: Path) -> dict:
    """Parse a .dat file for peak von Mises stress and max displacement.

    Returns dict with keys: max_displacement_m, peak_von_mises_pa,
    n_disp_records, n_stress_records.
    """
    num_line_4 = re.compile(
        r"^\s*\d+\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)"
    )
    stress_line = re.compile(
        r"^\s*\d+\s+\d+\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+"
        r"([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+"
        r"([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)"
    )

    max_u = 0.0
    max_vm = 0.0
    n_disp = 0
    n_stress = 0

    for line in dat_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = num_line_4.match(line)
        if m:
            ux, uy, uz = map(float, m.groups())
            u = math.sqrt(ux * ux + uy * uy + uz * uz)
            if u > max_u:
                max_u = u
            n_disp += 1
            continue

        s = stress_line.match(line)
        if s:
            sxx, syy, szz, sxy, sxz, syz = map(float, s.groups())
            vm = math.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2)
            )
            if vm > max_vm:
                max_vm = vm
            n_stress += 1

    return {
        "max_displacement_m": max_u,
        "peak_von_mises_pa": max_vm,
        "n_disp_records": n_disp,
        "n_stress_records": n_stress,
    }


def main():
    print("=" * 70)
    print("  MESH CONVERGENCE STUDY")
    print("  Case: mewing (2 kPa tongue pressure, baseline materials)")
    print("=" * 70)
    print()

    # Read current geometry config
    cfg = json.loads(GEOMETRY_CFG.read_text())
    coarse_divs = cfg.get("divisions_v2", cfg.get("divisions", {}))
    nx_c = coarse_divs.get("nx", 44)
    ny_c = coarse_divs.get("ny", 12)
    nz_c = coarse_divs.get("nz", 4)
    n_elems_coarse = nx_c * ny_c * nz_c
    print(f"Coarse mesh: {nx_c} x {ny_c} x {nz_c} = {n_elems_coarse} elements")

    # Refined mesh: double each division
    nx_r = nx_c * 2
    ny_r = ny_c * 2
    nz_r = nz_c * 2
    n_elems_refined_est = nx_r * ny_r * nz_r
    print(f"Refined mesh: {nx_r} x {ny_r} x {nz_r} = {n_elems_refined_est} elements (estimated)")
    print()

    backup_files()

    try:
        # ------------------------------------------------------------------
        # STEP 1: Solve coarse mesh (current mesh)
        # ------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("  STEP 1: Coarse mesh solve")
        print("=" * 50)

        # Remove density override so make_inp uses baseline 2-material mode
        remove_density_file()

        # Re-generate mesh (should match current since divisions unchanged)
        n_nodes_c, n_elems_c = generate_mesh("coarse")
        print(f"  Coarse mesh: {n_nodes_c} nodes, {n_elems_c} elements")

        # Generate .inp deck
        generate_inp_mewing_only("coarse")

        # Solve
        solve_case(COARSE_WORK, "coarse", timeout=300)

        # Extract metrics
        dat_c = COARSE_WORK / f"{CASE_NAME}.dat"
        metrics_c = extract_metrics_from_dat(dat_c)
        print(f"  Coarse results:")
        print(f"    Max displacement: {metrics_c['max_displacement_m']:.6e} m")
        print(f"    Peak von Mises:   {metrics_c['peak_von_mises_pa']:.6e} Pa")
        print(f"    Disp records: {metrics_c['n_disp_records']}, Stress records: {metrics_c['n_stress_records']}")

        # ------------------------------------------------------------------
        # STEP 2: Solve refined mesh
        # ------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("  STEP 2: Refined mesh solve")
        print("=" * 50)

        # Modify geometry.json to double the divisions
        cfg_refined = json.loads(GEOMETRY_CFG.read_text())
        cfg_refined["divisions_v2"] = {"nx": nx_r, "ny": ny_r, "nz": nz_r}
        GEOMETRY_CFG.write_text(json.dumps(cfg_refined, indent=2))
        print(f"  Updated geometry.json: divisions_v2 = {nx_r} x {ny_r} x {nz_r}")

        # Make sure no density file
        remove_density_file()

        # Generate refined mesh
        n_nodes_r, n_elems_r = generate_mesh("refined")
        print(f"  Refined mesh: {n_nodes_r} nodes, {n_elems_r} elements")

        # Generate .inp deck
        generate_inp_mewing_only("refined")

        # Solve (larger mesh, allow more time)
        solve_case(REFINED_WORK, "refined", timeout=600)

        # Extract metrics
        dat_r = REFINED_WORK / f"{CASE_NAME}.dat"
        metrics_r = extract_metrics_from_dat(dat_r)
        print(f"  Refined results:")
        print(f"    Max displacement: {metrics_r['max_displacement_m']:.6e} m")
        print(f"    Peak von Mises:   {metrics_r['peak_von_mises_pa']:.6e} Pa")
        print(f"    Disp records: {metrics_r['n_disp_records']}, Stress records: {metrics_r['n_stress_records']}")

        # ------------------------------------------------------------------
        # STEP 3: Compute convergence
        # ------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("  STEP 3: Convergence analysis")
        print("=" * 50)

        vm_c = metrics_c["peak_von_mises_pa"]
        vm_r = metrics_r["peak_von_mises_pa"]
        u_c = metrics_c["max_displacement_m"]
        u_r = metrics_r["max_displacement_m"]

        if vm_r > 0:
            vm_ratio = abs(vm_r - vm_c) / vm_r
        else:
            vm_ratio = float("inf")

        if u_r > 0:
            u_ratio = abs(u_r - u_c) / u_r
        else:
            u_ratio = float("inf")

        def verdict(ratio):
            if ratio < 0.05:
                return "ADEQUATE (< 5%)"
            elif ratio < 0.10:
                return "ACCEPTABLE (< 10%)"
            else:
                return "REFINEMENT NEEDED (> 10%)"

        print()
        print(f"  Peak von Mises stress:")
        print(f"    Coarse  ({n_elems_c:>6d} elems): {vm_c:.6e} Pa")
        print(f"    Refined ({n_elems_r:>6d} elems): {vm_r:.6e} Pa")
        print(f"    Relative difference:            {vm_ratio:.4f} ({vm_ratio*100:.2f}%)")
        print(f"    Verdict: {verdict(vm_ratio)}")
        print()
        print(f"  Max displacement:")
        print(f"    Coarse  ({n_elems_c:>6d} elems): {u_c:.6e} m")
        print(f"    Refined ({n_elems_r:>6d} elems): {u_r:.6e} m")
        print(f"    Relative difference:            {u_ratio:.4f} ({u_ratio*100:.2f}%)")
        print(f"    Verdict: {verdict(u_ratio)}")
        print()

        overall = max(vm_ratio, u_ratio)
        print(f"  Overall convergence ratio: {overall:.4f} ({overall*100:.2f}%)")
        print(f"  Overall verdict: {verdict(overall)}")

        # ------------------------------------------------------------------
        # STEP 4: Write results CSV
        # ------------------------------------------------------------------
        csv_path = RESULTS_DIR / "mesh_convergence.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "mesh_level", "nx", "ny", "nz", "n_elements", "n_nodes",
                "peak_von_mises_pa", "max_displacement_m",
                "vm_convergence_ratio", "u_convergence_ratio", "verdict"
            ])
            w.writerow([
                "coarse", nx_c, ny_c, nz_c, n_elems_c, n_nodes_c,
                f"{vm_c:.6e}", f"{u_c:.6e}",
                "", "", ""
            ])
            w.writerow([
                "refined", nx_r, ny_r, nz_r, n_elems_r, n_nodes_r,
                f"{vm_r:.6e}", f"{u_r:.6e}",
                f"{vm_ratio:.6f}", f"{u_ratio:.6f}", verdict(overall)
            ])
        print(f"\n  Results written to {csv_path}")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("  MESH CONVERGENCE STUDY COMPLETE")
        print("=" * 70)
        print(f"  Coarse mesh: {nx_c}x{ny_c}x{nz_c} = {n_elems_c} elements")
        print(f"  Refined mesh: {nx_r}x{ny_r}x{nz_r} = {n_elems_r} elements")
        print(f"  Stress convergence: {vm_ratio*100:.2f}%  -> {verdict(vm_ratio)}")
        print(f"  Displacement convergence: {u_ratio*100:.2f}%  -> {verdict(u_ratio)}")
        print(f"  Overall: {verdict(overall)}")
        print("=" * 70)

    finally:
        # ALWAYS restore original files
        print("\n[cleanup] Restoring original files...")
        restore_files()
        # Clean up work directories (keep for inspection but note they exist)
        print(f"[cleanup] Convergence work dirs preserved at:")
        print(f"  {COARSE_WORK}")
        print(f"  {REFINED_WORK}")
        print("[cleanup] Original geometry.json and mesh_data.json restored.")


if __name__ == "__main__":
    main()

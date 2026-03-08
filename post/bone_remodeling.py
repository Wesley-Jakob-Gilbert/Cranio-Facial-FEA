#!/usr/bin/env python3
"""Huiskes-style density remodeling update per scenario/cycle.

Inputs:
- mesh/mesh_data.json (expects ESET_BONE / ESET_SUTURE_* when available)
- configs/remodeling.yaml
- results/<scenario>/elem_vm.csv
- results/<scenario>/elem_density.json (optional prior state)

Outputs:
- results/<scenario>/elem_density.json
- results/<scenario>/snapshots/cycle_<N>_density.json
- results/remodeling_summary.csv (append/update per cycle)
"""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_cfg():
    cfg = yaml.safe_load((ROOT / "configs" / "remodeling.yaml").read_text())
    return cfg["remodeling"]


def load_sets(mesh: dict):
    elems = {int(k): v for k, v in mesh["elements"].items()}
    all_eids = set(elems.keys())
    sets = mesh.get("sets", {})

    bone = set(sets.get("ESET_BONE", []))
    s_mid = set(sets.get("ESET_SUTURE_MID", []))
    s_lat = set(sets.get("ESET_SUTURE_LAT", []))
    suture = s_mid | s_lat

    if not bone:
        bone = all_eids - suture
    if not suture:
        suture = all_eids - bone

    # final safety partition
    overlap = bone & suture
    if overlap:
        bone -= overlap
    missing = all_eids - bone - suture
    bone |= missing

    return all_eids, bone, suture


def read_vm(path: Path):
    vm = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vm[int(row["elem_id"])] = float(row["vm_avg_pa"])
    return vm


def read_density(path: Path, all_eids: set[int]):
    if not path.exists():
        return {eid: 1.0 for eid in all_eids}
    data = json.loads(path.read_text())
    return {eid: float(data.get(str(eid), 1.0)) for eid in all_eids}


def write_density(path: Path, rho: dict[int, float]):
    payload = {str(k): float(v) for k, v in sorted(rho.items())}
    path.write_text(json.dumps(payload, indent=2))


def update_summary(summary_path: Path, row: dict):
    rows = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    # replace same scenario+cycle if exists
    rows = [r for r in rows if not (r.get("scenario") == row["scenario"] and int(r.get("cycle", -1)) == row["cycle"])]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: (r["scenario"], int(r["cycle"])))

    fields = [
        "scenario", "cycle", "mean_rho", "min_rho", "max_rho",
        "apposition_fraction", "resorption_fraction"
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: bone_remodeling.py <scenario> <cycle>")

    scenario = sys.argv[1]
    cycle = int(sys.argv[2])

    cfg = load_cfg()
    psi_ref = float(cfg["psi_ref_pa"])
    alpha = float(cfg["alpha"])
    rho_min = float(cfg["rho_min"])
    rho_max = float(cfg["rho_max"])
    n_power = float(cfg["n_power"])
    E_bone = float(cfg["E_bone_pa"])
    E_suture = float(cfg["E_suture_pa"])

    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    all_eids, bone, suture = load_sets(mesh)

    case_dir = ROOT / "results" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)

    vm_path = case_dir / "elem_vm.csv"
    if not vm_path.exists():
        raise SystemExit(f"Missing {vm_path}; run solve + extract_fields first")

    vm = read_vm(vm_path)
    rho_path = case_dir / "elem_density.json"
    rho_prev = read_density(rho_path, all_eids)

    rho_new = {}
    delta = {}

    for eid in all_eids:
        r0 = rho_prev[eid]
        if eid in suture:
            rho_new[eid] = 1.0
            delta[eid] = rho_new[eid] - r0
            continue

        vm_e = float(vm.get(eid, 0.0))
        E_prev = max(1e3, E_bone * (r0 ** n_power))
        psi = (vm_e * vm_e) / (2.0 * E_prev)
        stimulus = (psi / psi_ref) - 1.0
        drho = alpha * stimulus
        r1 = min(rho_max, max(rho_min, r0 + drho))
        rho_new[eid] = r1
        delta[eid] = r1 - r0

    write_density(rho_path, rho_new)

    snap_dir = case_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    write_density(snap_dir / f"cycle_{cycle}_density.json", rho_new)

    if bool(cfg.get("write_elem_modulus_debug", False)):
        mod = {}
        for eid, r in rho_new.items():
            if eid in suture:
                mod[str(eid)] = E_suture
            else:
                mod[str(eid)] = E_bone * (r ** n_power)
        (case_dir / "elem_modulus_debug.json").write_text(json.dumps(mod, indent=2))

    vals = list(rho_new.values())
    bone_ids = list(bone)
    app = sum(1 for eid in bone_ids if delta[eid] > 0) / max(1, len(bone_ids))
    res = sum(1 for eid in bone_ids if delta[eid] < 0) / max(1, len(bone_ids))

    update_summary(
        ROOT / "results" / "remodeling_summary.csv",
        {
            "scenario": scenario,
            "cycle": cycle,
            "mean_rho": f"{sum(vals)/len(vals):.6f}",
            "min_rho": f"{min(vals):.6f}",
            "max_rho": f"{max(vals):.6f}",
            "apposition_fraction": f"{app:.6f}",
            "resorption_fraction": f"{res:.6f}",
        },
    )

    print(f"Updated density for {scenario} cycle {cycle}")


if __name__ == "__main__":
    main()

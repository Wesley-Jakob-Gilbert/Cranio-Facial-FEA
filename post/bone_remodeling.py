#!/usr/bin/env python3
"""Huiskes-style density remodeling update per scenario/cycle.

Implements the Huiskes lazy-zone (dead-band) remodeling law:
  stimulus = (psi / psi_ref) - 1.0
  if stimulus > +s:   apposition  drho = alpha_app * (stimulus - s)
  if stimulus < -s:   resorption  drho = alpha_res * (stimulus + s)
  else:               lazy zone   drho = 0.0

Separate apposition/resorption rates reflect the biological reality
that bone forms slower than it resorbs.

Per-layer psi_ref: cortical and cancellous bone have different homeostatic
SED setpoints, reflecting the different mechanical environments of compact
vs trabecular bone. Fallback to global psi_ref for legacy (single-material)
meshes.

Supports layered bone (cortical/cancellous) with different base moduli,
tooth root inclusions (held at rho=1.0), and suture strain-dependent
modulus softening.

Inputs:
- mesh/mesh_data.json (expects ESET_BONE / ESET_SUTURE_* when available)
- configs/remodeling.yaml
- results/<scenario>/elem_vm.csv
- results/<scenario>/elem_density.json (optional prior state)

Outputs:
- results/<scenario>/elem_density.json
- results/<scenario>/snapshots/cycle_<N>_density.json
- results/<scenario>/suture_modulus.json (if suture remodeling enabled)
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

    # Fine-grained layer sets (may be empty for legacy meshes)
    cortical = set(sets.get("ESET_CORTICAL", []))
    cancellous = set(sets.get("ESET_CANCELLOUS", []))
    tooth_root = set(sets.get("ESET_TOOTH_ROOT", []))

    if not bone:
        bone = all_eids - suture
    if not suture:
        suture = all_eids - bone

    # final safety partition
    overlap = bone & suture
    if overlap:
        bone -= overlap
    missing = all_eids - bone - suture - tooth_root
    bone |= missing

    return all_eids, bone, suture, cortical, cancellous, tooth_root


def read_vm(path: Path):
    vm = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vm[int(row["elem_id"])] = float(row["vm_avg_pa"])
    return vm


def read_sed(path: Path):
    """Read true strain energy density from elem_sed.csv (CalculiX ENER output)."""
    sed = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sed[int(row["elem_id"])] = float(row["sed_avg_pa"])
    return sed


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
        "apposition_fraction", "resorption_fraction",
        "lazy_fraction", "mean_abs_drho",
        "mean_suture_E", "min_suture_E",
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
    psi_ref_global = float(cfg["psi_ref_pa"])
    rho_min = float(cfg["rho_min"])
    rho_max = float(cfg["rho_max"])
    n_power = float(cfg["n_power"])
    n_power_cortical = float(cfg.get("n_power_cortical", n_power))
    n_power_cancellous = float(cfg.get("n_power_cancellous", n_power))
    E_bone = float(cfg["E_bone_pa"])
    E_suture = float(cfg["E_suture_pa"])

    # Layered material moduli (fallback to E_bone for backward compat)
    E_cortical = float(cfg.get("E_cortical_pa", E_bone))
    E_cancellous = float(cfg.get("E_cancellous_pa", E_bone))

    # Per-layer homeostatic SED setpoints (fallback to global psi_ref)
    psi_ref_cortical = float(cfg.get("psi_ref_cortical_pa", psi_ref_global))
    psi_ref_cancellous = float(cfg.get("psi_ref_cancellous_pa", psi_ref_global))

    # Lazy-zone remodeling parameters (backward compat with old 'alpha' key)
    if "alpha_apposition" in cfg:
        alpha_app = float(cfg["alpha_apposition"])
        alpha_res = float(cfg["alpha_resorption"])
    else:
        alpha_app = float(cfg["alpha"])
        alpha_res = float(cfg["alpha"])
    lazy_s = float(cfg.get("lazy_zone_s", 0.0))

    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    all_eids, bone, suture, cortical, cancellous, tooth_root = load_sets(mesh)
    _has_layers = bool(cortical) or bool(cancellous) or bool(tooth_root)

    case_dir = ROOT / "results" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)

    vm_path = case_dir / "elem_vm.csv"
    if not vm_path.exists():
        raise SystemExit(f"Missing {vm_path}; run solve + extract_fields first")

    vm = read_vm(vm_path)

    # Use true SED from CalculiX ENER output if available; fall back to vm²/2E
    sed_path = case_dir / "elem_sed.csv"
    sed_direct = read_sed(sed_path) if sed_path.exists() else None
    if sed_direct:
        print(f"  Using true SED from {sed_path.name} ({len(sed_direct)} elements)")
    else:
        print(f"  Warning: {sed_path.name} not found, falling back to vm²/2E approximation")

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

        # Tooth root elements: held at rho=1.0 (like suture, no remodeling)
        if eid in tooth_root:
            rho_new[eid] = 1.0
            delta[eid] = rho_new[eid] - r0
            continue

        # Select layer-specific psi_ref and n_power
        if _has_layers and eid in cortical:
            psi_ref = psi_ref_cortical
            n_pow = n_power_cortical
        elif _has_layers and eid in cancellous:
            psi_ref = psi_ref_cancellous
            n_pow = n_power_cancellous
        else:
            psi_ref = psi_ref_global
            n_pow = n_power

        # Use true SED from CalculiX if available; otherwise approximate
        if sed_direct and eid in sed_direct:
            psi = sed_direct[eid]
        else:
            vm_e = float(vm.get(eid, 0.0))
            if _has_layers and eid in cortical:
                E_base = E_cortical
            elif _has_layers and eid in cancellous:
                E_base = E_cancellous
            else:
                E_base = E_bone
            E_prev = max(1e3, E_base * (r0 ** n_pow))
            psi = (vm_e * vm_e) / (2.0 * E_prev)

        stimulus = (psi / psi_ref) - 1.0

        # Lazy-zone (dead-band) remodeling law
        if stimulus > lazy_s:
            drho = alpha_app * (stimulus - lazy_s)
        elif stimulus < -lazy_s:
            drho = alpha_res * (stimulus + lazy_s)
        else:
            drho = 0.0

        r1 = min(rho_max, max(rho_min, r0 + drho))
        rho_new[eid] = r1
        delta[eid] = r1 - r0

    write_density(rho_path, rho_new)

    snap_dir = case_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    write_density(snap_dir / f"cycle_{cycle}_density.json", rho_new)

    # --- Suture strain-dependent modulus update ---
    suture_cfg = cfg.get("suture_remodeling", {})
    suture_remod_enabled = suture_cfg.get("enabled", False)
    suture_modulus = {}

    if suture_remod_enabled:
        beta = float(suture_cfg.get("beta", 0.5))
        E_suture_baseline = float(cfg.get("E_suture_pa", 5.0e7))
        E_min = float(suture_cfg.get("E_min_pa", 1.0e7))
        strain_thresh = float(suture_cfg.get("strain_threshold", 0.001))

        # Load prior suture modulus (for cumulative softening)
        suture_mod_path = case_dir / "suture_modulus.json"
        prior_suture_mod = {}
        if suture_mod_path.exists():
            prior_suture_mod = json.loads(suture_mod_path.read_text())

        for eid in suture:
            E_prev_suture = float(prior_suture_mod.get(str(eid), E_suture_baseline))
            vm_e = float(vm.get(eid, 0.0))
            strain = vm_e / E_prev_suture if E_prev_suture > 0 else 0.0

            if strain > strain_thresh:
                E_new = E_prev_suture * (1.0 - beta * (strain - strain_thresh))
                E_new = max(E_min, min(E_suture_baseline, E_new))
            else:
                E_new = E_prev_suture

            suture_modulus[str(eid)] = E_new

        # Write suture modulus state
        suture_mod_path.write_text(json.dumps(suture_modulus, indent=2))

        # Write snapshot
        (snap_dir / f"cycle_{cycle}_suture_modulus.json").write_text(
            json.dumps(suture_modulus, indent=2)
        )

        # Stats for summary
        suture_E_vals = list(suture_modulus.values())
        mean_suture_E = sum(suture_E_vals) / max(1, len(suture_E_vals))
        min_suture_E = min(suture_E_vals) if suture_E_vals else E_suture_baseline

    if bool(cfg.get("write_elem_modulus_debug", False)):
        mod = {}
        for eid, r in rho_new.items():
            if eid in suture:
                mod[str(eid)] = E_suture
            elif eid in tooth_root:
                mod[str(eid)] = float(cfg.get("E_tooth_root_pa", 2.0e10))
            elif _has_layers and eid in cortical:
                mod[str(eid)] = E_cortical * (r ** n_power_cortical)
            elif _has_layers and eid in cancellous:
                mod[str(eid)] = E_cancellous * (r ** n_power_cancellous)
            else:
                mod[str(eid)] = E_bone * (r ** n_power)
        (case_dir / "elem_modulus_debug.json").write_text(json.dumps(mod, indent=2))

    vals = list(rho_new.values())
    bone_ids = list(bone)
    n_bone = max(1, len(bone_ids))
    app = sum(1 for eid in bone_ids if delta[eid] > 1e-12) / n_bone
    res = sum(1 for eid in bone_ids if delta[eid] < -1e-12) / n_bone
    lazy = sum(1 for eid in bone_ids if abs(delta[eid]) <= 1e-12) / n_bone

    bone_rhos = [rho_new[eid] for eid in bone_ids]
    bone_deltas = [abs(delta[eid]) for eid in bone_ids if abs(delta[eid]) > 1e-12]
    mean_abs_drho = sum(bone_deltas) / max(1, len(bone_deltas)) if bone_deltas else 0.0

    summary_row = {
        "scenario": scenario,
        "cycle": cycle,
        "mean_rho": f"{sum(bone_rhos)/len(bone_rhos):.6f}",
        "min_rho": f"{min(bone_rhos):.6f}",
        "max_rho": f"{max(bone_rhos):.6f}",
        "apposition_fraction": f"{app:.6f}",
        "resorption_fraction": f"{res:.6f}",
        "lazy_fraction": f"{lazy:.6f}",
        "mean_abs_drho": f"{mean_abs_drho:.6f}",
    }
    if suture_remod_enabled and suture_modulus:
        summary_row["mean_suture_E"] = f"{mean_suture_E:.6e}"
        summary_row["min_suture_E"] = f"{min_suture_E:.6e}"

    update_summary(ROOT / "results" / "remodeling_summary.csv", summary_row)

    suture_info = ""
    if suture_remod_enabled and suture_modulus:
        suture_info = f"  mean_suture_E={mean_suture_E:.4e}  min_suture_E={min_suture_E:.4e}"

    print(f"[remodel] {scenario} cycle {cycle}: "
          f"mean_rho={sum(bone_rhos)/len(bone_rhos):.4f}  "
          f"app={app:.1%}  res={res:.1%}  lazy={lazy:.1%}  "
          f"mean|drho|={mean_abs_drho:.6f}{suture_info}")


if __name__ == "__main__":
    main()

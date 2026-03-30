#!/usr/bin/env python3
"""Huiskes-style density remodeling update per scenario/cycle.

## Remodeling law
  stimulus = (psi_eff / psi_ref) - 1.0
  if stimulus > +s:   apposition  drho = alpha_app * (stimulus - s)
  if stimulus < -s:   resorption  drho = alpha_res * (stimulus + s)
  else:               lazy zone   drho = 0.0

Lazy-zone half-width *s* calibrated to physiological range (±15%, not ±50%).

## Cyclic-load correction (Mechanostat / dynamic loading)
Real bone remodeling is driven by **cyclic dynamic loads** (Frost's Mechanostat),
not sustained static posture. The resting tongue-on-palate SED is too small to
drive significant remodeling on its own; the intermittent swallowing bolus (peak
30–50 kPa, ~2000×/day) delivers the osteogenic strain signal.

Effective SED per cycle:
  psi_eff = rest_frac * psi_rest + swallow_frac * psi_swallow_est
where:
  swallow_frac     = min(swallow_cycles_per_day * swallow_duration_s / 86400, 1.0)
  rest_frac        = 1.0 - swallow_frac
  psi_swallow_est  = psi_rest * (swallow_peak_kpa / tongue_kpa_ref)^2
(Quadratic scaling from linear elasticity: doubling pressure → 4× SED.)

## Suture softening (exponential + recovery)
Exponential decay replaces the linear approximation:
  E_new = E_prev * exp(-beta * excess_strain)   [loading path]
  E_new = E_prev + recovery_rate * (E_baseline - E_prev)  [recovery, strain < thresh]
This prevents the physically unrealistic linear collapse and models both
progressive softening and partial secondary ossification.

## Orthotropic cortical bone (material.yaml scaffold)
Orthotropic material constants are loaded from material.yaml if present under
materials.cortical_ortho. The .inp files still use isotropic CORTICAL material
unless the orthotropic flag is enabled — handled in make_inp.py. The remodeling
loop uses the principal modulus E1 (periosteal/endosteal loading axis) for the
SED approximation fallback.

Inputs:
- mesh/mesh_data.json
- configs/remodeling.yaml
- results/<scenario>/elem_vm.csv
- results/<scenario>/elem_sed.csv  (preferred; true CalculiX ENER SED)
- results/<scenario>/elem_density.json  (optional prior state)
- results/<scenario>/suture_modulus.json  (optional prior state)

Outputs:
- results/<scenario>/elem_density.json
- results/<scenario>/snapshots/cycle_<N>_density.json
- results/<scenario>/suture_modulus.json
- results/remodeling_summary.csv
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

    cortical = set(sets.get("ESET_CORTICAL", []))
    cancellous = set(sets.get("ESET_CANCELLOUS", []))
    tooth_root = set(sets.get("ESET_TOOTH_ROOT", []))

    if not bone:
        bone = all_eids - suture
    if not suture:
        suture = all_eids - bone

    overlap = bone & suture
    if overlap:
        bone -= overlap
    missing = all_eids - bone - suture - tooth_root
    bone |= missing

    return all_eids, bone, suture, cortical, cancellous, tooth_root


def read_vm(path: Path) -> dict[int, float]:
    vm = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            vm[int(row["elem_id"])] = float(row["vm_avg_pa"])
    return vm


def read_sed(path: Path) -> dict[int, float]:
    """Read true strain energy density from elem_sed.csv (CalculiX ENER output)."""
    sed = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sed[int(row["elem_id"])] = float(row["sed_avg_pa"])
    return sed


def read_density(path: Path, all_eids: set[int]) -> dict[int, float]:
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

    rows = [r for r in rows if not (r.get("scenario") == row["scenario"]
                                     and int(r.get("cycle", -1)) == row["cycle"])]
    rows.append({k: str(v) for k, v in row.items()})
    rows.sort(key=lambda r: (r["scenario"], int(r["cycle"])))

    fields = [
        "scenario", "cycle", "mean_rho", "min_rho", "max_rho",
        "apposition_fraction", "resorption_fraction",
        "lazy_fraction", "mean_abs_drho",
        "mean_suture_E", "min_suture_E",
        "cyclic_swallow_frac", "psi_swallow_scale",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Cyclic-load duty-cycle correction
# ---------------------------------------------------------------------------

def compute_cyclic_factors(cyclic_cfg: dict, tongue_kpa: float) -> tuple[float, float]:
    """Return (swallow_frac, psi_scale) for the cyclic-load correction.

    psi_scale: the factor by which the raw (static) SED is scaled to obtain
    the effective dynamic SED seen by bone cells per unit time.

    psi_eff = psi_rest * [(rest_frac) + swallow_frac * (peak_kpa/ref_kpa)^2]

    Args:
        cyclic_cfg: the 'cyclic_loading' sub-dict from remodeling.yaml
        tongue_kpa: resting tongue pressure for this load case (kPa)

    Returns:
        swallow_frac: fraction of day occupied by swallowing bolus
        psi_scale: total SED multiplier (>1 when swallowing dominates)
    """
    swallow_cycles = float(cyclic_cfg.get("swallow_cycles_per_day", 2000))
    swallow_dur_s = float(cyclic_cfg.get("swallow_duration_s", 0.5))
    peak_kpa = float(cyclic_cfg.get("swallow_peak_kpa", 35.0))
    ref_kpa = float(cyclic_cfg.get("tongue_kpa_ref", 2.0))

    swallow_frac = min(swallow_cycles * swallow_dur_s / 86400.0, 1.0)
    rest_frac = 1.0 - swallow_frac

    # SED scales as pressure^2 in linear elasticity (psi = sigma^2/2E)
    if ref_kpa > 0 and tongue_kpa > 0:
        pressure_ratio = peak_kpa / tongue_kpa
    elif ref_kpa > 0:
        # mouth-breathing: tongue_kpa = 0; swallowing still occurs but is
        # the only loading event. Use ref_kpa as denominator.
        pressure_ratio = peak_kpa / ref_kpa
    else:
        pressure_ratio = 1.0

    psi_swallow_scale = pressure_ratio ** 2
    psi_scale = rest_frac + swallow_frac * psi_swallow_scale
    return swallow_frac, psi_scale


# ---------------------------------------------------------------------------
# Suture exponential softening + recovery
# ---------------------------------------------------------------------------

def update_suture_modulus(
    eid: int,
    vm_e: float,
    E_prev: float,
    E_baseline: float,
    suture_cfg: dict,
) -> float:
    """Compute updated suture modulus for one element.

    Exponential softening (replaces linear approximation):
        E_new = E_prev * exp(-beta * excess_strain)   when strain > threshold
    Sub-threshold recovery (partial secondary ossification):
        E_new = E_prev + recovery_rate * (E_baseline - E_prev)  when strain <= threshold

    Args:
        eid:         element id (for debug, not used in math)
        vm_e:        von Mises stress at this element (Pa), used to estimate strain
        E_prev:      current Young's modulus for this suture element (Pa)
        E_baseline:  intact suture Young's modulus (Pa)
        suture_cfg:  sub-dict from remodeling.yaml -> suture_remodeling

    Returns:
        Updated Young's modulus (Pa), clamped to [E_min, E_baseline].
    """
    beta = float(suture_cfg.get("beta", 40.0))
    E_min = float(suture_cfg.get("E_min_pa", 1.0e7))
    strain_thresh = float(suture_cfg.get("strain_threshold", 0.00015))
    recovery_rate = float(suture_cfg.get("recovery_rate", 0.05))

    # Approximate strain from vm stress / current modulus
    strain = vm_e / E_prev if E_prev > 0 else 0.0

    if strain > strain_thresh:
        excess = strain - strain_thresh
        # Exponential decay: physically prevents collapse at high strains
        E_new = E_prev * math.exp(-beta * excess)
        E_new = max(E_min, min(E_baseline, E_new))
    else:
        # Recovery toward baseline (secondary ossification analog)
        E_new = E_prev + recovery_rate * (E_baseline - E_prev)
        E_new = min(E_baseline, E_new)

    return E_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        raise SystemExit("Usage: bone_remodeling.py <scenario> <cycle> [tongue_kpa]")

    scenario = sys.argv[1]
    cycle = int(sys.argv[2])
    # Optional: tongue_kpa passed from remodeling_loop.sh for cyclic scaling
    tongue_kpa = float(sys.argv[3]) if len(sys.argv) >= 4 else 2.0

    cfg = load_cfg()
    psi_ref_global = float(cfg["psi_ref_pa"])
    rho_min = float(cfg["rho_min"])
    rho_max = float(cfg["rho_max"])
    n_power = float(cfg["n_power"])
    n_power_cortical = float(cfg.get("n_power_cortical", n_power))
    n_power_cancellous = float(cfg.get("n_power_cancellous", n_power))
    E_bone = float(cfg["E_bone_pa"])
    E_suture = float(cfg["E_suture_pa"])
    E_cortical = float(cfg.get("E_cortical_pa", E_bone))
    E_cancellous = float(cfg.get("E_cancellous_pa", E_bone))
    psi_ref_cortical = float(cfg.get("psi_ref_cortical_pa", psi_ref_global))
    psi_ref_cancellous = float(cfg.get("psi_ref_cancellous_pa", psi_ref_global))

    if "alpha_apposition" in cfg:
        alpha_app = float(cfg["alpha_apposition"])
        alpha_res = float(cfg["alpha_resorption"])
    else:
        alpha_app = float(cfg["alpha"])
        alpha_res = float(cfg["alpha"])
    lazy_s = float(cfg.get("lazy_zone_s", 0.15))

    # --- Cyclic-load correction ---
    cyclic_cfg = cfg.get("cyclic_loading", {})
    cyclic_enabled = bool(cyclic_cfg.get("enabled", False))
    if cyclic_enabled:
        swallow_frac, psi_scale = compute_cyclic_factors(cyclic_cfg, tongue_kpa)
        print(f"  [cyclic] swallow_frac={swallow_frac:.4f}  psi_scale={psi_scale:.3f}x")
    else:
        swallow_frac, psi_scale = 0.0, 1.0

    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    all_eids, bone, suture, cortical, cancellous, tooth_root = load_sets(mesh)
    _has_layers = bool(cortical) or bool(cancellous) or bool(tooth_root)

    case_dir = ROOT / "results" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)

    vm_path = case_dir / "elem_vm.csv"
    if not vm_path.exists():
        raise SystemExit(f"Missing {vm_path}; run solve + extract_fields first")

    vm = read_vm(vm_path)

    # Prefer true SED from CalculiX ENER output
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

        if eid in suture or eid in tooth_root:
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

        # Raw (static) SED
        if sed_direct and eid in sed_direct:
            psi_raw = sed_direct[eid]
        else:
            vm_e = float(vm.get(eid, 0.0))
            if _has_layers and eid in cortical:
                E_base = E_cortical
            elif _has_layers and eid in cancellous:
                E_base = E_cancellous
            else:
                E_base = E_bone
            E_prev = max(1e3, E_base * (r0 ** n_pow))
            psi_raw = (vm_e * vm_e) / (2.0 * E_prev)

        # Apply cyclic-load correction (scales raw static SED to effective dynamic SED)
        psi = psi_raw * psi_scale

        stimulus = (psi / psi_ref) - 1.0

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

    # --- Suture modulus update (exponential softening + recovery) ---
    suture_cfg = cfg.get("suture_remodeling", {})
    suture_remod_enabled = suture_cfg.get("enabled", False)
    suture_modulus = {}
    mean_suture_E = float(E_suture)
    min_suture_E = float(E_suture)

    if suture_remod_enabled:
        suture_mod_path = case_dir / "suture_modulus.json"
        prior_suture_mod = {}
        if suture_mod_path.exists():
            prior_suture_mod = json.loads(suture_mod_path.read_text())

        E_suture_baseline = float(cfg.get("E_suture_pa", 5.0e7))

        for eid in suture:
            E_prev_suture = float(prior_suture_mod.get(str(eid), E_suture_baseline))
            vm_e = float(vm.get(eid, 0.0))

            E_new = update_suture_modulus(
                eid, vm_e, E_prev_suture, E_suture_baseline, suture_cfg
            )
            suture_modulus[str(eid)] = E_new

        suture_mod_path.write_text(json.dumps(suture_modulus, indent=2))
        (snap_dir / f"cycle_{cycle}_suture_modulus.json").write_text(
            json.dumps(suture_modulus, indent=2)
        )

        suture_E_vals = list(suture_modulus.values())
        if suture_E_vals:
            mean_suture_E = sum(suture_E_vals) / len(suture_E_vals)
            min_suture_E = min(suture_E_vals)

    # --- Modulus debug output ---
    if bool(cfg.get("write_elem_modulus_debug", False)):
        mod = {}
        for eid, r in rho_new.items():
            if eid in suture:
                mod[str(eid)] = float(suture_modulus.get(str(eid), E_suture))
            elif eid in tooth_root:
                mod[str(eid)] = float(cfg.get("E_tooth_root_pa", 2.0e10))
            elif _has_layers and eid in cortical:
                mod[str(eid)] = E_cortical * (r ** n_power_cortical)
            elif _has_layers and eid in cancellous:
                mod[str(eid)] = E_cancellous * (r ** n_power_cancellous)
            else:
                mod[str(eid)] = E_bone * (r ** n_power)
        (case_dir / "elem_modulus_debug.json").write_text(json.dumps(mod, indent=2))

    # --- Summary stats ---
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
        "cyclic_swallow_frac": f"{swallow_frac:.6f}",
        "psi_swallow_scale": f"{psi_scale:.4f}",
    }
    if suture_remod_enabled and suture_modulus:
        summary_row["mean_suture_E"] = f"{mean_suture_E:.6e}"
        summary_row["min_suture_E"] = f"{min_suture_E:.6e}"

    update_summary(ROOT / "results" / "remodeling_summary.csv", summary_row)

    suture_info = ""
    if suture_remod_enabled and suture_modulus:
        suture_info = f"  mean_suture_E={mean_suture_E:.4e}  min_suture_E={min_suture_E:.4e}"

    print(
        f"[remodel] {scenario} cycle {cycle} (tongue={tongue_kpa}kPa psi_scale={psi_scale:.2f}x): "
        f"mean_rho={sum(bone_rhos)/len(bone_rhos):.4f}  "
        f"app={app:.1%}  res={res:.1%}  lazy={lazy:.1%}  "
        f"mean|drho|={mean_abs_drho:.6f}{suture_info}"
    )


if __name__ == "__main__":
    main()

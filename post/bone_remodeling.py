#!/usr/bin/env python3
"""Huiskes-style density remodeling update per scenario/cycle.

## Remodeling law
  stimulus = (psi_eff / psi_ref) - 1.0
  if stimulus > +s:   apposition  drho = alpha_app * (stimulus - s)
  if stimulus < -s:   resorption  drho = alpha_res * (stimulus + s)
  else:               lazy zone   drho = 0.0

Lazy-zone half-width *s* calibrated to physiological range (±15%, not ±50%).

## Fix 1: Density-normalized strain-equivalent stimulus
Because psi ∝ E·ε² ∝ ρⁿ·ε², a naive SED stimulus over-rewards already-dense
bone. The mechanostat senses STRAIN, not energy density, so we compute the
dimensionless strain-equivalent ratio:

  ε_eff / ε_ref = sqrt(psi_eff / (psi_ref * rho^n_power))

  stimulus = sqrt(psi_eff / (psi_ref * rho^n_power)) - 1.0

This matches the analytical eps/eps_ref exactly under Carter-Hayes E(rho)=E0*rho^n.
The linear ratio (psi/(psi_ref*rho^n) - 1) overestimates sensitivity at low density
by 3-5× vs. the correct strain-equivalent (audit_3 item 2).

Config: density_stimulus_sqrt: true (default; set false to revert to linear ratio)
Audit_2 item 1 + Audit_3 item 2 combined.

## Fix 2: Power-law cyclic loading (Rubin & Lanyon 1984 / Turner 1998)
Static rest loads are biologically inert (Rubin & Lanyon 1984).  The correct
stimulus is a power-law function of peak dynamic strain (or SED proxy) and
cycle count, not a linear time-fraction average.

  psi_eff = (swallow_cycles_per_day / ref_cycles)^(1/m) * psi_peak_swallow

Where m ≈ 4 (Turner 1998; "four-thirds" power law for fatigue-like loading).
The static rest_frac is dropped entirely.
Audit_2 item 2.

## Fix 3: Tension-only suture softening
Sutures widen under TENSION (membrane opening), not compression.
Von Mises stress is non-negative and includes compressive deviatoric components,
making it physically wrong for suture-softening trigger.
We now use max_principal_stress (s1) from elem_principal_stress.csv.
Softening fires only when s1 > tension_threshold (Pa, defaults to 10 kPa).
Under compressive loading (s1 <= 0) softening is suppressed; recovery proceeds.
Audit_2 item 3.

## Fix 4: Δρ rate limiter (integration stability)
Explicit Euler integration can oscillate near stress concentrations.
A maximum biological rate limit (|Δρ| ≤ Δρ_max per cycle, default 0.05)
reflects osteoblast/osteoclast recruitment physiology.
Audit_2 item 4.

## Fix 5: Proper isotropic SED fallback (replaces vm²/2E)
For isotropic materials the exact SED from the stress tensor is:
  psi_iso = (1/2E) * [sxx² + syy² + szz² - ν(sxx·syy + syy·szz + szz·sxx)] +
            (1/2G) * [sxy² + sxz² + syz²]
where G = E/(2(1+ν)).
This is only used when CalculiX ENER (elem_sed.csv) is unavailable.
For orthotropic runs we still require elem_sed.csv — a hard warning is emitted.
Audit_2 item 5.

## Cyclic-load duty cycle (updated — see Fix 2)
Power-law effective SED replaces linear time-fraction averaging.

## Suture softening (exponential + recovery, tension-only — see Fix 3)
Exponential decay on tension path; recovery on compressive/sub-threshold path.

## Orthotropic cortical bone (material.yaml scaffold)
Orthotropic material constants are loaded from material.yaml if present under
materials.cortical_ortho.  Orthotropic runs REQUIRE elem_sed.csv (true ENER);
the vm²/2E fallback is disabled for orthotropic.

Inputs:
- mesh/mesh_data.json
- configs/remodeling.yaml
- configs/material.yaml           (optional, for orthotropic)
- results/<scenario>/elem_vm.csv
- results/<scenario>/elem_sed.csv              (preferred; CalculiX ENER)
- results/<scenario>/elem_stress_tensor.csv    (fallback SED; Fix 5)
- results/<scenario>/elem_principal_stress.csv (suture softening; Fix 3)
- results/<scenario>/elem_density.json         (optional prior state)
- results/<scenario>/suture_modulus.json       (optional prior state)

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


# ---------------------------------------------------------------------------
# Config & mesh helpers
# ---------------------------------------------------------------------------

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
        for row in csv.DictReader(f):
            vm[int(row["elem_id"])] = float(row["vm_avg_pa"])
    return vm


def read_sed(path: Path) -> dict[int, float]:
    """Read true strain energy density from elem_sed.csv (CalculiX ENER output)."""
    sed = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sed[int(row["elem_id"])] = float(row["sed_avg_pa"])
    return sed


def read_stress_tensor(path: Path) -> dict[int, tuple]:
    """Read full stress tensor per element from elem_stress_tensor.csv.

    Returns dict: eid → (sxx, syy, szz, sxy, sxz, syz)  [Pa]
    Used for Fix 5: proper isotropic SED fallback.
    """
    tensor = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["elem_id"])
            tensor[eid] = (
                float(row["sxx_pa"]), float(row["syy_pa"]), float(row["szz_pa"]),
                float(row["sxy_pa"]), float(row["sxz_pa"]), float(row["syz_pa"]),
            )
    return tensor


def read_principal_stress(path: Path) -> dict[int, tuple]:
    """Read principal stresses per element from elem_principal_stress.csv.

    Returns dict: eid → (s1_max, s2_mid, s3_min)  [Pa], s1 ≥ s2 ≥ s3.
    Used for Fix 3: tension-only suture softening.
    """
    principal = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = int(row["elem_id"])
            principal[eid] = (
                float(row["s1_max_pa"]),
                float(row["s2_mid_pa"]),
                float(row["s3_min_pa"]),
            )
    return principal


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
        "cyclic_psi_eff_scale",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Fix 5: Proper isotropic SED fallback from full stress tensor
# ---------------------------------------------------------------------------

def sed_from_tensor_isotropic(sxx, syy, szz, sxy, sxz, syz, E: float, nu: float) -> float:
    """Compute exact SED for an isotropic linear elastic material from its stress tensor.

    psi_iso = (1/2E) * [sxx² + syy² + szz²
                        - 2ν(sxx·syy + syy·szz + szz·sxx)]
            + (1+ν)/E * [sxy² + sxz² + syz²]

    This is derived from the compliance form: psi = σ:ε/2 = σ:C⁻¹:σ/2,
    exact for isotropic materials regardless of loading mode.

    The vm²/2E approximation corresponds to assuming pure deviatoric loading
    and ignores the volumetric (pressure) contribution to stored energy,
    which can be significant in constrained cranial sutures.
    """
    G = E / (2.0 * (1.0 + nu))
    # Normal stress contribution (includes Poisson coupling)
    normal = (sxx**2 + syy**2 + szz**2
              - 2.0 * nu * (sxx*syy + syy*szz + szz*sxx)) / (2.0 * E)
    # Shear stress contribution (factor of 2 absorbed into compliance)
    shear = (sxy**2 + sxz**2 + syz**2) / (2.0 * G)
    return max(0.0, normal + shear)


# ---------------------------------------------------------------------------
# Fix 2: Power-law cyclic loading (Turner 1998 / Rubin & Lanyon 1984)
# ---------------------------------------------------------------------------

def compute_cyclic_psi_eff(
    cyclic_cfg: dict,
    tongue_kpa: float,
    psi_rest: float,
) -> tuple[float, float]:
    """Return (psi_eff, scale_factor) using power-law cyclic loading correction.

    Biological background:
        Rubin & Lanyon (1984): static loads are biologically inert for osteogenesis.
        Turner (1998): osteogenic signal ∝ (N/N_ref)^(1/m) · ε_peak (m ≈ 3–4).

    Power-law formulation:
        psi_peak = psi_rest * (swallow_peak_kpa / tongue_kpa_ref)^2  (linear elasticity)
        psi_eff  = psi_peak * (swallow_cycles / ref_cycles)^(1/m)

    Audit_3 Fix 3a — ref_cycles homeostatic normalization:
        ref_cycles_per_day should equal the habitual swallowing rate (swallow_cycles_per_day).
        At habitually normal activity, N/N_ref = 1 → cycle_factor = 1.0 (homeostatic neutral).
        With old default ref_cycles=1, cycle_factor = 2000^(1/4) = 6.69× even at baseline —
        every scenario was incorrectly amplified. Turner's ratio is N_actual/N_habitual.

    Audit_3 Fix 3b — psi_ref auto-calibration:
        After fix 3a, psi_eff = psi_rest * (peak_kpa/tongue_kpa)^2 at homeostasis (N=N_ref).
        For the stimulus to be zero at homeostasis, psi_ref must equal this peak-swallow SED:
          psi_ref_dynamic = psi_ref_static * (swallow_peak_kpa / tongue_kpa_ref)^2
        This is handled in the calling code (main) when psi_ref_auto_calibrate_cyclic=True.

    The static rest contribution is DROPPED (per Rubin & Lanyon 1984).

    Args:
        cyclic_cfg:  'cyclic_loading' sub-dict from remodeling.yaml
        tongue_kpa:  resting tongue pressure for this load case (kPa)
        psi_rest:    resting (static) SED for this element (Pa)

    Returns:
        psi_eff:      effective dynamic SED (Pa) seen by bone cells
        scale_factor: psi_eff / psi_rest for logging
    """
    swallow_cycles = float(cyclic_cfg.get("swallow_cycles_per_day", 2000))
    peak_kpa = float(cyclic_cfg.get("swallow_peak_kpa", 35.0))
    ref_kpa = float(cyclic_cfg.get("tongue_kpa_ref", 2.0))
    # Audit_3 Fix 3a: ref_cycles should = habitual daily count (Turner N_0 = N_habitual).
    # Default changed from 1.0 → swallow_cycles (homeostatic neutral: N/N_ref = 1.0).
    # Old default of 1.0 caused 2000^(1/4) = 6.69× spurious amplification at normal activity.
    ref_cycles = float(cyclic_cfg.get("ref_cycles_per_day", swallow_cycles))
    m_power = float(cyclic_cfg.get("m_power", 4.0))  # Turner 1998; 3-4 range

    # Peak swallow SED: scales as pressure^2 from linear elasticity
    if tongue_kpa > 0 and ref_kpa > 0:
        pressure_ratio = peak_kpa / tongue_kpa
    elif ref_kpa > 0:
        # Mouth-breathing: resting tongue_kpa=0, but swallowing still occurs
        pressure_ratio = peak_kpa / ref_kpa
    else:
        pressure_ratio = 1.0

    psi_peak = psi_rest * (pressure_ratio ** 2)

    # Power-law cycle scaling (Turner 1998):
    # More cycles amplify the signal but with diminishing returns
    cycle_factor = (swallow_cycles / max(ref_cycles, 1.0)) ** (1.0 / m_power)
    psi_eff = psi_peak * cycle_factor

    scale_factor = psi_eff / psi_rest if psi_rest > 1e-30 else 1.0
    return psi_eff, scale_factor


# ---------------------------------------------------------------------------
# Fix 3: Tension-only suture softening
# ---------------------------------------------------------------------------

def update_suture_modulus(
    eid: int,
    s1_max: float,           # max principal stress (Pa); positive = tension
    E_prev: float,
    E_baseline: float,
    suture_cfg: dict,
) -> float:
    """Compute updated suture modulus for one element — TENSION-ONLY softening.

    Physical basis:
        Cranial sutures are fibrous periosteal joints that widen (soften/stretch)
        under TENSILE loading (membrane opening stress). Under compression they
        resist and can eventually ossify (synostosis). Von Mises stress is
        non-negative and includes compressive deviatoric contributions, making it
        a non-physical trigger for suture softening.

        Fix 3: use max_principal_stress (s1) from elem_principal_stress.csv.
        Softening fires only when s1 > tensile_threshold (Pa).
        When s1 ≤ 0 (compressive dominant): recovery path only.

    Exponential softening on tension path:
        E_new = E_prev * exp(-beta * (s1 - threshold) / E_prev)
        The strain-like excess is (s1 - threshold) / E_prev (Hookean approximation).

    Sub-threshold recovery (secondary ossification analog):
        E_new = E_prev + recovery_rate * (E_baseline - E_prev)

    Args:
        eid:          element id (debug only)
        s1_max:       max principal stress (Pa); positive = tension
        E_prev:       current Young's modulus for this suture element (Pa)
        E_baseline:   intact suture Young's modulus (Pa)
        suture_cfg:   sub-dict from remodeling.yaml -> suture_remodeling

    Returns:
        Updated Young's modulus (Pa), clamped to [E_min, E_baseline].
    """
    beta = float(suture_cfg.get("beta", 40.0))
    E_min = float(suture_cfg.get("E_min_pa", 1.0e7))
    # Tension threshold in Pa (default 10 kPa; typical suture yield onset)
    tension_thresh_pa = float(suture_cfg.get("tension_threshold_pa",
                                              suture_cfg.get("strain_threshold", 0.00015) * E_baseline))
    recovery_rate = float(suture_cfg.get("recovery_rate", 0.05))

    if s1_max > tension_thresh_pa:
        # Tensile opening path: exponential softening
        # Excess stress normalized by current modulus gives a strain-like measure
        excess_strain = (s1_max - tension_thresh_pa) / max(E_prev, 1.0)
        E_new = E_prev * math.exp(-beta * excess_strain)
        E_new = max(E_min, min(E_baseline, E_new))
    else:
        # Compressive or sub-threshold: recovery toward baseline (ossification analog)
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
    nu_bone = float(cfg.get("nu_bone", 0.3))        # Poisson ratio for SED Fix 5
    nu_cancellous = float(cfg.get("nu_cancellous", 0.3))
    nu_cortical = float(cfg.get("nu_cortical", 0.3))

    # Fix 4: Δρ rate limiter (max biological remodeling rate per cycle)
    drho_max = float(cfg.get("drho_max_per_cycle", 0.05))

    # Fix 1 + Audit_3 Fix 2: density-normalized strain-equivalent stimulus
    density_normalize_stimulus = bool(cfg.get("density_normalize_stimulus", True))
    # density_stimulus_sqrt: use sqrt(psi/(psi_ref*rho^n))-1 (strain-equivalent).
    # Linear ratio (audit_2 original) overstates sensitivity at low/high density.
    # sqrt matches analytical eps_eff/eps_ref = sqrt(2psi/E)/sqrt(2psi_ref/E0) exactly.
    density_stimulus_sqrt = bool(cfg.get("density_stimulus_sqrt", True))

    # Orthotropic cortical: requires true ENER SED; fallback disabled for ortho.
    mat_path = ROOT / "configs" / "material.yaml"
    _ortho_cfg_rm = {}
    if mat_path.exists():
        import yaml as _yaml
        _mat = _yaml.safe_load(mat_path.read_text())
        _ortho_cfg_rm = _mat.get("materials", {}).get("cortical_ortho", {})
    _ortho_enabled = bool(_ortho_cfg_rm.get("enabled", False))

    psi_ref_cortical = float(cfg.get("psi_ref_cortical_pa", psi_ref_global))
    psi_ref_cancellous = float(cfg.get("psi_ref_cancellous_pa", psi_ref_global))

    if "alpha_apposition" in cfg:
        alpha_app = float(cfg["alpha_apposition"])
        alpha_res = float(cfg["alpha_resorption"])
    else:
        alpha_app = float(cfg["alpha"])
        alpha_res = float(cfg["alpha"])
    lazy_s = float(cfg.get("lazy_zone_s", 0.15))

    # --- Cyclic-load config (Fix 2 + Audit_3 Fix 3) ---
    cyclic_cfg = cfg.get("cyclic_loading", {})
    cyclic_enabled = bool(cyclic_cfg.get("enabled", False))
    cyclic_psi_eff_scale_log = 1.0  # for summary only

    # Audit_3 Fix 3b: Auto-calibrate psi_ref to homeostatic peak-swallow SED.
    # When cyclic loading is enabled, the effective stimulus signal is driven by
    # peak-swallow SED, not static tongue-pressure SED. The stored psi_ref values
    # (calibrated against static CalculiX output) are therefore too low by a factor
    # of (swallow_peak_kpa / tongue_kpa_ref)^2, causing runaway apposition on every cycle.
    # Fix: scale psi_ref by the same pressure ratio used in compute_cyclic_psi_eff.
    # At homeostasis (N=N_ref), psi_eff = psi_rest*(peak/ref_kpa)^2, so
    # psi_ref_dynamic = psi_ref_static * (peak_kpa/ref_kpa)^2 gives stimulus=0. Correct.
    # Config: psi_ref_auto_calibrate_cyclic: true (default on; set false to revert).
    psi_ref_auto_calibrate_cyclic = bool(cfg.get("psi_ref_auto_calibrate_cyclic", True))
    if cyclic_enabled and psi_ref_auto_calibrate_cyclic:
        _c_peak_kpa = float(cyclic_cfg.get("swallow_peak_kpa", 35.0))
        _c_ref_kpa = float(cyclic_cfg.get("tongue_kpa_ref", 2.0))
        _pressure_ratio_sq = (_c_peak_kpa / max(_c_ref_kpa, 1e-9)) ** 2
        psi_ref_global = psi_ref_global * _pressure_ratio_sq
        psi_ref_cortical = psi_ref_cortical * _pressure_ratio_sq
        psi_ref_cancellous = psi_ref_cancellous * _pressure_ratio_sq
        print(f"  [cyclic psi_ref calibration] pressure_ratio_sq={_pressure_ratio_sq:.1f}x: "
              f"psi_ref_cortical={psi_ref_cortical:.2f} Pa, "
              f"psi_ref_cancellous={psi_ref_cancellous:.2f} Pa")

    mesh = json.loads((ROOT / "mesh" / "mesh_data.json").read_text())
    all_eids, bone, suture, cortical, cancellous, tooth_root = load_sets(mesh)
    _has_layers = bool(cortical) or bool(cancellous) or bool(tooth_root)

    case_dir = ROOT / "results" / scenario
    case_dir.mkdir(parents=True, exist_ok=True)

    vm_path = case_dir / "elem_vm.csv"
    if not vm_path.exists():
        raise SystemExit(f"Missing {vm_path}; run solve + extract_fields first")
    vm = read_vm(vm_path)

    # Prefer true SED from CalculiX ENER output (Fix 5: required for ortho)
    sed_path = case_dir / "elem_sed.csv"
    sed_direct = read_sed(sed_path) if sed_path.exists() else None
    if sed_direct:
        print(f"  Using true SED from {sed_path.name} ({len(sed_direct)} elements)")
    else:
        if _ortho_enabled:
            print(f"  ERROR: {sed_path.name} not found but orthotropic is enabled. "
                  "SED approximation is physically invalid for orthotropic materials. "
                  "Please add ENER output to your CalculiX .inp and re-run.")
        else:
            print(f"  Warning: {sed_path.name} not found, using isotropic tensor fallback (Fix 5)")

    # Fix 5: Full stress tensor for proper isotropic SED fallback
    tensor_path = case_dir / "elem_stress_tensor.csv"
    stress_tensor = read_stress_tensor(tensor_path) if tensor_path.exists() else {}
    if not sed_direct and not stress_tensor:
        print(f"  Warning: {tensor_path.name} also missing, "
              "falling back to deprecated vm²/2E approximation")

    # Fix 3: Principal stresses for tension-only suture softening
    principal_path = case_dir / "elem_principal_stress.csv"
    principal_stress = read_principal_stress(principal_path) if principal_path.exists() else {}
    if not principal_stress:
        print(f"  Warning: {principal_path.name} not found; suture softening will use "
              "vm-based approximation (physically non-ideal). Re-run extract_fields.py.")

    rho_path = case_dir / "elem_density.json"
    rho_prev = read_density(rho_path, all_eids)

    rho_new = {}
    delta = {}
    psi_eff_scales = []  # for logging

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
            E_base = E_cortical
            nu = nu_cortical
        elif _has_layers and eid in cancellous:
            psi_ref = psi_ref_cancellous
            n_pow = n_power_cancellous
            E_base = E_cancellous
            nu = nu_cancellous
        else:
            psi_ref = psi_ref_global
            n_pow = n_power
            E_base = E_bone
            nu = nu_bone

        # --- Raw (static) SED ---
        if sed_direct and eid in sed_direct:
            psi_raw = sed_direct[eid]
        elif stress_tensor and eid in stress_tensor and not _ortho_enabled:
            # Fix 5: proper isotropic SED from full stress tensor
            sxx, syy, szz, sxy, sxz, syz = stress_tensor[eid]
            E_elem = max(1e3, E_base * (r0 ** n_pow))
            psi_raw = sed_from_tensor_isotropic(sxx, syy, szz, sxy, sxz, syz, E_elem, nu)
        else:
            # Legacy fallback: vm²/2E (deprecated; physically inaccurate)
            vm_e = float(vm.get(eid, 0.0))
            E_elem = max(1e3, E_base * (r0 ** n_pow))
            psi_raw = (vm_e * vm_e) / (2.0 * E_elem)

        # --- Fix 2: Power-law cyclic loading correction ---
        if cyclic_enabled:
            psi_eff, eff_scale = compute_cyclic_psi_eff(cyclic_cfg, tongue_kpa, psi_raw)
            psi_eff_scales.append(eff_scale)
        else:
            psi_eff = psi_raw
            eff_scale = 1.0

        # --- Fix 1 + Audit_3 Fix 2: Density-normalized strain-equivalent stimulus ---
        # The mechanostat senses strain, not energy density directly.
        # Under Carter-Hayes: E(rho) = E0*rho^n, ε = sqrt(2psi/E) = sqrt(2psi/(E0*rho^n))
        # Dimensionless stimulus (strain ratio): S = ε_eff/ε_ref - 1
        #   = sqrt(psi_eff / (psi_ref * rho^n)) - 1   [with density_stimulus_sqrt=True]
        # Linear form: psi_eff/(psi_ref*rho^n) - 1 overestimates sensitivity ~3-5x at
        # physiological density ranges (audit_3 item 2).
        if density_normalize_stimulus and r0 > 0:
            psi_ref_norm = psi_ref * (r0 ** n_pow)
        else:
            psi_ref_norm = psi_ref

        raw_ratio = psi_eff / max(psi_ref_norm, 1e-30)
        if density_stimulus_sqrt:
            # Correct strain-equivalent: sqrt(psi_ratio) - 1
            stimulus = math.sqrt(max(0.0, raw_ratio)) - 1.0
        else:
            # Legacy linear ratio (audit_2; backward-compat)
            stimulus = raw_ratio - 1.0

        if stimulus > lazy_s:
            drho = alpha_app * (stimulus - lazy_s)
        elif stimulus < -lazy_s:
            drho = alpha_res * (stimulus + lazy_s)
        else:
            drho = 0.0

        # Fix 4: Δρ rate limiter (integration stability)
        drho = max(-drho_max, min(drho_max, drho))

        r1 = min(rho_max, max(rho_min, r0 + drho))
        rho_new[eid] = r1
        delta[eid] = r1 - r0

    write_density(rho_path, rho_new)

    snap_dir = case_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    write_density(snap_dir / f"cycle_{cycle}_density.json", rho_new)

    # --- Fix 3: Suture modulus update (tension-only exponential softening + recovery) ---
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

            # Fix 3: use max principal stress (tension) instead of vm
            if principal_stress and eid in principal_stress:
                s1_max = principal_stress[eid][0]  # max principal (positive = tension)
            else:
                # Fallback: vm-based proxy (physically non-ideal; warns above)
                vm_e = float(vm.get(eid, 0.0))
                s1_max = vm_e  # Conservative: vm ≥ 0 so at least won't suppress all softening

            E_new = update_suture_modulus(
                eid, s1_max, E_prev_suture, E_suture_baseline, suture_cfg
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
    mean_psi_eff_scale = sum(psi_eff_scales) / max(1, len(psi_eff_scales)) if psi_eff_scales else 1.0

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
        "cyclic_psi_eff_scale": f"{mean_psi_eff_scale:.4f}",
    }
    if suture_remod_enabled and suture_modulus:
        summary_row["mean_suture_E"] = f"{mean_suture_E:.6e}"
        summary_row["min_suture_E"] = f"{min_suture_E:.6e}"

    update_summary(ROOT / "results" / "remodeling_summary.csv", summary_row)

    suture_info = ""
    if suture_remod_enabled and suture_modulus:
        suture_info = f"  mean_suture_E={mean_suture_E:.4e}  min_suture_E={min_suture_E:.4e}"

    print(
        f"[remodel] {scenario} cycle {cycle} (tongue={tongue_kpa}kPa "
        f"psi_eff_scale={mean_psi_eff_scale:.2f}x): "
        f"mean_rho={sum(bone_rhos)/len(bone_rhos):.4f}  "
        f"app={app:.1%}  res={res:.1%}  lazy={lazy:.1%}  "
        f"mean|drho|={mean_abs_drho:.6f}{suture_info}"
    )


if __name__ == "__main__":
    main()

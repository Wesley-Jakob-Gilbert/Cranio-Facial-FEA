# Audit_4 Fix Report

**Date:** 2026-03-30
**Branch:** audit_4_fixes
**Files Changed:**
- `post/bone_remodeling.py`
- `solver/make_inp.py`
- `configs/remodeling.yaml`

---

## Fix A4-1: Continuous Density-Driven Modulus Law

**Issue:** E(ρ) was assigned based on element set membership (ESET_CORTICAL → E=15 GPa×ρ³, ESET_CANCELLOUS → E=0.5 GPa×ρ²), creating a non-physical modulus discontinuity at the cortical/cancellous boundary. A resorbed cortical element at ρ=0.4 jumped from ~960 MPa to ~80 MPa if reclassified, inducing artificial stress concentrations.

**Fix:**
- Added `continuous_E_from_density(rho, cfg)` in `bone_remodeling.py` — returns E(ρ) and n_eff based purely on density:
  - ρ ≥ `rho_transition` (default 1.2): E = E_cortical × ρ^n_cortical (15 GPa × ρ³)
  - ρ < `rho_transition`: E = E_cancellous_derived × ρ^n_cancellous
  - C0 continuity enforced by deriving E_cancellous = E_cortical × ρ_t^(n_cortical−n_cancellous), ensuring E matches at the transition point
- `make_inp.py` bin material assignment uses the same `_continuous_E_from_rho()` helper
- `psi_ref` still layer-set-based (physiological homeostatic setpoints differ)
- Config: `rho_transition: 1.2`, `continuous_modulus_enforce_c0: true`

**Impact:** Eliminates spurious stress concentrations at layer boundaries; allows true "corticalization" as trabecular bone densifies under increased load.

---

## Fix A4-2: Progressive Suture Ossification (Synostosis)

**Issue:** Recovery path in `update_suture_modulus()` clamped at `E_baseline` (50 MPa). Sustained compressive loading would never produce synostosis — the model could not represent craniosynostosis.

**Fix:**
- Under compressive loading (s1 ≤ 0): E targets `E_bone_max_pa` (5 GPa) at `recovery_rate × ossification_rate_factor = 0.05 × 0.2 = 0.01 fraction/cycle`
- Under sub-threshold tensile loading (0 < s1 ≤ threshold): E recovers toward fibrous `E_baseline` at `recovery_rate = 0.05`
- Clamp range extended to `[E_min, E_bone_max_pa]`
- Config: `suture_remodeling.E_bone_max_pa: 5.0e9`, `ossification_rate_factor: 0.2`

**Impact:** Enables simulation of craniosynostosis under chronic compressive load (e.g., premature mewing with tongue-forward posture). Ossification is 5× slower than fibrous recovery, reflecting the chronic mineralization timeline.

---

## Fix A4-3: Masticatory Amplification for Cyclic SED

**Issue:** `compute_cyclic_psi_eff` scaled peak SED as a pure pressure ratio of resting tongue load. This ignores the dominant masticatory muscle contribution during swallowing (masseter + temporalis co-contraction), which fundamentally changes the stress field orientation, not just its magnitude.

**Fix:**
- Added `masticatory_amplification_factor` (default 2.5) in `compute_cyclic_psi_eff()` — multiplies `psi_peak` before cycle-count scaling
- Based on Bakke 1993 (deglutition EMG showing ~40-60% maximal jaw-muscle activation) and van Eijden 1991
- Factor of 2.5 is conservative interim; ideal fix is a dedicated FEA load case with jaw-muscle forces during swallowing
- Config: `cyclic_loading.masticatory_amplification_factor: 2.5`
- Set to 1.0 to restore audit_3 behavior

**Impact:** Swallow-driven SED is now ~2.5× higher than the pure tongue-pressure model predicted. This increases the cyclic osteogenic stimulus, especially in the maxillary body and pterygoid processes where jaw muscles attach.

---

## Fix A4-4: Osteoclast/Osteoblast Kinetic Asymmetry

**Issue:** `alpha_resorption = 0.002` vs `alpha_apposition = 0.001` (2:1 ratio). Physiologically, osteoclasts work 10–100× faster per BMU than osteoblasts. The 2:1 ratio significantly underestimates bone loss under disuse.

**Fix:**
- `alpha_resorption: 0.005` (was 0.002) — 5:1 ratio vs apposition
- `drho_max_per_cycle: 0.03` (was 0.05) — tighter limiter to prevent numerical oscillation with faster resorption
- Rationale: 5:1 reflects per-cycle Δρ rates over physiological months; the raw BMU 10-100× figure includes osteoclast/osteoblast surface coverage differences already partly absorbed by the lazy-zone parameterization
- Reference: Parfitt 1994 (BMU activation frequency), Eriksen 2010 (remodeling space temporal dynamics)

**Impact:** Disuse scenarios (mouth-breathing, low tongue activity) will show more rapid and realistic bone loss. Apposition under mewing will be slower to accumulate, creating a more asymmetric remodeling trajectory.

---

## Configuration Summary (remodeling.yaml new values)

| Parameter | Old | New |
|-----------|-----|-----|
| alpha_resorption | 0.002 | 0.005 |
| drho_max_per_cycle | 0.05 | 0.03 |
| rho_transition | — | 1.2 |
| continuous_modulus_enforce_c0 | — | true |
| suture_remodeling.E_bone_max_pa | — | 5.0e9 |
| suture_remodeling.ossification_rate_factor | — | 0.2 |
| cyclic_loading.masticatory_amplification_factor | — | 2.5 |


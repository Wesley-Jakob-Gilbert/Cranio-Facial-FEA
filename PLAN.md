# cranio_FEA — North Star Upgrade Plan
# Week of 2026-03-08

## North Star
A mechanobiologically-grounded simulation of adolescent cranio-facial bone remodeling
over years of loading, comparing proper tongue posture ("mewing") against mouth breathing,
with animated output showing where bone apposition and resorption occur.

## Scientific Framing
- **Remodeling law:** Huiskes strain-adaptive density rule (SED-stimulus driven)
- **Growth site:** Midpalatal + lateral sutural zones modeled as low-modulus compliant layers
- **Age group:** Adolescent (sutures open, high sutural compliance)
- **Scenarios:** (A) tongue-on-palate sustained load vs. (B) no palate load (mouth breathing)
- **Output:** Comparative — not clinical. All parameters explicit and exposed.

---

## Labor Division (Two Agents)

| Domain | Owner | Files |
|---|---|---|
| Sutural zone mesh tagging | **This agent** | `mesh/build_mesh.py`, `configs/geometry.json` |
| Multi-material INP generation | **This agent** | `solver/make_inp.py`, `configs/material.yaml` |
| Mouth-breathing scenario | **This agent** | `configs/loads.yaml` |
| Bone remodeling engine | **OpenClaw agent** | `post/bone_remodeling.py` (new) |
| Iterative pipeline loop | **OpenClaw agent** | `scripts/remodeling_loop.sh` (new) |
| Animation + visualization | **OpenClaw agent** | `post/make_animation.py` (new), `post/make_3d_visuals.py` |

### Interface Contract (mesh_data.json additions)
Both agents agree on these new keys before coding begins:
```json
{
  "sets": {
    "ESET_BONE":        [...],   // element IDs: bulk bone (updated by remodeling)
    "ESET_SUTURE_MID":  [...],   // element IDs: midpalatal suture strip
    "ESET_SUTURE_LAT":  [...]    // element IDs: bilateral lateral sutural strips
  }
}
```
OpenClaw's remodeling engine reads these keys. This agent writes them. Agreed.

---

## Day-by-Day Plan

### Day 1 — Sutural Zone Mesh Tagging
**Owner: This agent**

**Goal:** Extend `mesh/build_mesh.py` to identify and tag sutural element sets.
The midpalatal suture runs anteroposterior along the sagittal midline (j ≈ ny/2).
Lateral sutures run along the zygomatic junction (i ≈ nx * 0.7, all j, all k).

**Changes:**
1. `mesh/build_mesh.py`
   - Add `ESET_BONE`, `ESET_SUTURE_MID`, `ESET_SUTURE_LAT` element sets
   - Midpalatal: elements where `j == ny//2` (1 element thick sagittal strip)
   - Lateral: elements where `i >= int(0.7 * nx)` (anterior-lateral band, both sides)
   - `ESET_BONE` = all elements not in any suture set
   - Write all three to `mesh_data.json["sets"]`

2. `configs/geometry.json`
   - Add `suture_mid_thickness` (default: 1 element) and `suture_lat_width` (default: 2 elements)
   - Makes sutural zone width configurable without code changes

**Deliverable:** `mesh/mesh_data.json` with `ESET_BONE`, `ESET_SUTURE_MID`, `ESET_SUTURE_LAT`

---

### Day 2 — Multi-Material INP Generation + Material Config
**Owner: This agent**

**Goal:** `solver/make_inp.py` writes separate `*MATERIAL` and `*SOLID SECTION` blocks
for bone vs. suture zones, so the solver sees a mechanically distinct sutural layer.

**Changes:**
1. `configs/material.yaml` — extend to multi-material:
   ```yaml
   materials:
     bone:
       youngs_modulus_pa: 1.0e9
       poisson_ratio: 0.3
     suture:
       youngs_modulus_pa: 5.0e7   # ~20x softer than bone (adolescent open suture)
       poisson_ratio: 0.40        # higher compliance
   ```

2. `solver/make_inp.py`
   - Read `ESET_BONE`, `ESET_SUTURE_MID`, `ESET_SUTURE_LAT` from mesh_data.json
   - Write `*ELSET` blocks for each zone
   - Write two `*MATERIAL` blocks (BONE, SUTURE)
   - Write two `*SOLID SECTION` blocks assigning each elset to its material
   - Remove the single `EALL` material assignment
   - The remodeling loop will later update the per-element Young's modulus by
     regenerating this file with a new material field (density → E mapping)

3. Support an optional `elem_modulus.json` sidecar: if present, `make_inp.py` reads
   per-element E values from it and overrides the default material block per element.
   This is the hook the remodeling engine uses to feed updated stiffness back in.

**Deliverable:** `solver/<case>.inp` with distinct bone and suture material zones

---

### Day 3 — Mouth-Breathing Scenario + Remodeling Scenario Configs
**Owner: This agent**

**Goal:** Add the two comparison scenarios that drive the whole study.

**Changes:**
1. `configs/loads.yaml` — add two dedicated remodeling study cases:
   ```yaml
   # Remodeling study cases (used by remodeling_loop.sh)
   mewing:
     tongue_kpa: 2.0        # sustained medium palate pressure
     muscle_force_n: 0.0
     scenario: mewing
   mouth_breathing:
     tongue_kpa: 0.0        # zero palate load
     muscle_force_n: 0.0
     scenario: mouth_breathing
   ```
   - `tongue_kpa: 0.0` for mouth breathing produces zero palate nodal forces,
     which is the correct mechanical analog for no tongue-palate contact.

2. `solver/make_inp.py` — handle `tongue_kpa: 0.0` gracefully (no `*CLOAD` for
   palate nodes in that case, already works via the force calc but confirm/test).

**Deliverable:** Two new solver-ready cases representing the study's core comparison

---

### Day 4 — Bone Remodeling Engine (OpenClaw)
**Owner: OpenClaw agent**

**Goal:** Replace `post/simulate_adaptation.py` with a physically grounded
iterative remodeling module: `post/bone_remodeling.py`.

**Algorithm — Huiskes Strain-Adaptive Remodeling:**
```
For each element e at iteration t:
  ψ_e = SED at element e  (= σ_e² / 2E_e, computed from von Mises + E)
  stimulus_e = (ψ_e / ψ_ref) - 1
  Δρ_e = α * stimulus_e              # density rate
  ρ_e(t+1) = clamp(ρ_e(t) + Δρ_e, [ρ_min, ρ_max])
  E_e(t+1) = E_bone * ρ_e(t+1)^n    # power-law: typically n=2 or 3
```

**Parameters (all exposed in a new `configs/remodeling.yaml`):**
```yaml
remodeling:
  ψ_ref_pa: 0.005          # reference SED (lazy zone center)
  α: 0.02                  # adaptation rate per cycle
  ρ_min: 0.4               # minimum relative density (severe resorption)
  ρ_max: 1.5               # maximum relative density (apposition limit)
  n_power: 2               # power-law exponent for E(ρ)
  E_bone_pa: 1.0e9         # baseline bone modulus
  E_suture_pa: 5.0e7       # suture modulus (held fixed, not remodeled)
  cycles: 12               # number of remodeling iterations (≈ months/cycle)
  cycle_label: months      # for axis labeling in plots
```

**I/O:**
- Reads: `results/<case>/elem_vm.csv`, `mesh_data.json`, `configs/remodeling.yaml`
- Reads (if exists): `results/<case>/elem_density.json` (from previous iteration)
- Writes: `results/<case>/elem_density.json` (updated ρ per element)
- Writes: `solver/elem_modulus.json` (E per element, for make_inp.py to consume)

**Suture handling:** Elements in `ESET_SUTURE_*` are NOT remodeled — their E is
held at `E_suture_pa` and their density at 1.0 throughout. They respond to load
but do not undergo density-driven stiffness change (they grow via cellular biology,
not density remodeling — we flag this caveat explicitly).

---

### Day 5 — Iterative Pipeline Loop (OpenClaw)
**Owner: OpenClaw agent**

**Goal:** A shell script that runs the full FEA → extract → remodel → regenerate → loop
cycle for both scenarios, storing snapshots at each iteration.

**New file:** `scripts/remodeling_loop.sh`
```bash
# For each scenario (mewing, mouth_breathing):
#   For each cycle 1..N:
#     1. python3 solver/make_inp.py (reads elem_modulus.json if present)
#     2. ./solver/run_case.sh <scenario>
#     3. python3 post/extract_fields.py
#     4. python3 post/bone_remodeling.py <scenario> <cycle>
#        (writes elem_density.json + elem_modulus.json for next iteration)
#     5. cp results/<scenario>/elem_density.json snapshots/<scenario>/cycle_<N>.json
```

**Snapshot storage:** `results/<scenario>/snapshots/cycle_<N>_density.json`
Each snapshot is a `{elem_id: rho}` dict. These feed the animation.

---

### Day 6 — Animation + Visualization (OpenClaw)
**Owner: OpenClaw agent**

**Goal:** Generate the headline artifact — a side-by-side animation of mewing vs.
mouth-breathing bone density evolution over the simulated period.

**New file:** `post/make_animation.py`

**Output artifacts:**
1. `results/mewing/density_evolution.png` — multi-panel grid, one panel per cycle
2. `results/mouth_breathing/density_evolution.png` — same
3. `results/comparison_animation.gif` — side-by-side animated GIF:
   - Left panel: mewing scenario density field (3D scatter, color = Δρ)
   - Right panel: mouth breathing density field
   - Color scale: blue = resorption, white = neutral, red = apposition
   - Frame title: "Month N" (or "Year N" depending on cycle_label)
4. `results/remodeling_summary.csv` — per-cycle: scenario, cycle, mean_rho, max_rho,
   min_rho, apposition_fraction, resorption_fraction

**Visualization pattern:** Extend existing `make_3d_visuals.py` centroid-scatter
approach. Color field = `ρ(cycle) - ρ(0)` (delta from baseline), not absolute ρ.
This makes apposition/resorption zones immediately readable.

---

### Day 7 — Integration, Sensitivity, and Documentation
**Both agents**

**Goal:** End-to-end run from clean state, sensitivity sweep, final documentation.

**Tasks:**
1. **Integration test:** `./scripts/remodeling_loop.sh` runs clean from blank results/
2. **Sensitivity sweep** (this agent): vary `suture.youngs_modulus_pa` across
   [1e7, 5e7, 1e8] Pa to show how sutural stiffness (i.e., age proxy) affects the
   magnitude of predicted response
3. **Update `reproduce_all.sh`** (OpenClaw): add remodeling loop as step [9/11]
   after existing pipeline steps
4. **Update `RESULTS_MANIFEST.md`**: include density snapshots and animation in SHA256 manifest
5. **Caveats file** (this agent): `results/SCIENTIFIC_CAVEATS.md`
   - Force magnitudes are estimated
   - Suture growth is modeled as mechanical compliance only (not cellular biology)
   - Mouth breathing is approximated as zero palate pressure (not negative pressure)
   - Results are directional/comparative — not quantitative clinical predictions
   - Validated only for adolescent-range sutural compliance parameters

---

## Deliverables by End of Week

| Artifact | Description |
|---|---|
| `mesh/mesh_data.json` | Sutural element sets tagged (ESET_BONE, ESET_SUTURE_MID, ESET_SUTURE_LAT) |
| `solver/<case>.inp` | Multi-material deck with distinct bone + suture zones |
| `configs/material.yaml` | Bone + suture material specs |
| `configs/loads.yaml` | `mewing` and `mouth_breathing` scenarios added |
| `configs/remodeling.yaml` | All remodeling parameters exposed |
| `post/bone_remodeling.py` | Huiskes SED-stimulus remodeling engine |
| `scripts/remodeling_loop.sh` | Full iterative FEA-remodel pipeline |
| `results/<scenario>/snapshots/` | Per-cycle density field snapshots |
| `results/comparison_animation.gif` | Side-by-side mewing vs. mouth-breathing animation |
| `results/remodeling_summary.csv` | Per-cycle aggregate metrics |
| `results/SCIENTIFIC_CAVEATS.md` | Explicit assumptions and limitations |

---

## What This Is and Is Not

**IS:** A mechanobiologically self-consistent hypothesis explorer for adolescent
craniofacial loading effects. Useful for research communication, sensitivity
analysis, and as a foundation for future validated clinical tools.

**IS NOT:** A clinical predictor. The remodeling law is real; the parameter values
are estimated. Outputs should be read as directional, not quantitative.

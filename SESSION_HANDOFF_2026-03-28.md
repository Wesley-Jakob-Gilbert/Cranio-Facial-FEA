# Session Handoff — 2026-03-28

## What Happened This Session

This session continued the Tier 2 realism + PI critic loop that the user requested as an overnight autonomous run. The session spanned two context windows (compaction happened mid-session).

### Commits Made (both pushed to `origin/main`)

1. **`a60b77c` — Tier 2 realism** — layered bone, tooth roots, per-layer psi_ref, suture widening
2. **`ff04b40` — PI critic round 1 fixes** — true SED from CalculiX ENER, gravity loading, per-layer n_power

### What Was Done

#### Tier 2 Realism (commit a60b77c)
- **Cortical/cancellous bone split**: k=0,k=nz-1 surface layers = cortical (E=15 GPa), interior = cancellous (E=0.5 GPa)
- **Tooth root dentin inclusions**: E=20 GPa at anatomical tooth positions (7 tooth types, j=[0,1,2])
- **Element set partition**: ESET_CORTICAL(746) + ESET_CANCELLOUS(746) + ESET_TOOTH_ROOT(340) + ESET_SUTURE_MID(176) + ESET_SUTURE_LAT(104) = 2112
- **Per-layer psi_ref**: cortical and cancellous bone have different homeostatic SED setpoints
- **Suture strain-dependent widening**: vm_stress / E_suture > threshold => E_suture softens
- **Lazy zone widened** from 15% to 50% (biologically, bone doesn't respond to minor perturbations)
- **Parameter sensitivity scripts** added: `scripts/parameter_sensitivity.py`, `post/make_sensitivity_plots.py`
- **4-branch material writing** in make_inp.py: density-binned-layered, density-binned-legacy, baseline-layered, baseline-legacy

#### PI Critic Round 1 (commit ff04b40)
Spawned an Opus subagent as a "stingy PI who specializes in bone remodeling simulation and computational biophysics." It produced a structured critique with 3 critical, 5 major, 6 moderate issues.

**Issues fixed:**
- **C1 (CRITICAL)**: SED was computed as `vm^2 / (2E)` which is only correct for uniaxial stress. True SED from CalculiX `*EL PRINT ENER` output is now used. The old approximation was **2-3x too high**. Pipeline: make_inp.py writes ENER request → .dat file contains "internal energy density" section → extract_fields.py parses it into `elem_sed.csv` → bone_remodeling.py reads it directly instead of computing from von Mises.
- **C2 (CRITICAL, partial)**: Mouth-breathing had zero load → trivially deterministic. Added gravity body force via `*DLOAD GRAV` and `*DENSITY` cards on all materials. Gravity alone (~540 Pa self-weight) is still too small to prevent uniform resorption — baseline masticatory loads still needed for a truly meaningful comparison.
- **M4**: Suture remodeling was inactive because `strain_threshold=0.001` was 3.3x above max observed strain (0.0003). Lowered to 0.00015, raised beta 0.5→5.0. Suture softening now activates: min_suture_E drops from 50 MPa to 49.3 MPa over 12 cycles.
- **m2**: Suture Poisson ratio 0.40→0.30 (prevents volumetric locking in C3D8 elements)
- **m6**: Per-layer Carter-Hayes exponent: n=3 for cortical (Carter & Hayes 1977), n=2 for cancellous
- **psi_ref recalibrated** to true SED medians: cortical 0.24 Pa, cancellous 0.02 Pa

**Issues NOT yet fixed (remaining for next session):**
- **C3**: No mesh convergence study with current model (script exists at `scripts/mesh_convergence.py` but hasn't been re-run with Tier 2 materials)
- **M1**: Legacy E_bone=1 GPa path still exists in bone_remodeling.py fallback
- **M2**: Alpha values (0.001/0.002) uncited — no literature reference provided
- **M3**: Cycle duration undefined — "months" label has no physical calibration
- **M5**: extract_fields.py processes all cases every iteration (not just current scenario)
- **m1**: Tributary area still uses uniform area_per_node instead of per-node tributary areas
- **m3**: Fixed BCs may be too restrictive (spring BCs suggested)
- **m4**: Gravity added but baseline masticatory loads still missing for mouth_breathing
- **m5**: Integration-point averaging loses stress gradient info
- **S1-S6**: Various suggestions (total strain energy tracking, element quality gating, adaptive alpha, RME validation, pressure gradient citation, ParaView output)

### Current Remodeling Results (12 cycles, true SED)

**Mewing:**
- 55% apposition, 15% resorption, 30% lazy zone
- mean_rho: 1.000 → 1.039 (3.9% net density gain)
- max_rho hits ceiling (1.50) by cycle 8
- Suture softening: min_suture_E = 49.3 MPa (1.4% reduction)

**Mouth breathing (gravity only):**
- 100% resorption (gravity too weak to maintain bone)
- mean_rho: 1.000 → 0.988 (1.2% density loss)
- Suture unchanged (no load = no strain)

### Animation GIFs Regenerated
- `results/mewing/density_evolution.gif`
- `results/mouth_breathing/density_evolution.gif`
- `results/comparison_animation.gif` (side-by-side north star artifact)

---

## Agent Orchestration

### Team Structure
- **The Architect** (Claude Opus 4.6, this agent): orchestrator, strategic decisions, QA coordination
- **Neo** (Claude Sonnet 4.6): upstream — mesh, solver, configs (offline this session)
- **Morpheus** (GPT5.2 Codex, OpenClaw): downstream — remodeling engine, loop, animation (offline)
- **Trinity** (Sonnet subagent, spawned on demand): QA reviewer

### Subagents Spawned This Session
1. **SED analysis agent** (Opus) — Computed per-layer SED statistics from mewing baseline. Determined cortical median SED = 0.524 Pa (vm²/2E approx), cancellous = 0.046. Recommended per-layer psi_ref.
2. **CCX ENER research agent** (Opus) — Confirmed CalculiX can output true SED via `*EL PRINT ENER`. Tested with minimal model. Recommended ENER approach over full tensor computation.
3. **Suture diagnosis agent** (Sonnet) — Found strain_threshold=0.001 was 3.3x above max observed strain. Recommended lowering to 0.00015.
4. **PI Critic agent** (Opus) — Full structured review with C1-C3, M1-M5, m1-m6, S1-S6 issues.
5. **PI Critic round 2 agent** (Opus) — Hit usage limit before completing. No output.
6. **Worktree agents from previous context** (before compaction) — agent-AC (cortical/cancellous), agent-B (suture widening), agent-D (sensitivity scripts). Used git worktree isolation.

### User's Standing Instructions
- **Autonomous overnight operation**: Don't ask for confirmation. Take initiative on next logical task.
- **Critique-edit loop**: Spawn PI critic → implement fixes → re-run critic → repeat until morning or tokens exhausted
- **Cron job**: `trig_01T5VpH4Fzt9ASfeRVKDwcEe` was created to run every 2 hours. May need verification.
- **Use Opus for all subagents**: User has plenty of tokens, don't spare compute
- **No plan approval needed**: Enter planning mode but don't wait for user approval

---

## Key File Changes (what was modified and why)

| File | Changes |
|------|---------|
| `configs/remodeling.yaml` | per-layer psi_ref (0.24/0.02), per-layer n_power (3/2), lazy_zone_s 0.50, suture threshold 0.00015, beta 5.0 |
| `configs/material.yaml` | suture nu 0.40→0.30; added cortical/cancellous/tooth_root materials |
| `configs/loads.yaml` | `gravity: true` on mouth_breathing case |
| `configs/geometry.json` | tooth_roots section with 7 anatomical i-range positions |
| `post/bone_remodeling.py` | true SED from elem_sed.csv, per-layer n_power, per-layer psi_ref, suture softening |
| `post/extract_fields.py` | parses "internal energy density" section from .dat → writes elem_sed.csv |
| `solver/make_inp.py` | ENER output request, *DENSITY cards, *DLOAD GRAV, per-layer n_power, _write_elastic_density helper |
| `mesh/build_mesh.py` | ESET_CORTICAL/CANCELLOUS/TOOTH_ROOT subdivision, partition assertion |
| `scripts/remodeling_loop.sh` | suture_modulus.json feedback path |
| `post/make_animation.py` | Poly3DCollection solid rendering (from Tier 1) |

---

## What To Do Next (prioritized)

1. **PI Critic Round 2**: Re-run the critic review to verify fixes and identify new issues
2. **C3 — Mesh convergence study**: Run `scripts/mesh_convergence.py` with current Tier 2 materials (2112 vs 16896 elements)
3. **C2 — Baseline masticatory loads**: Add realistic chewing forces (~50-100 N bilateral, 1-2 Hz cycle) to mouth_breathing. Without this, the mewing-vs-mouth_breathing comparison is "something vs nothing"
4. **M5 — extract_fields.py scope**: Add optional case-name argument so it only processes the current scenario during remodeling loop
5. **m1 — Per-node tributary areas**: Accumulate 1/4 of each adjacent quad face area to each node (consistent nodal force approach)
6. **Parameter sensitivity sweep**: Run `scripts/parameter_sensitivity.py` and generate plots with `post/make_sensitivity_plots.py`
7. **Literature citations**: Add references for alpha, psi_ref, n_power values to SCIENTIFIC_CAVEATS.md

---

## Key Interface Contracts (current state)

- `mesh_data.json["sets"]` has: ESET_BONE (=cortical|cancellous), ESET_CORTICAL, ESET_CANCELLOUS, ESET_TOOTH_ROOT, ESET_SUTURE_MID, ESET_SUTURE_LAT
- `elem_density.json`: `{"elem_id_str": rho_float}` — bone element densities
- `suture_modulus.json`: `{"elem_id_str": E_float}` — per-element suture modulus
- `elem_sed.csv`: `elem_id, sed_avg_pa` — true SED from CalculiX ENER (new this session)
- `elem_vm.csv`: `elem_id, vm_avg_pa` — von Mises stress (kept for visualization)
- Remodeling loop copies `results/<scenario>/elem_density.json` → `solver/elem_density.json` each cycle
- Remodeling loop copies `results/<scenario>/suture_modulus.json` → `solver/suture_modulus.json` each cycle

---

## Environment Notes
- CalculiX (`ccx`) is installed and working at `/usr/bin/ccx`
- Python deps: matplotlib, numpy, pyyaml all available
- Git remote: `https://github.com/Wesley-Jakob-Gilbert/Cranio-Facial-FEA.git`
- Branch: `main` (all work on main, no feature branches)
- WSL2 Linux environment

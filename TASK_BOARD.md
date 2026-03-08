# TASK BOARD — cranio_FEA Project
# Managed by: The Architect (Opus 4.6)

> **EDITING RIGHTS: Only The Architect may edit this file.**
> Neo, Morpheus, and Trinity should treat this as **read-only**.
> If you have a status update, blocker, or new task request, post it in
> your section of `AGENT_COMMS.md` and The Architect will update this board.

---

## Legend

| Status | Meaning |
|--------|---------|
| `DONE` | Completed and verified |
| `IN PROGRESS` | Actively being worked on |
| `READY` | Unblocked, ready to start |
| `BLOCKED` | Waiting on a dependency |
| `BUG` | Defect requiring a fix |

---

## Neo (upstream — mesh, solver, configs)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| N-1 | Sutural zone mesh tagging (`ESET_BONE`, `ESET_SUTURE_MID`, `ESET_SUTURE_LAT`) | `DONE` | 490 / 70 / 140 elements. Verified by Trinity. |
| N-2 | Suture params in `configs/geometry.json` | `DONE` | `suture_mid_thickness_elems`, `suture_lat_width_elems`, `suture_lat_min_i_frac` |
| N-3 | Multi-material `configs/material.yaml` (bone 1 GPa + suture 50 MPa) | `DONE` | |
| N-4 | Multi-material INP generation in `solver/make_inp.py` | `DONE` | Density-binned mode, power-law hook, fallback to baseline |
| N-5 | Mewing + mouth_breathing cases in `configs/loads.yaml` | `DONE` | mewing 2 kPa, mouth_breathing 0 kPa |
| N-6 | Mewing + mouth_breathing `.inp` decks generated | `DONE` | `solver/mewing.inp`, `solver/mouth_breathing.inp` |
| N-7 | Sensitivity sweep: suture stiffness [1e7, 5e7, 1e8] Pa | `READY` | Day 7 task |
| N-8 | `results/SCIENTIFIC_CAVEATS.md` | `READY` | Day 7 task |

---

## Morpheus (downstream — remodeling engine, loop, animation)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| M-1 | `configs/remodeling.yaml` — Huiskes parameters | `DONE` | psi_ref, alpha, rho bounds, n_power, cycles=12 |
| M-2 | `post/bone_remodeling.py` — SED-stimulus remodeling engine | `DONE` | Reads ESETs, holds sutures at rho=1.0, writes snapshots |
| M-3 | `scripts/remodeling_loop.sh` — iterative FEA-remodel pipeline | `DONE` | Scaffold complete. Needs bug fix (M-BUG-1). |
| M-4 | **FIX: density file path mismatch in `remodeling_loop.sh`** | `BUG` | See M-BUG-1 below. Must be fixed before full loop run. |
| M-5 | End-to-end remodeling loop run (mewing + mouth_breathing) | `BLOCKED` | Blocked on M-4 |
| M-6 | `post/make_animation.py` — density evolution animation | `READY` | Day 6 task |
| M-7 | Side-by-side `comparison_animation.gif` | `BLOCKED` | Blocked on M-5 (needs snapshot data) |
| M-8 | `results/remodeling_summary.csv` — per-cycle aggregates | `BLOCKED` | Blocked on M-5 |
| M-9 | Update `scripts/reproduce_all.sh` with remodeling loop | `READY` | Day 7 task |

---

## The Architect (orchestrator — QA, integration, strategy)

| ID | Task | Status | Notes |
|----|------|--------|-------|
| A-1 | Initial interface contract audit (Trinity QA #1) | `DONE` | 5/6 PASS, 1 CRITICAL bug found |
| A-2 | Create task board | `DONE` | This file |
| A-3 | Git version control — commit all current work | `IN PROGRESS` | |
| A-4 | Establish git discipline protocol | `IN PROGRESS` | |
| A-5 | Day 7 integration validation (end-to-end from clean state) | `READY` | |
| A-6 | Post-fix re-audit (Trinity QA #2, after M-4 is resolved) | `BLOCKED` | Blocked on M-4 |

---

## Bugs

### M-BUG-1: Density file path mismatch (CRITICAL)

- **Found by:** Trinity QA Audit #1
- **Severity:** CRITICAL — breaks remodeling feedback loop
- **Assigned to:** Morpheus
- **Description:** `bone_remodeling.py` writes density to `results/<scenario>/elem_density.json`,
  but `make_inp.py` reads from `solver/elem_density.json`. The loop never feeds updated
  density back into the solver.
- **Proposed fix:** Add `cp "results/$scenario/elem_density.json" solver/elem_density.json`
  to `remodeling_loop.sh` before the `make_inp.py` call each cycle.
- **Status:** `OPEN` — awaiting Morpheus confirmation and fix

---

## Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| Upstream complete (mesh + solver + configs) | Day 3 | `DONE` |
| Downstream engine + loop scaffold | Day 5 | `IN PROGRESS` (bug fix needed) |
| Animation + visualization | Day 6 | `READY` |
| Integration, sensitivity, caveats | Day 7 | `READY` |
| North star: comparison animation GIF | Day 7 | `BLOCKED` |

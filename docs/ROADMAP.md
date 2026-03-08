# Cranio-Maxillary FEA Roadmap (MVP → Full Fidelity)

_Last updated: 2026-03-04_

## Goal
Build a staged finite element modeling workflow to study how simplified oral forces (starting with tongue pressure on the palate/maxilla) may influence craniofacial bone loading patterns, then progressively add realism (jaw muscles, sutures, growth/remodeling assumptions, subject-specific geometry).

---

## Guiding Principle
Start with a **toy model that teaches us the pipeline**, not a clinically realistic solver.

- Keep MVP geometry simple
- Use isotropic linear elastic material for bone
- Apply tongue force as a basic distributed load on palate region
- Focus on: “Can we run, visualize, compare scenarios reliably?”

---

## Scope Levels

## Level 0 — MVP (2 weeks)
### Target outcome
A reproducible scriptable workflow that:
1. Builds/imports a simplified craniofacial geometry (maxilla-focused)
2. Meshed into a valid FE model
3. Solves at least 2-3 load cases (low/medium/high tongue pressure)
4. Produces displacement + stress maps and simple comparative plots

### Simplifications
- Geometry: coarse/approximate facial-maxillary shell/block hybrid or very simplified segmented shape
- Material: single isotropic elastic bone material
- Loads: static normal pressure on palate patch (tongue)
- Boundary conditions: simple skull-base constraints (or posterior fixation proxy)
- No growth law, no remodeling, no anisotropy, no sutures, no muscles yet

### Deliverables
- `README` with assumptions
- FE input generation script(s)
- Load-case config file (JSON/YAML)
- Results notebook/plots (stress/displacement comparison)
- “Known limitations” document

### Proposed schedule (2 weeks)
**Week 1**
- Day 1-2: Define geometry abstraction + solver stack
- Day 3-4: Mesh + boundary condition sanity checks
- Day 5: First successful solve and debug

**Week 2**
- Day 6-7: Implement 3 tongue-force magnitudes
- Day 8-9: Post-processing plots + comparison metrics
- Day 10: Package MVP (docs, scripts, reproducibility pass)

---

## Level 1 — Early Realism (Weeks 3-6)
### Objective
Add first-order biomechanical realism while keeping runtime manageable.

### Additions
- Separate cortical/cancellous regions (still isotropic)
- Better geometry from a template skull/maxilla mesh
- Add simplified jaw muscle resultant forces (masseter/temporalis/medial pterygoid as vectors)
- Add bite/contact proxy load cases (optional)
- Sensitivity analysis on uncertain parameters

### Output
- Scenario matrix: tongue-only vs tongue+jaw loads
- Sensitivity ranking of major parameters
- Updated assumptions + validation checks

---

## Level 2 — Intermediate Realism (Months 2-3)
### Objective
Improve anatomical and mechanical credibility.

### Additions
- Sutures modeled as softer interfaces or contact regions
- Heterogeneous material map (e.g., density-informed where possible)
- Better constraints (TMJ/skull coupling approximations)
- Optional nonlinearities (if needed and stable)

### Output
- More realistic strain/stress distributions
- Robustness checks across mesh density and BC variants

---

## Level 3 — Toward High-Fidelity Product (Months 3-6+)
### Objective
Approach research-grade / translational model quality.

### Additions
- Subject-specific geometry from imaging pipeline
- Muscle force estimation from anatomy/EMG-informed priors
- Bone adaptation/remodeling or growth modeling over pseudo-time
- Calibration/validation against literature or experimental benchmarks
- Uncertainty quantification + reproducibility package

### Output
- Versioned “full model” pipeline
- Validation report
- Decision-grade comparisons and limitations

---

## Practical MVP Success Criteria (2-week checkpoint)
MVP is successful if all are true:
1. End-to-end run in one command/script
2. Three tongue load cases run without solver failure
3. Quantitative outputs exported (max displacement, peak Von Mises or principal stress, ROI averages)
4. Plot/report clearly compares load-case trends
5. Assumptions/limitations are explicit

---

## Suggested Tooling (keep minimal)
- Geometry/mesh: gmsh or prebuilt simple mesh + Python tooling
- Solver: FEniCSx / CalculiX / FEBio (pick one and stick to it for MVP)
- Post-processing: Python (NumPy/Pandas/Matplotlib/PyVista)
- Reproducibility: simple folder conventions + config-driven load cases

---

## Risks & Controls
- **Risk:** Overbuilding realism too early  
  **Control:** Freeze MVP assumptions for 2 weeks.
- **Risk:** Unstable BCs/rigid-body motion  
  **Control:** Start with conservative fixation and test with unit loads.
- **Risk:** Ambiguous interpretation of stress maps  
  **Control:** Predefine comparison metrics and ROIs.

---

## Immediate Next Step (Planning Session)
Before implementation, decide 4 lock-ins:
1. Solver choice for MVP
2. Geometry source (toy CAD vs simplified mesh extraction)
3. Boundary condition convention
4. Tongue pressure range(s) for low/medium/high cases

Once these are locked, implementation can start with low risk of churn.

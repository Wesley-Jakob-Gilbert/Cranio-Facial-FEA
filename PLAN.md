# cranio_FEA Execution Plan (Incremental)

## Project Goal
Build a simplified, reproducible cranio-maxillary FEA MVP to compare bone response trends under low/medium/high tongue pressure.

## Scope Lock (MVP)
- Solver: CalculiX (linear static)
- Geometry: toy maxilla/palate approximation
- Material: isotropic linear elastic bone
- BC: posterior/base fixed-support proxy
- Loads: 0.5, 2.0, 5.0 kPa

## 2-Week Sprint Breakdown

### Week 1: Pipeline skeleton + single-case success
1. Define geometry abstraction and palate ROI
2. Create mesh generation stub and data contracts
3. Generate solver input deck for one load case
4. Solve one case successfully
5. Extract first metrics (max displacement, peak stress)

### Week 2: Three-case comparison + packaging
6. Add low/medium/high case config handling
7. Run all 3 load cases
8. Build comparative plots
9. Validate assumptions + limitations docs
10. Reproducibility run from clean state

## Acceptance Criteria
- 3 load cases complete without solver failure
- Metrics exported per case
- Comparative plots generated
- Reproducible run instructions in README
- Assumptions/limitations explicit

## Task Board (Current)
- [x] Project scaffold created
- [x] Initial configs created
- [x] Geometry builder (toy dimensions + palate/fixed ROI definitions via mesh script)
- [x] Mesh builder
- [x] Input deck generator (CalculiX .inp)
- [x] Single-case solve script
- [x] Metrics extraction script
- [x] Plot script
- [x] Dry-run pipeline command documented

## Risk Controls
- Freeze MVP assumptions until acceptance criteria met
- No sutures/muscles/nonlinearities before MVP complete
- Treat outputs as comparative trends only

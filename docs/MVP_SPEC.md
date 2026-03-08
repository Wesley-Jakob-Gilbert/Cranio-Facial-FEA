# Cranio-Maxillary Tongue-Load FEA — MVP Spec Sheet

_Last updated: 2026-03-04_
_Owner: Wes + Morpheus_

## 1) MVP Objective (2 weeks)
Build a reproducible, simplified FE pipeline that quantifies how increasing tongue pressure on the palate changes displacement and stress patterns in a simplified maxilla/facial-bone model.

**Out of scope (for MVP):** growth/remodeling laws, anisotropy, sutures, subject-specific CT segmentation, nonlinear contact, full jaw-muscle system.

---

## 2) Locked MVP Decisions
1. **Solver:** CalculiX (linear static)
2. **Geometry:** simplified analytic/toy maxilla-palate geometry (coarse anatomical approximation)
3. **Boundary conditions:** posterior/base fixed-support proxy
4. **Load cases (palate pressure):**
   - Low: **0.5 kPa**
   - Medium: **2.0 kPa**
   - High: **5.0 kPa**

---

## 3) Core Assumptions
- Bone represented as a single **isotropic, linear-elastic** material
- Quasi-static loading
- Tongue represented as a distributed normal pressure over a defined palate ROI
- Results used for **relative trend comparison** only (not clinical inference)

---

## 4) Deliverables
By end of MVP:
1. Scripted model build + solve workflow
2. Three successful load-case runs
3. Exported quantitative metrics per case:
   - max displacement
   - peak stress (e.g., von Mises)
   - palate ROI average stress/strain proxy
4. Plots:
   - load vs max displacement
   - load vs peak stress
   - side-by-side stress maps for low/med/high
5. `ASSUMPTIONS_LIMITATIONS.md`
6. `README.md` with one-command (or short command sequence) run instructions

---

## 5) Suggested Repo Layout
```text
cranio_fea/
  README.md
  ASSUMPTIONS_LIMITATIONS.md
  configs/
    material.yaml
    bc.yaml
    loads.yaml
  geometry/
    build_geometry.py
    palate_roi.json
  mesh/
    build_mesh.py
  solver/
    make_inp.py
    run_case.sh
  post/
    extract_metrics.py
    make_plots.py
  results/
    low/
    medium/
    high/
```

---

## 6) Execution Plan (2 Weeks)

## Week 1 — Build/Debug pipeline
- **Day 1:** Create geometry + palate ROI tagging
- **Day 2:** Mesh generation and quality sanity checks
- **Day 3:** Build CalculiX input deck generator (material + BC + load)
- **Day 4:** Run first single-case solve; debug rigid-body/BC issues
- **Day 5:** Validate output extraction for displacement/stress metrics

## Week 2 — Multi-case + reporting
- **Day 6:** Implement low/med/high pressure configs
- **Day 7:** Batch-run all 3 cases
- **Day 8:** Post-process metrics + create comparison plots
- **Day 9:** Write assumptions/limitations + README
- **Day 10:** Reproducibility pass (fresh run from clean state)

---

## 7) Acceptance Criteria (MVP complete when all true)
- [ ] All 3 load cases solve without failure
- [ ] Metrics exported to CSV/JSON for each case
- [ ] Trend is monotonic/sensible (higher load → higher response in key metrics)
- [ ] Visuals generated and interpretable
- [ ] Full run can be repeated by following README

---

## 8) Risk Controls
- **Rigid-body motion / singular solve:** start with conservative fixed-support proxy
- **Over-complex geometry:** freeze toy geometry for MVP duration
- **Scope creep:** no muscles/sutures until post-MVP
- **Interpretation risk:** report relative trends, not absolute biological claims

---

## 9) Post-MVP Upgrade Path (already planned)
1. Add simplified jaw muscle resultant vectors
2. Split cortical vs cancellous materials
3. Replace toy geometry with template anatomical mesh
4. Add sutures/interfaces + parameter sensitivity
5. Move toward growth/remodeling framework

---

## 10) Immediate Next Actions
1. Initialize `cranio_fea/` project skeleton
2. Draft toy geometry and palate ROI definition
3. Create first CalculiX input generator for single load case
4. Run smoke test and lock baseline output format

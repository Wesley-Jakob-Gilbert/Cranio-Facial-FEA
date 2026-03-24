# Cranio-Facial FEA: Adolescent Bone Remodeling Under Tongue Posture

A computational pipeline that models how sustained tongue-on-palate posture ("mewing") versus mouth breathing influences adolescent maxillary bone density over time, using finite element analysis and Huiskes strain-adaptive remodeling.

The pipeline generates animated density-evolution GIFs comparing the two scenarios side-by-side — the primary deliverable of this project.

> **This is a mechanobiological hypothesis explorer, not a clinical predictor.**
> All outputs are comparative and directional. See [docs/SCIENTIFIC_CAVEATS.md](docs/SCIENTIFIC_CAVEATS.md) for full limitations.

---

## Key Results

| Scenario | 12-Cycle Density Trend | Interpretation |
|----------|----------------------|----------------|
| **Mewing** (2 kPa tongue load) | Bone density saturates at &rho;=1.5 within 1 cycle | Strong mechanical stimulus drives apposition |
| **Mouth breathing** (0 kPa) | Bone density decays from 1.0 &rarr; 0.76 over 12 cycles | Disuse-driven resorption under absent palatal load |

The divergence is consistent with Wolff's law and the mechanostat hypothesis: loaded bone strengthens, unloaded bone weakens.

---

## Method

### Geometry

An analytic U-shaped maxillary arch mesh (2,925 nodes, 2,112 C3D8 hexahedral elements) with three material zones:

- **Bone** (1,832 elements) — bulk maxillary bone, E = 1 GPa
- **Midpalatal suture** (176 elements) — compliant midline zone, E = 50 MPa
- **Lateral sutures** (104 elements) — bilateral compliant zones, E = 50 MPa

Arch dimensions: 48 mm depth, 38 mm width, 13 mm palatal vault, 3 mm bone thickness. The geometry is parameterically generated (not CT-derived) with anatomical features including palatal vault curvature, alveolar ridge elevation, posterior arm flare, and variable bone thickness.

### Loading

- **Surface-normal pressure**: tongue force acts perpendicular to the curved palatal surface at each node (area-weighted normals from adjacent element faces)
- **Anterior-posterior gradient**: anterior palate receives 2.5x the pressure of the posterior palate, modeling real tongue-tip contact patterns
- **Muscle loading** (optional): bilateral jaw-muscle resultant forces for load-case sensitivity studies

### Solver

[CalculiX](http://www.dhondt.de/) linear static solver (C3D8 elements, isotropic elastic materials). Each remodeling cycle runs a full FEA solve, extracts element-level von Mises stress, then updates density.

### Remodeling Law

Huiskes SED-stimulus bone remodeling ([Huiskes et al., 1987](https://doi.org/10.1016/0021-9290(87)90030-3)):

```
SED = sigma_vm^2 / (2 * E)
stimulus = (SED / SED_ref) - 1
rho(t+1) = clamp(rho(t) + alpha * stimulus, [rho_min, rho_max])
E(t+1) = E_bone * rho(t+1)^n
```

Parameters sourced from the trabecular bone remodeling literature. Suture elements hold fixed density (rho = 1.0) throughout.

---

## Quick Start

### Prerequisites

- **Python 3.8+** with `pip install -r requirements.txt`
- **CalculiX** (`ccx`) in `PATH` — `sudo apt install calculix-ccx` on Ubuntu/WSL

### Run the Full Pipeline

```bash
./scripts/reproduce_all.sh
```

This runs mesh generation, all solver cases, field extraction, remodeling loop, animations, and post-processing in sequence.

### Step-by-Step

```bash
python3 mesh/build_mesh.py              # Generate mesh (mesh/mesh_data.json)
python3 solver/make_inp.py              # Generate CalculiX .inp decks
./solver/run_case.sh mewing             # Solve a single case
python3 post/extract_fields.py          # Extract displacement + stress fields
./scripts/remodeling_loop.sh            # 12-cycle remodeling (mewing + mouth_breathing)
python3 post/make_animation.py          # Generate density evolution GIFs
```

### Outputs

- `results/comparison_animation.gif` — side-by-side mewing vs. mouth-breathing density evolution
- `results/<scenario>/density_evolution.gif` — per-scenario animation
- `results/remodeling_summary.csv` — density statistics per cycle
- `results/metrics_summary.csv` — displacement and stress peaks per load case
- `results/mesh_quality_summary.json` — element quality metrics (Jacobian, aspect ratio)

---

## Project Structure

```
configs/          Configuration files (geometry, materials, loads, remodeling params)
mesh/             Mesh generator and mesh_data.json
solver/           CalculiX input deck generator, solver wrapper, .inp files
post/             Post-processing: field extraction, remodeling engine, visualization
scripts/          Pipeline runners, sensitivity sweeps, mesh quality checks
docs/             Specifications, roadmap, scientific caveats
results/          Generated outputs (gitignored; reproduced via pipeline)
```

### Load Cases

| Case | Tongue (kPa) | Muscle (N) | Purpose |
|------|-------------|-----------|---------|
| `mewing` | 2.0 | 0 | Remodeling study: sustained tongue posture |
| `mouth_breathing` | 0.0 | 0 | Remodeling study: absent palatal load |
| `low/medium/high_kpa` | 0.5 / 2.0 / 5.0 | 0 | Single-shot load comparison |
| `*_plus_muscle` | 0.5–5.0 | 2–8 | Tongue + jaw muscle combined loading |

---

## Mesh Quality

The v2 mesh passes all quality checks (run `python3 scripts/mesh_quality_report.py`):

| Metric | Value |
|--------|-------|
| Elements | 2,112 (C3D8 hex) |
| Negative Jacobians | 0 |
| Aspect ratio (P95) | 3.74 |
| Scaled Jacobian (mean) | 0.94 |

---

## Limitations

This model is intentionally simplified to explore the **directional hypothesis** that tongue posture affects maxillary bone remodeling. Key limitations:

- Geometry is analytic (not CT-derived) — stress magnitudes are not anatomically calibrated
- Material is isotropic linear elastic — no cortical/cancellous distinction
- Remodeling parameters are from hip/spine literature, not validated for craniofacial bone
- Suture biology is modeled as mechanical compliance only, not cellular response
- Mouth-breathing is modeled as zero palatal load (simplified from the full craniofacial loading environment)

For a thorough discussion, see [docs/SCIENTIFIC_CAVEATS.md](docs/SCIENTIFIC_CAVEATS.md).

---

## Future Directions

- CT-derived mesh from open-source maxillary scan data
- FEBio or nonlinear solver integration for large-deformation suture mechanics
- Transversely isotropic cortical bone material model
- Suture widening mechanics under tensile strain
- Parameter sensitivity dashboard across remodeling constants

---

## References

1. Huiskes, R., et al. (1987). Adaptive bone-remodeling theory applied to prosthetic-design analysis. *J. Biomechanics*, 20(11-12), 1135-1150.
2. Mew, J.R.C. (2004). The postural basis of malocclusion: A philosophical overview. *Am. J. Orthod. Dentofac. Orthop.*, 126(6), 729-738.
3. Proffit, W.R. (1978). Equilibrium theory revisited. *Angle Orthod.*, 48(3), 175-186.
4. Frost, H.M. (2003). Bone's mechanostat: A 2003 update. *Anat. Rec.*, 275A(2), 1081-1101.

---

## License

This project is provided for educational and research purposes. Not for clinical use.

# Cranio-Facial FEA: Adolescent Bone Remodeling Under Tongue Posture

A computational pipeline that models how sustained tongue-on-palate posture ("mewing") versus mouth breathing influences adolescent maxillary bone density over time, using finite element analysis and Huiskes strain-adaptive remodeling.

> **This is a mechanobiological hypothesis explorer, not a clinical predictor.**
> All outputs are comparative and directional. See [docs/SCIENTIFIC_CAVEATS.md](docs/SCIENTIFIC_CAVEATS.md) for full limitations.

---

## Density Evolution: Mewing vs. Mouth Breathing

![Mewing vs Mouth Breathing — 12-cycle density evolution](results/comparison_animation.gif)

*Left: mewing (2 kPa sustained tongue load on palatal vault). Right: mouth breathing (gravity only, no palatal load). Color = &Delta;&rho; from baseline; 12 remodeling cycles on CT-derived maxillary mesh.*

---

## Key Results

| Scenario | 12-Cycle Surface Trend | Interpretation |
|----------|----------------------|----------------|
| **Mewing** (2 kPa tongue load) | Localized bone formation on palatal vault (max &Delta;&rho; &asymp; +0.19), less overall surface resorption | Tongue pressure drives anabolic stimulus at the palate |
| **Mouth breathing** (gravity only) | Uniform surface resorption, no localized formation | Absent palatal load produces disuse-driven bone loss |

The divergence is consistent with Wolff's law and the mechanostat hypothesis: loaded bone strengthens, unloaded bone weakens. The spatial pattern — formation concentrated where the tongue contacts the palate — matches the expected mechanical stimulus distribution.

---

## Method

### Geometry

CT-derived maxillary mesh from the [Open-Full-Jaw](https://github.com/diku-dk/Open-Full-Jaw) dataset (Patient 3): **184,617 nodes, 853,068 C3D4 tetrahedral elements**. The volumetric mesh is spatially labeled into material zones via boundary-face and centroid heuristics:

- **Cortical bone** (139,362 elements) — surface layer (elements with boundary faces), E = 15 GPa
- **Cancellous bone** (694,073 elements) — interior bone, E = 0.5 GPa
- **Midpalatal suture** (17,033 elements) — compliant sagittal midplane zone, E = 50 MPa
- **Lateral sutures** (2,600 elements) — anterior-lateral compliant zones, E = 50 MPa

A parametric template mesh (2,112 C3D8 hex elements) is also available for fast iteration via the `profile` setting in `configs/geometry.json`.

### Loading

- **Surface-normal pressure**: tongue force acts perpendicular to the curved palatal surface at each node (area-weighted normals from adjacent element faces)
- **Anterior-posterior gradient**: anterior palate receives 2.5x the pressure of the posterior palate, modeling real tongue-tip contact patterns
- **Gravity body force**: self-weight loading on all elements via mass-consistent density cards
- **Muscle loading** (optional): bilateral jaw-muscle resultant forces for load-case sensitivity studies

### Solver

[CalculiX](http://www.dhondt.de/) linear static solver (C3D4 tetrahedral elements, isotropic elastic materials). Each remodeling cycle runs a full FEA solve (~62 s on the 853k-element mesh), extracts element-level strain energy density from CalculiX ENER output, then updates density.

### Remodeling Law

Huiskes lazy-zone SED-stimulus bone remodeling ([Huiskes et al., 1987](https://doi.org/10.1016/0021-9290(87)90030-3)):

```
SED = true strain energy density (from CalculiX *EL PRINT ENER)
stimulus = (SED / SED_ref) - 1
if stimulus >  s:  drho = alpha_app * (stimulus - s)    [apposition]
if stimulus < -s:  drho = alpha_res * (stimulus + s)    [resorption]
else:              drho = 0                              [lazy zone]
rho(t+1) = clamp(rho(t) + drho, [rho_min, rho_max])
E(t+1) = E_base * rho(t+1)^n
```

Per-layer homeostatic SED setpoints (cortical vs. cancellous) and per-layer Carter-Hayes power law exponents (n=3 cortical, n=2 cancellous). Suture elements undergo strain-dependent modulus softening. Tooth root elements are held at fixed density.

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
# CT mesh (default — requires meshio: pip install meshio)
python3 mesh/build_mesh_ct.py           # Convert CT .msh → mesh/mesh_data.json

# Or template mesh (fast, no external data needed)
python3 mesh/build_mesh.py              # Generate parametric mesh_data.json

# Pipeline
python3 solver/make_inp.py              # Generate CalculiX .inp decks
./solver/run_case.sh mewing             # Solve a single case
python3 post/extract_fields.py          # Extract displacement + stress + SED fields
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
results/          Generated outputs (gitignored except comparison GIF; reproduced via pipeline)
```

### Load Cases

| Case | Tongue (kPa) | Muscle (N) | Purpose |
|------|-------------|-----------|---------|
| `mewing` | 2.0 | 0 | Remodeling study: sustained tongue posture |
| `mouth_breathing` | 0.0 | 0 | Remodeling study: absent palatal load (gravity only) |
| `low/medium/high_kpa` | 0.5 / 2.0 / 5.0 | 0 | Single-shot load comparison |
| `*_plus_muscle` | 0.5–5.0 | 2–8 | Tongue + jaw muscle combined loading |

---

## Mesh Quality

The CT mesh (run `python3 scripts/mesh_quality_report.py`):

| Metric | Value |
|--------|-------|
| Elements | 853,068 (C3D4 tet) |
| Nodes | 184,617 |
| Source | Open-Full-Jaw Patient 3 |
| Degenerate tets removed | ~0 (volume filter at build time) |

---

## Limitations

This model explores the **directional hypothesis** that tongue posture affects maxillary bone remodeling. Key limitations:

- Material is isotropic linear elastic (cortical/cancellous split by boundary-face heuristic, not microstructural anisotropy)
- Remodeling parameters are from hip/spine literature, not validated for craniofacial bone
- Suture biology is modeled as mechanical compliance only, not cellular response
- Mouth-breathing scenario includes only gravity — no baseline masticatory loading
- Element set labeling (sutures, cortical/cancellous) uses spatial heuristics, not manual segmentation

For a thorough discussion, see [docs/SCIENTIFIC_CAVEATS.md](docs/SCIENTIFIC_CAVEATS.md).

---

## Future Directions

- FEBio or nonlinear solver integration for large-deformation suture mechanics (GPU-accelerated via CUDA)
- Transversely isotropic cortical bone material model
- Baseline masticatory loading for the mouth-breathing comparison
- Multi-patient mesh comparison across Open-Full-Jaw dataset
- Parameter sensitivity dashboard across remodeling constants

---

## References

1. Huiskes, R., et al. (1987). Adaptive bone-remodeling theory applied to prosthetic-design analysis. *J. Biomechanics*, 20(11-12), 1135-1150.
2. Carter, D.R. & Hayes, W.C. (1977). The compressive behavior of bone as a two-phase porous structure. *J. Bone Joint Surg.*, 59(7), 954-962.
3. Frost, H.M. (2003). Bone's mechanostat: A 2003 update. *Anat. Rec.*, 275A(2), 1081-1101.
4. Mew, J.R.C. (2004). The postural basis of malocclusion: A philosophical overview. *Am. J. Orthod. Dentofac. Orthop.*, 126(6), 729-738.
5. Proffit, W.R. (1978). Equilibrium theory revisited. *Angle Orthod.*, 48(3), 175-186.
6. Gholamalizadeh, T., et al. (2022). Open-Full-Jaw: An open-access dataset and pipeline for finite element models of human jaw. *Computer Methods and Programs in Biomedicine*, 224, 107009.



---

## License

This project is provided for educational and research purposes. Not for clinical use.

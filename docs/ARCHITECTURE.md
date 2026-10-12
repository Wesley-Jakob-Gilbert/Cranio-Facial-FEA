# Architecture

`craniofea` is a Python package (`src/craniofea`) organised around one data
model (`craniofea/model.py`) that every stage reads and writes.

```
mesh (CT import / template / remesh) ──► Mesh (.npz)
                                           │
materials (tissue + density -> C) ─────────┤
loading (habit -> LoadCases) ──────────────┤
                                           ▼
                                        Problem ──► Solver ──► Solution per load case
                                                    (jax | reference | calculix | febio)
                                           ▲                      │
remodel (daily stimulus, setpoint, law) ◄──┴──────────────────────┘  density update
```

## Data model (contract)

See the module docstring of `craniofea/model.py`. In short:

| Object | Content |
|---|---|
| `Mesh` | `nodes (N,3)` m, `tets (M,4)` 0-based positively oriented; `node_sets`, `elem_sets`, `surface_sets` (outward boundary triangles), `elem_fields` (per-element arrays such as surface normals / depth), `frame` (anatomical axes), `meta`. Saved as compressed `.npz`. |
| `Material` | `C (M,6,6)` global Voigt stiffness, `mass_density (M,)`, plus descriptive `E, nu, ortho, axes` for backends that need them. |
| `LoadCase` | named; `pressures` on surface sets (scalar or per-face), `gravity`, `nodal_forces`, `cycles_per_day`. |
| `Problem` | mesh + material + Dirichlet (`fixed_nodes`, optional `fixed_dofs (N,3)`) + load cases. |
| `Solution` | `u (N,3)`, element `strain`, `stress` (Voigt, engineering shear), `sed`, `info`. |

Voigt order is `[xx, yy, zz, yz, xz, xy]`; positive pressure pushes into the solid.

## Package layout

| Module | Responsibility |
|---|---|
| `mesh/primitives.py` | box meshes, boundary-face extraction, face normals/areas |
| `mesh/ct.py`, `mesh/frame.py`, `mesh/labeling.py` | Open-Full-Jaw import, anatomical frame detection, anatomical sets |
| `mesh/template.py`, `mesh/remesh.py`, `mesh/quality.py` | parametric arch, coarse CT remesh, element quality |
| `data.py` | dataset download and checksum |
| `fem/constitutive.py` | isotropic/orthotropic Voigt stiffness and rotations |
| `fem/reference.py` | small NumPy/SciPy direct solver used as a test oracle |
| `fem/jax_solver.py` | production solver: JAX kernels, AMG-preconditioned CG, implicit-adjoint gradients |
| `backends/calculix.py`, `backends/febio.py` | deck writers, runners and result readers for cross-verification |
| `materials.py` | tissue layers + density field -> `Material` |
| `remodel/` | load history, daily strain stimulus, homeostatic setpoint, density evolution |
| `viz/` | figures and animations |
| `cli.py` | `craniofea` command-line entry point |

## Development environment

The reference environment is WSL Ubuntu 22.04 with a `uv` venv at
`~/cfea-venv` (Python 3.12, JAX with CUDA 12, CalculiX 2.17 at `/usr/bin/ccx`).
`tests/conftest.py` puts this checkout's `src/` first on `sys.path`.

```bash
python -m pytest -m "not slow"     # fast suite (CI)
python -m pytest                   # full verification suite
```

## Canonical anatomical labels

Every anatomical mesh (CT import, coarse remesh, parametric template) provides
the same names so that loading, materials and reporting are mesh-agnostic.

| Kind | Name | Meaning |
|---|---|---|
| frame | `Mesh.frame` | `lr` patient left->right, `ap` posterior->anterior, `si` inferior->superior |
| surface set | `palate` | oral surface of the palatal vault (tongue contact) |
| surface set | `occlusal` | alveolar crest / tooth-socket surface that carries bite reactions |
| surface set | `fixed` | cut surfaces where the maxilla joins the cranial base / zygoma |
| surface set | `all` | complete outward boundary |
| node set | `fixed` | nodes of the `fixed` surface (all dofs clamped) |
| elem set | `cortical`, `cancellous`, `suture_mid` | material layers (disjoint, cover all elements) |
| elem set | `region_palatal_vault`, `region_alveolar_anterior`, `region_alveolar_posterior`, `region_basal` | reporting regions (disjoint, cover all elements) |
| elem field | `depth` | (M,) distance from element centroid to the outer surface, m |
| elem field | `surface_normal` | (M,3) unit outward normal of the nearest boundary face |

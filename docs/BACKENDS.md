# External solver backends

`craniofea.backends` translates a `Problem` (see `craniofea/model.py`) into the
input language of two established finite-element codes, runs them, and reads the
results back into `Solution` objects in craniofea conventions:

| Backend | Class / writer | Element | Analysis |
|---|---|---|---|
| CalculiX 2.17 (`ccx`) | `CalculixSolver`, `write_inp` | C3D4 | linear static |
| FEBio 4 (`febio4`) | `FebioSolver`, `write_feb` | tet4 | static, finite deformation (see below) |

Both implement the `Solver` protocol, so they are drop-in replacements for the
in-house solvers:

```python
from craniofea.backends import CalculixSolver, FebioSolver, write_feb

results = CalculixSolver(n_threads=4).solve(problem)       # {case name: Solution}
results = FebioSolver().solve(problem)
write_feb(problem, "maxilla_bite.feb", case="bite")         # open in FEBio Studio
```

Pass `work_dir=` to keep decks and raw solver output; otherwise a temporary
directory is used and removed. Every `Solution.info` records the executable, the
work directory, timings, the number of materials written, and a nodal-equilibrium
residual (see [Result verification](#result-verification)).

## Data-model mapping

| Contract | CalculiX deck | FEBio 4.0 file |
|---|---|---|
| `nodes`, `tets` (0-based) | `*NODE`, `*ELEMENT, TYPE=C3D4`, id = index + 1 | `<Nodes>`, `<Elements type="tet4">`, id = index + 1 |
| `Material.C` | grouped into `*MATERIAL`s; isotropic → `*ELASTIC` (E, ν), otherwise `*ELASTIC, TYPE=ANISO` (21 constants of the global `C`) | `isotropic elastic` / `orthotropic elastic`; parameters varying per element are `ElementData` maps |
| `Material.axes` | not needed (ANISO is written in global axes) | per-element `mat_axis` (`<a>` = axis 1, `<d>` = axis 2) |
| `mass_density` | `*DENSITY` per material | `density` (mapped) |
| `fixed_nodes`, `fixed_dofs` | `*BOUNDARY node, dof, dof` per constrained dof | `zero displacement` BCs on node sets grouped by fixed-dof pattern |
| `pressures` (scalar or per face) | `*DLOAD elem, Pk, p` on the owning element face | `pressure` surface load on a `<Surface>` of `tri3`, per-face values via `SurfaceData` |
| `gravity` | `*DLOAD EALL, GRAV, abs(g), direction` | `body force` with `force = -g` |
| `nodal_forces` | `*CLOAD node, dof, F` | `nodal_force` with a `NodeData` vec3 map |
| load cases | one job per case: `caseNN_<name>.inp` includes a shared `model.inp` | one `.feb` per case |
| outputs | `*NODE PRINT` U, `*EL PRINT` S, E, ENER → `.dat` | logfile `node_data` ux;uy;uz, `element_data` sx..sxy, Ex..Exy, sed |

Each load case is solved independently, so nothing carries over between cases.
That matters for FEBio in particular: its steps are cumulative in the deformed
configuration.

## Conventions and pitfalls

### Voigt order and shear strain

The contract uses `[xx, yy, zz, yz, xz, xy]` with **engineering** shear strain
(γ = 2ε). Both codes report components as `(xx, yy, zz, xy, xz, yz)` (FEBio by
name), and both report **tensor** shear strains. The readers reorder the columns
and double the shear strains. The unit test
`test_parse_dat_fixture_and_convert_to_contract` pins this down on a stored
`.dat` excerpt.

### CalculiX `TYPE=ANISO` ordering

The 21 constants are read eight per line in the order

```
D1111 D1122 D2222 D1133 D2233 D3333 D1112 D2212
D3312 D1212 D1113 D2213 D3313 D1213 D1313 D1123
D2223 D3323 D1223 D1323 D2323
```

CalculiX's `D_ijkl` multiplies the tensor strain, summed over both `kl` and
`lk`: σ₁₁ = … + D₁₁₁₂ ε₁₂ + D₁₁₂₁ ε₂₁ = … + D₁₁₁₂ γ₁₂. The tensor constants
therefore equal the engineering-shear Voigt entries one-to-one (`D1212 = C66 = G`
for isotropy), with no factors of 2. The mapping is `ANISO_INDEX` in
`calculix.py` and is verified three ways:

- against the manual's order (`test_aniso_order_matches_calculix_manual`);
- by a round trip;
- by the cross-verification problem, whose orthotropic layer is rotated by 0.7 rad
  about (1, 2, 3), so all 21 constants are non-zero and any misplaced entry
  would show up as a large error.

CalculiX truncates input lines at 132 characters. Eight constants per line
therefore use 9 significant digits (relative rounding ≤ 5·10⁻¹⁰); all other
values are written with 17 digits (lossless).

### C3D4 face numbering and pressure sign

CalculiX faces are P1 = nodes 1-2-3, P2 = 1-4-2, P3 = 2-4-3, P4 = 3-4-1
(`TET_FACE_NODES`). `face_owners` matches every surface triangle to its element
face by sorted node triplets (vectorised: int64 codes, sort and `searchsorted`).
It returns a sign: +1 if the triangle's right-hand normal points out of that
element.

The contract's pressure acts along −n_triangle; a CalculiX face pressure pushes
into the element. The deck value is therefore `sign · p`. For a face shared by
two elements (an internal surface), the element on the outward side is chosen.
Contributions of several sets on the same face are summed, because CalculiX
would otherwise keep only one of them.

FEBio computes the facet normal from the node order (N = x,r × x,s) and applies
traction t = −p N. The writer keeps the contract's (outward) node order, so the
values are passed through unchanged.

### Materials in CalculiX: grouping and binning

CalculiX needs one `*MATERIAL` + `*SOLID SECTION` per distinct stiffness.
Elements are grouped by bitwise-identical `(C, ρ)`; ρ is part of the key only
when some load case has gravity. Measured cost of distinct materials in ccx 2.17
(isotropic, pressure load, solve included):

| elements | materials | ccx time per case |
|---|---|---|
| 20 736 | 178 (binned) | 1.2 s |
| 20 736 | 20 736 (one per element) | 4.0 s |
| 49 152 | 190 (binned) | 3.0 s |
| 49 152 | 49 152 (one per element) | 17.5 s |

The overhead grows roughly quadratically with the number of materials (+2.7 s at
21k, +14.5 s at 49k), which extrapolates to more than an hour for an 850k-element
CT mesh. One material per element is therefore only offered on request
(`max_materials=None`).

The default `max_materials=20_000` bins continuous fields:

- the key is the normalised stiffness `C/‖C‖_F`, rounded to 9 digits, so
  different anisotropy shapes or orientations are never merged;
- within each shape, elements are binned uniformly in `log ‖C‖_F` (for
  density-driven laws C = f(ρ)·C₀ this is binning in log-modulus);
- each bin is represented by the mean C and mean ρ of its members.

The approximation is reported in `Solution.info["material_binning"]`
(`max_rel_C_error`, `mean_rel_C_error`, `max_rel_density_error`). On an
843 648-element box with a 10× continuous modulus range, the default produced
1 470 materials with a maximum stiffness error of 6.3·10⁻⁵. If the number of
distinct orientations alone exceeds the limit (for example, axes following a
per-element surface normal), binning cannot help and a `ValueError` explains
the options.

FEBio needs no binning: every varying parameter is an exact per-element map.

### FEBio: no linear solid solver

FEBio solves every static problem with a finite-deformation Newton method; there
is no geometrically linear option. `isotropic elastic` and `orthotropic elastic`
are St. Venant–Kirchhoff materials (S = C : E_Green). They reduce to linear
elasticity to first order, so results differ from a linear solver by
O(max strain), relative. This is measured, not assumed: the table below shows the
error scaling exactly with the load. The regression test
`test_febio_deviation_is_first_order_in_load` checks this; a mapping error (sign,
axis, unit, face) would not scale this way.

Other FEBio details handled by the writer:

- `<linear>1</linear>` on pressure loads (no follower load).
- One load step with full Newton (`max_ups 0`). This converges in 3 iterations.
  Tolerances tighter than the defaults (`dtol 1e-7`, `etol 1e-10`) sit below the
  round-off floor and make FEBio report non-convergence.
- **Body force sign**: FEBio adds −ρ·f to the external load vector
  (`FEElasticSolidDomain::BodyForce`), so the writer sets `force = -gravity`.
- **Poisson ratios**: FEBio uses ν₁₂, ν₂₃, ν₃₁ with S₁₃ = −ν₃₁/E₃, so
  ν₃₁ = ν₁₃·E₃/E₁, and G₃₁ = G₁₃. All nine constants are recomputed from `C`
  rotated into `Material.axes`, so they are exact for any rotated orthotropic
  matrix.
- **No general anisotropic linear material** exists in FEBio 4. Elements that are
  neither isotropic nor orthotropic in `Material.axes` are rejected with a clear
  error.
- **Strain output**: FEBio 4.13's infinitesimal-strain log variables `eyz` and
  `exz` are swapped. `FEElasticMaterialPoint::SmallStrain` passes the xz and yz
  components to the `mat3ds(xx, yy, zz, xy, yz, xz)` constructor in the wrong
  order. The backend reads the Green–Lagrange strain `Ex..Exz` instead, which is
  computed correctly and is the strain the SVK material actually uses. It differs
  from sym(∇u) by ½∇uᵀ∇u, again O(strain).
- **Node-set lists** must stay one comma-separated sequence across line breaks
  (`1,2,…,10,⏎11,…`). Without the trailing comma, FEBio silently merges ids
  across lines.
- **Precision**: logfile data are printed with 12 significant digits only when
  no `format` attribute is given (a format string drops them to 6).
- Without MKL, FEBio selects its skyline direct solver itself; MKL builds use
  pardiso. Override with `FebioSolver(linear_solver=...)`.

### CalculiX output precision

`.dat` values carry 7 significant digits (`E13.6`), which sets the agreement
floor at about 5·10⁻⁷. Exponents below −99 are printed without the `E`
(`1.000000-120`); the reader handles that form.

### Result verification

Each result is checked independently of the solver. Nodal internal forces are
rebuilt from the reported element stresses, f_int,a = V σ ∇N_a, and compared
with the contract's load vector at the free dofs. `info["equilibrium_residual"]`
is max|f_int − f_ext| / max(|f_ext|, Σ|f_int,e|):

- correct CalculiX runs give ~10⁻⁷ (output rounding);
- FEBio runs give O(max strain);
- a 1 % error in a single element's stress gives > 10⁻⁴.

During development, two runs of the same deck on a heavily loaded host (four
other jobs sharing the machine) returned stiffness-corrupted solutions with
4 threads. The residual was localised at two interior nodes and the u error was
0.7 % and 48 %. Single- and multi-threaded reruns on an idle host were
bit-identical and correct. `CalculixSolver` therefore reruns a case
single-threaded if the residual exceeds `equilibrium_tol` (default 10⁻⁴), and
raises `BackendError` if it still fails. `FebioSolver` accepts
`equilibrium_tol + 10·max|strain|`.

## Measured agreement with `ReferenceSolver`

Benchmark: `craniofea.backends.verification.benchmark_problem()`, a
20 × 10 × 8 mm box with 140 nodes and 432 tets.

- **Lower half**: isotropic, E = 1–11 GPa varying along x, ν = 0.3.
- **Upper half**: orthotropic (17/11/9 GPa) with axes rotated 0.7 rad about
  (1, 2, 3).
- **Density**: 1800–2000 kg/m³.
- **Supports**: xmin clamped, ymin on rollers.
- **Load cases**: per-face pressure, gravity, and point forces.

Values are max |a − b| / max |a| over all nodes/elements.

CalculiX (`load_scale=1`, max strain 5·10⁻⁴–10⁻³):

| case | u | stress | strain | sed |
|---|---|---|---|---|
| pressure | 4.6e-7 | 1.6e-7 | 1.9e-7 | 9.3e-8 |
| gravity | 2.5e-7 | 8.8e-8 | 1.0e-7 | 2.3e-7 |
| nodal | 2.7e-7 | 1.4e-7 | 1.2e-7 | 2.2e-7 |

FEBio 4.13 (max u difference; the other fields behave identically):

| load scale | max strain (pressure case) | pressure | gravity | nodal |
|---|---|---|---|---|
| 1e-3 | 5.1e-7 | 4.0e-7 | 5.6e-7 | 1.6e-7 |
| 1e-2 | 5.1e-6 | 4.0e-6 | 5.6e-6 | 1.6e-6 |
| 1e-1 | 5.1e-5 | 4.0e-5 | 5.6e-5 | 1.6e-5 |
| 1 | 5.1e-4 | 4.0e-4 | 5.6e-4 | 1.6e-4 |

At physiological maxillary strains (10⁻⁴–10⁻³), expect FEBio to agree with the
linear solvers to 10⁻⁴–10⁻³ relative. That is far below modelling uncertainty,
but it is not round-off.

Tests (`pytest -m calculix`, `pytest -m febio`) assert < 2·10⁻⁶ for CalculiX and
< 2·10⁻⁵ for FEBio at `load_scale=1e-3`. They skip automatically when the
executable is missing; all other backend tests run in CI without solvers.

## Performance

Writers are vectorised: rows are formatted in 100k-row chunks with a single `%`
operation each, and face matching and material grouping are pure NumPy. On an
843 648-element / 148 877-node box with a continuous modulus field:

- CalculiX deck (binned to 1 470 materials, pressure on the full boundary,
  gravity): 4.0 s, 184 MB;
- FEBio file: 2.4 s, 131 MB;
- peak memory of the writing process: 1.4 GB.

## Building FEBio

`scripts/install_febio.sh` builds FEBio 4 from source without root:

```bash
bash scripts/install_febio.sh            # FEBIO_TAG=v4.13 FEBIO_PREFIX=~/opt/febio JOBS=4
export FEBIO=~/opt/febio/bin/febio4      # optional; ~/opt/febio/bin/febio4 is found by default
```

It shallow-clones the requested tag, configures CMake with every optional
dependency disabled (MKL/pardiso, HYPRE, MMG, levmar, NLopt, FFTW, zlib), builds
the `febio4` target, and installs the executable and its shared libraries to
`$FEBIO_PREFIX/{bin,lib}` (the executable's rpath is `../lib`).

Without MKL, FEBio's default build links the OpenMP runtime into only some of its
libraries, and linking `febio4` fails with `undefined reference to GOMP_*`. The
script passes `-fopenmp` to the shared and executable linker flags to fix this.

Status in the reference environment (WSL Ubuntu 22.04, gcc 11.4, CMake 3.22):
FEBio v4.13 built successfully. Compilation took about 18 minutes with 4 jobs,
plus about 1 minute to relink after the OpenMP fix. The result is
`~/opt/febio/bin/febio4`, using the skyline linear solver. For large models,
install the official FEBio release (MKL/pardiso) and point `FEBIO` at it; the
generated files are identical.

## Relation to the Open-Full-Jaw FEBio template

The Open-Full-Jaw dataset ships a minimal `template.feb` (spec 2.5) containing
only `Module`, `Control`, `Globals` and `Output`. `write_feb` produces a complete
spec 4.0 model; the Geometry section of 2.5 becomes `Mesh`, `MeshDomains` and
`MeshData` in 4.0. Compared with the template:

- **Same**: `Module type="solid"`, static analysis, and the time-stepper
  settings (`max_retries 5`, `opt_iter 10`, `dtmin 0.01`).
- **One step instead of ten**: a single step of size 1 replaces 10 × 0.1, because
  a small-strain elastic problem needs no load increments.
- **Newton settings**: full Newton (`max_ups 0`) instead of BFGS with 10 updates,
  and `dtol 1e-7` / `etol 1e-10` instead of the FEBio defaults 0.001 / 0.01. The
  defaults stop at ~10⁻³ accuracy, which is too loose for verification.
- **Omitted**: `Globals/Constants` (T, R, Fc are only used by multiphysics
  modules).
- **Output**: the plotfile keeps displacement and stress (relative volume
  dropped); a logfile is added so results can be read back without parsing
  `.xplt`.

## Limitations

- Linear tetrahedra only (C3D4 / tet4); homogeneous Dirichlet conditions only,
  matching the data model.
- CalculiX: continuous fields beyond `max_materials` are binned (reported).
  One material per element is exact but scales poorly in ccx 2.17.
- FEBio: geometrically nonlinear SVK (O(strain) deviation); isotropic and
  orthotropic materials only; stress is Cauchy and strain Green–Lagrange.
- Results are read from text output: 7 significant digits (CalculiX) and 12
  (FEBio).

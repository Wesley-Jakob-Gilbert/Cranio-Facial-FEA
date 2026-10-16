"""CalculiX (ccx) backend: C3D4 deck writer, runner and ``.dat`` result reader.

Mapping of the data model (see ``docs/BACKENDS.md`` for details):

- nodes / elements: ``*NODE`` and ``*ELEMENT, TYPE=C3D4`` with 1-based ids
  (id = index + 1);
- materials: elements with identical ``(C, mass_density)`` share one
  ``*MATERIAL``. Isotropic matrices are written as ``*ELASTIC`` (E, nu); every
  other matrix as ``*ELASTIC, TYPE=ANISO`` with the 21 constants of the global
  ``C`` (no ``*ORIENTATION`` needed). Continuous fields with more distinct
  matrices than ``max_materials`` are binned (documented approximation, reported
  in ``Solution.info``);
- Dirichlet: one ``*BOUNDARY`` line per constrained dof;
- pressure: ``*DLOAD`` on element faces ``P1..P4`` (per-face values);
- gravity: ``*DLOAD ... GRAV`` with ``*DENSITY`` per material;
- nodal forces: ``*CLOAD``.

Each load case is solved as its own job; the model definition is written once
to ``model.inp`` and included by every ``<case>.inp`` deck, so nothing carries
over between load cases.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from craniofea.backends._common import (
    BackendError,
    collect_face_pressures,
    equilibrium_residual,
    find_executable,
    isotropic_parameters,
    tail,
    write_rows,
)
from craniofea.model import LoadCase, Problem, Solution

# CalculiX TYPE=ANISO order of the 21 constants D_ijkl, as (row, col) of the
# craniofea Voigt matrix [xx, yy, zz, yz, xz, xy]:
#   D1111 D1122 D2222 D1133 D2233 D3333 D1112 D2212
#   D3312 D1212 D1113 D2213 D3313 D1213 D1313 D1123
#   D2223 D3323 D1223 D1323 D2323
# CalculiX's D_ijkl multiplies TENSOR strain, sigma_ij = D_ijkl eps_kl, summed
# over all k, l. Because eps_12 and eps_21 both appear, D1112 * 2 eps_12 =
# D1112 * gamma_12, so the tensor constants equal the engineering-shear Voigt
# entries one-to-one (D1212 = C_66 = G for isotropy); no factors are needed.
ANISO_INDEX = np.array([
    (0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 5), (1, 5),
    (2, 5), (5, 5), (0, 4), (1, 4), (2, 4), (5, 4), (4, 4), (0, 3),
    (1, 3), (2, 3), (5, 3), (4, 3), (3, 3),
])

# ccx output component order (xx, yy, zz, xy, xz, yz) -> craniofea Voigt order.
_CCX_TO_VOIGT = np.array([0, 1, 2, 5, 4, 3])

# CalculiX truncates input lines at 132 characters. Nine significant digits keep
# eight ANISO constants per line below that limit (relative rounding <= 5e-10).
_ANISO_FMT = "%.9g"


def aniso_constants(C: np.ndarray) -> np.ndarray:
    """(..., 21) CalculiX TYPE=ANISO constants from (..., 6, 6) Voigt stiffness."""
    C = np.asarray(C, dtype=float)
    return C[..., ANISO_INDEX[:, 0], ANISO_INDEX[:, 1]]


def aniso_to_voigt(D: np.ndarray) -> np.ndarray:
    """Inverse of :func:`aniso_constants` (symmetric (..., 6, 6) matrix)."""
    D = np.asarray(D, dtype=float)
    C = np.zeros(D.shape[:-1] + (6, 6))
    C[..., ANISO_INDEX[:, 0], ANISO_INDEX[:, 1]] = D
    C[..., ANISO_INDEX[:, 1], ANISO_INDEX[:, 0]] = D
    return C


# --------------------------------------------------------------------------- #
# Material grouping
# --------------------------------------------------------------------------- #
@dataclass
class MaterialTable:
    """Distinct materials written to the deck.

    elem_material : (M,) material index of every element.
    C : (K, 6, 6) stiffness of each material.
    density : (K,) mass density of each material.
    exact : False if the table was produced by binning.
    stats : approximation statistics (empty when exact).
    """

    elem_material: np.ndarray
    C: np.ndarray
    density: np.ndarray
    exact: bool = True
    stats: dict = field(default_factory=dict)

    @property
    def n_materials(self) -> int:
        return int(self.C.shape[0])


def _upper(C: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(6)
    return C[:, iu[0], iu[1]]


def unique_rows(key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group bitwise-identical rows of a float matrix.

    Returns (first (K,) index of one representative row per group, inverse (n,)
    group of every row), groups ordered by first occurrence. Rows are sorted by
    two random projections and adjacent rows compared exactly, which is several
    times faster than ``np.unique(axis=0)``. Identical rows always share a group;
    a projection collision between different rows can at worst split a group,
    never merge distinct rows.
    """
    key = np.ascontiguousarray(key, dtype=float)
    n = key.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    r = np.random.default_rng(12345).standard_normal((key.shape[1], 2))
    h = key @ r
    order = np.lexsort((h[:, 1], h[:, 0]))
    ks = key[order]
    new = np.ones(n, dtype=bool)
    new[1:] = np.any(ks[1:] != ks[:-1], axis=1)
    gid_sorted = np.cumsum(new) - 1
    gid = np.empty(n, dtype=np.int64)
    gid[order] = gid_sorted
    first = np.full(gid_sorted[-1] + 1, n, dtype=np.int64)
    np.minimum.at(first, gid, np.arange(n))
    # Renumber groups by first occurrence for a deterministic, readable deck.
    rank = np.empty_like(first)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return np.sort(first), rank[gid]


def group_materials(C: np.ndarray, density: np.ndarray, *, use_density: bool = True,
                    max_materials: int | None = None, shape_decimals: int = 9
                    ) -> MaterialTable:
    """Group elements into CalculiX materials.

    Exact grouping: elements whose stiffness (and, if `use_density`, mass
    density) are bitwise identical share a material.

    If the number of distinct materials exceeds `max_materials`, elements are
    binned instead. Binning key: the normalised stiffness ``C / ||C||_F``
    rounded to `shape_decimals` digits (so differently oriented or differently
    shaped anisotropy is never merged), combined with a logarithmic bin of the
    scalar stiffness magnitude ``||C||_F``. For density-driven modulus laws
    (C = f(rho) C0) this is binning in log-modulus. Each bin's material is the
    arithmetic mean of its members' C and density. The induced error is returned
    in ``stats`` (max relative Frobenius error of C, max relative density error).
    """
    C = np.asarray(C, dtype=float)
    M = C.shape[0]
    rho = np.asarray(density, dtype=float).reshape(M)
    key = _upper(C)
    if use_density:
        key = np.column_stack([key, rho])
    first, inv = unique_rows(key)
    if max_materials is None or len(first) <= max_materials:
        return MaterialTable(inv, C[first].copy(), rho[first], exact=True)

    norm = np.linalg.norm(C.reshape(M, -1), axis=1)
    shape = np.round(_upper(C) / norm[:, None], shape_decimals)
    shape_first, shape_id = unique_rows(shape)
    n_shapes = len(shape_first)
    if n_shapes > max_materials:
        raise ValueError(
            f"{n_shapes} distinct anisotropy shapes/orientations exceed max_materials="
            f"{max_materials}; binning by modulus cannot represent them. Use "
            "max_materials=None to write one material per element.")
    n_bins = max(1, max_materials // n_shapes)
    logn = np.log(norm)
    bin_id = np.zeros(M, dtype=np.int64)
    for s in range(n_shapes):
        sel = shape_id == s
        lo, hi = logn[sel].min(), logn[sel].max()
        if hi > lo:
            b = np.floor((logn[sel] - lo) / (hi - lo) * n_bins).astype(np.int64)
            bin_id[sel] = np.minimum(b, n_bins - 1)
    group = shape_id * n_bins + bin_id
    _, inv = np.unique(group, return_inverse=True)
    inv = inv.ravel()
    K = inv.max() + 1
    count = np.bincount(inv, minlength=K).astype(float)
    Ck = np.zeros((K, 36))
    np.add.at(Ck, inv, C.reshape(M, 36))
    Ck = (Ck / count[:, None]).reshape(K, 6, 6)
    rk = np.bincount(inv, weights=rho, minlength=K) / count
    errC = np.linalg.norm((C - Ck[inv]).reshape(M, -1), axis=1) / norm
    errR = np.abs(rho - rk[inv]) / np.maximum(np.abs(rho), 1e-300)
    stats = {"n_distinct": int(len(first)), "n_materials": int(K), "n_shapes": int(n_shapes),
             "max_rel_C_error": float(errC.max()), "mean_rel_C_error": float(errC.mean()),
             "max_rel_density_error": float(errR.max()) if use_density else 0.0}
    return MaterialTable(inv, Ck, rk, exact=False, stats=stats)


# --------------------------------------------------------------------------- #
# Deck writer
# --------------------------------------------------------------------------- #
def _needs_density(problem: Problem) -> bool:
    return any(c.gravity is not None and np.any(np.asarray(c.gravity) != 0)
               for c in problem.load_cases)


def write_model_inp(problem: Problem, path: str | Path, *,
                    max_materials: int | None = 20_000) -> MaterialTable:
    """Write the model definition (mesh, materials, Dirichlet BCs) to `path`.

    The file contains no ``*STEP``; it is meant to be included by load-case
    decks (see :func:`write_case_inp`). Returns the material table used.
    """
    mesh, mat = problem.mesh, problem.material
    M, N = mesh.n_elems, mesh.n_nodes
    table = group_materials(mat.C, mat.mass_density, use_density=_needs_density(problem),
                            max_materials=max_materials)
    iso, E, nu = isotropic_parameters(table.C)
    order = np.argsort(table.elem_material, kind="stable")
    bounds = np.searchsorted(table.elem_material[order], np.arange(table.n_materials + 1))
    elem_rows = np.column_stack([np.arange(1, M + 1), mesh.tets + 1]).astype(np.float64)

    with open(path, "w", newline="\n") as fh:
        fh.write(f"** craniofea model: {N} nodes, {M} C3D4 elements, "
                 f"{table.n_materials} materials ({'exact' if table.exact else 'binned'})\n")
        fh.write("*NODE, NSET=NALL\n")
        write_rows(fh, "%d, %.17g, %.17g, %.17g\n",
                   np.column_stack([np.arange(1, N + 1), mesh.nodes]))
        # Elements are written grouped by material so each block defines its elset.
        for k in range(table.n_materials):
            ids = order[bounds[k]:bounds[k + 1]]
            fh.write(f"*ELEMENT, TYPE=C3D4, ELSET=M{k + 1}\n")
            write_rows(fh, "%d, %d, %d, %d, %d\n", elem_rows[ids])
        fh.write(f"*ELSET, ELSET=EALL, GENERATE\n1, {M}, 1\n")

        dens = table.density
        aniso = aniso_constants(table.C)
        for k in range(table.n_materials):
            fh.write(f"*MATERIAL, NAME=M{k + 1}\n")
            if iso[k]:
                fh.write(f"*ELASTIC\n{E[k]:.17g}, {nu[k]:.17g}\n")
            else:
                d = [_ANISO_FMT % v for v in aniso[k]]
                fh.write("*ELASTIC, TYPE=ANISO\n" + ",".join(d[:8]) + "\n"
                         + ",".join(d[8:16]) + "\n" + ",".join(d[16:]) + "\n")
            fh.write(f"*DENSITY\n{dens[k]:.17g}\n")
            fh.write(f"*SOLID SECTION, ELSET=M{k + 1}, MATERIAL=M{k + 1}\n")

        mask = problem.constrained_mask()
        node, dof = np.nonzero(mask)
        if node.size:
            fh.write("*BOUNDARY\n")
            write_rows(fh, "%d, %d, %d\n", np.column_stack([node + 1, dof + 1, dof + 1]))
    return table


def write_case_inp(problem: Problem, case: LoadCase, path: str | Path,
                   model_file: str = "model.inp") -> None:
    """Write one static step for `case` that includes `model_file`."""
    mesh = problem.mesh
    with open(path, "w", newline="\n") as fh:
        fh.write(f"** craniofea load case: {case.name}\n")
        fh.write(f"*INCLUDE, INPUT={model_file}\n")
        fh.write("*STEP\n*STATIC\n")
        if case.nodal_forces is not None:
            F = np.asarray(case.nodal_forces, dtype=float).reshape(mesh.n_nodes, 3)
            node, dof = np.nonzero(F)
            if node.size:
                fh.write("*CLOAD\n")
                write_rows(fh, "%d, %d, %.17g\n",
                           np.column_stack([node + 1, dof + 1, F[node, dof]]))
        elem, lface, p = collect_face_pressures(mesh, case.pressures)
        nz = p != 0
        if np.any(nz):
            fh.write("*DLOAD\n")
            write_rows(fh, "%d, P%d, %.17g\n",
                       np.column_stack([elem[nz] + 1, lface[nz] + 1, p[nz]]))
        if case.gravity is not None:
            g = np.asarray(case.gravity, dtype=float)
            gn = float(np.linalg.norm(g))
            if gn > 0:
                d = g / gn
                fh.write(f"*DLOAD\nEALL, GRAV, {gn:.17g}, {d[0]:.17g}, {d[1]:.17g}, "
                         f"{d[2]:.17g}\n")
        fh.write("*NODE PRINT, NSET=NALL\nU\n")
        fh.write("*EL PRINT, ELSET=EALL\nS, E, ENER\n")
        fh.write("*END STEP\n")


def write_inp(problem: Problem, work_dir: str | Path, *,
              max_materials: int | None = 20_000) -> tuple[list[Path], MaterialTable]:
    """Write ``model.inp`` plus one ``<job>.inp`` per load case into `work_dir`.

    Returns (list of case deck paths in load-case order, material table).
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    table = write_model_inp(problem, work / "model.inp", max_materials=max_materials)
    decks = []
    for i, case in enumerate(problem.load_cases):
        p = work / f"{_job_name(i, case.name)}.inp"
        write_case_inp(problem, case, p)
        decks.append(p)
    return decks, table


def _job_name(i: int, name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")[:40] or "case"
    return f"case{i:02d}_{safe}"


# --------------------------------------------------------------------------- #
# Result reader
# --------------------------------------------------------------------------- #
_HEADER = re.compile(r"^\s*([a-z][a-z ]*?)\s*\(", re.MULTILINE)
_EXP3 = re.compile(r"(?<=\d)([+-]\d{3})\b")   # Fortran drops the 'E' for |exp| > 99


def _numbers(text: str) -> np.ndarray:
    try:
        return np.array(text.split(), dtype=float)
    except ValueError:
        return np.array(_EXP3.sub(r"E\1", text).split(), dtype=float)


def parse_dat(text: str) -> dict[str, np.ndarray]:
    """Parse the result blocks of a CalculiX ``.dat`` file (last time point).

    Returns raw tables keyed by block name: ``displacements`` (n, 4) with node
    id first; ``stresses``/``strains`` (n, 8) with element id and integration
    point first; ``internal energy density`` (n, 3). Values are in CalculiX
    order (xx, yy, zz, xy, xz, yz; tensor shear strains).
    """
    ncols = {"displacements": 4, "stresses": 8, "strains": 8,
             "internal energy density": 3}
    heads = list(_HEADER.finditer(text))
    out: dict[str, np.ndarray] = {}
    for i, h in enumerate(heads):
        name = h.group(1).strip()
        if name not in ncols:
            continue
        start = text.index("\n", h.end()) + 1
        stop = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        vals = _numbers(text[start:stop])
        if vals.size % ncols[name]:
            raise BackendError(f"malformed '{name}' block in .dat output")
        out[name] = vals.reshape(-1, ncols[name])
    return out


def dat_to_solution(text: str, n_nodes: int, n_elems: int) -> Solution:
    """Convert ``.dat`` text into a :class:`Solution` in craniofea conventions."""
    blocks = parse_dat(text)
    missing = {"displacements", "stresses", "strains", "internal energy density"} - set(blocks)
    if missing:
        raise BackendError(f"CalculiX .dat output lacks blocks: {sorted(missing)}")
    d = blocks["displacements"]
    u = np.zeros((n_nodes, 3))
    u[d[:, 0].astype(np.int64) - 1] = d[:, 1:4]

    def elem_mean(tab: np.ndarray, ncomp: int) -> np.ndarray:
        eid = tab[:, 0].astype(np.int64) - 1
        acc = np.zeros((n_elems, ncomp))
        np.add.at(acc, eid, tab[:, 2:2 + ncomp])
        cnt = np.bincount(eid, minlength=n_elems)
        if np.any(cnt == 0):
            raise BackendError("CalculiX .dat output misses some elements")
        return acc / cnt[:, None]

    stress = elem_mean(blocks["stresses"], 6)[:, _CCX_TO_VOIGT]
    strain = elem_mean(blocks["strains"], 6)[:, _CCX_TO_VOIGT]
    strain[:, 3:] *= 2.0                        # tensor -> engineering shear
    sed = elem_mean(blocks["internal energy density"], 1)[:, 0]
    return Solution(u=u, strain=strain, stress=stress, sed=sed)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class CalculixSolver:
    """Run a :class:`~craniofea.model.Problem` through CalculiX ``ccx``.

    Parameters
    ----------
    work_dir : directory for decks and results. ``None`` uses a temporary
        directory that is deleted afterwards; a given directory is kept.
    ccx : executable name or path (also read from ``$CCX`` if left default).
    n_threads : OpenMP threads for ccx (``OMP_NUM_THREADS``).
    max_materials : distinct-material limit before stiffness binning kicks in
        (``None`` = always exact, one material per distinct C).
    timeout : seconds per load case.
    equilibrium_tol : every result is checked for nodal equilibrium (internal
        forces rebuilt from the reported stresses vs. the applied loads, see
        :func:`~craniofea.backends._common.equilibrium_residual`). A multi-threaded
        run that fails the check is repeated once single-threaded; if it still
        fails a :class:`BackendError` is raised.
    """

    name = "calculix"

    def __init__(self, work_dir: str | Path | None = None, ccx: str = "ccx",
                 n_threads: int = 4, max_materials: int | None = 20_000,
                 timeout: float | None = None, equilibrium_tol: float = 1e-4):
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.ccx = ccx
        self.n_threads = int(n_threads)
        self.max_materials = max_materials
        self.timeout = timeout
        self.equilibrium_tol = float(equilibrium_tol)

    def executable(self) -> str | None:
        """Resolved ccx path, or None if not available."""
        cands = [self.ccx] if self.ccx != "ccx" else [os.environ.get("CCX", ""), "ccx"]
        return find_executable(cands)

    def solve(self, problem: Problem) -> dict[str, Solution]:
        exe = self.executable()
        if exe is None:
            raise BackendError(f"CalculiX executable {self.ccx!r} not found")
        if self.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="craniofea_ccx_") as tmp:
                return self._solve_in(problem, Path(tmp), exe)
        return self._solve_in(problem, self.work_dir, exe)

    def _run(self, exe: str, work: Path, job: str, n_threads: int, case_name: str) -> Path:
        """Run one ccx job; return the .dat path or raise with the tail of the log."""
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads),
                   CCX_NPROC_EQUATION_SOLVER=str(n_threads),
                   CCX_NPROC_STIFFNESS=str(n_threads), CCX_NPROC_RESULTS=str(n_threads))
        log = work / f"{job}.log"
        dat = work / f"{job}.dat"
        dat.unlink(missing_ok=True)
        with open(log, "w") as fh:
            try:
                proc = subprocess.run([exe, "-i", job], cwd=work, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT, timeout=self.timeout)
            except subprocess.TimeoutExpired as err:
                raise BackendError(f"ccx timed out on load case {case_name!r}") from err
        log_text = log.read_text(errors="replace")
        if proc.returncode != 0 or "*ERROR" in log_text or not dat.exists():
            raise BackendError(
                f"ccx failed on load case {case_name!r} (exit {proc.returncode}); "
                f"see {log}\n--- ccx output (tail) ---\n{tail(log)}")
        return dat

    def _solve_in(self, problem: Problem, work: Path, exe: str) -> dict[str, Solution]:
        t0 = time.perf_counter()
        decks, table = write_inp(problem, work, max_materials=self.max_materials)
        t_write = time.perf_counter() - t0
        mesh = problem.mesh
        out: dict[str, Solution] = {}
        for case, deck in zip(problem.load_cases, decks, strict=True):
            job = deck.stem
            t1 = time.perf_counter()
            dat = self._run(exe, work, job, self.n_threads, case.name)
            sol = dat_to_solution(dat.read_text(), mesh.n_nodes, mesh.n_elems)
            res = equilibrium_residual(problem, case, sol.stress)
            retried = False
            if not res <= self.equilibrium_tol and self.n_threads > 1:
                # Rare corrupted multi-threaded runs of ccx 2.17 have been observed on a
                # heavily loaded host; a single-threaded rerun is deterministic.
                retried = True
                dat = self._run(exe, work, job, 1, case.name)
                sol = dat_to_solution(dat.read_text(), mesh.n_nodes, mesh.n_elems)
                res = equilibrium_residual(problem, case, sol.stress)
            if not res <= self.equilibrium_tol:          # also catches NaN results
                raise BackendError(
                    f"CalculiX result for load case {case.name!r} violates nodal equilibrium "
                    f"(relative residual {res:.2e} > {self.equilibrium_tol:.0e}); see {work}")
            sol.info = {"solver": self.name, "executable": exe, "job": job,
                        "work_dir": str(work), "n_materials": table.n_materials,
                        "materials_exact": table.exact, "material_binning": table.stats,
                        "equilibrium_residual": res, "retried_single_thread": retried,
                        "time_write_s": t_write, "time_case_s": time.perf_counter() - t1,
                        "time_s": time.perf_counter() - t0}
            out[case.name] = sol
        return out

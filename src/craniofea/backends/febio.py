"""FEBio 4 backend: ``.feb`` (spec 4.0) writer, runner and logfile reader.

Mapping of the data model (see ``docs/BACKENDS.md``):

- nodes / elements: ``<Nodes>`` and ``tet4`` ``<Elements>`` parts with 1-based
  ids (id = index + 1). Elements are split into at most two parts, ``iso`` and
  ``ortho``, each with one material;
- materials: ``isotropic elastic`` (E, v) and ``orthotropic elastic`` (E1..v31
  plus a per-element ``mat_axis``). Every parameter that varies between
  elements is an ``ElementData`` map, so continuous density fields are written
  exactly. Isotropy is detected from ``C``; orthotropic constants are derived
  from ``C`` rotated into ``Material.axes``. FEBio has no general (21-constant)
  anisotropic linear material, so other matrices are rejected;
- Dirichlet: ``zero displacement`` BCs on node sets grouped by fixed-dof pattern;
- pressure: ``pressure`` surface loads with per-face ``SurfaceData`` maps;
- gravity: ``body force`` with mass densities;
- nodal forces: ``nodal_force`` with a ``NodeData`` vec3 map.

FEBio solves every static problem with a finite-deformation Newton method; both
materials are St. Venant-Kirchhoff (``S = C : E_Green``). They reproduce linear
elasticity to first order in the strain, so results differ from a linear
solver by O(|strain|) relative (about 1e-6 for physiological bone loads); see
``docs/BACKENDS.md``. One ``.feb`` file is written per load case.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from craniofea.backends._common import (
    BackendError,
    equilibrium_residual,
    find_executable,
    isotropic_parameters,
    orthotropic_parameters,
    rotate_stiffness,
    tail,
    write_id_list,
    write_rows,
)
from craniofea.model import LoadCase, Problem, Solution

# Logfile variables in craniofea Voigt order [xx, yy, zz, yz, xz, xy].
STRESS_VARS = ("sx", "sy", "sz", "syz", "sxz", "sxy")
# Green-Lagrange strain (tensor shear), the work conjugate of the SVK material.
# It differs from sym(grad u) by 0.5 grad(u)^T grad(u), i.e. O(|strain|) relative.
# The infinitesimal-strain variables "eyz"/"exz" are not used: FEBio 4.13 swaps
# them (FEElasticMaterialPoint::SmallStrain passes xz and yz in the wrong order).
STRAIN_VARS = ("Ex", "Ey", "Ez", "Eyz", "Exz", "Exy")
DISP_VARS = ("ux", "uy", "uz")
ORTHO_PARAMS = ("E1", "E2", "E3", "G12", "G23", "G31", "v12", "v23", "v31")


@dataclass
class FebioMaterialSplit:
    """Per-element FEBio material description derived from a ``Material``.

    iso_elems / ortho_elems : element indices of each part (ascending).
    iso : (n_iso, 2) [E, v].
    ortho : (n_ortho, 9) FEBio constants in ``ORTHO_PARAMS`` order.
    axes : (n_ortho, 3, 3) material axes (rows) of the orthotropic elements.
    """

    iso_elems: np.ndarray
    ortho_elems: np.ndarray
    iso: np.ndarray
    ortho: np.ndarray
    axes: np.ndarray


def split_materials(material, rtol: float = 1e-8) -> FebioMaterialSplit:
    """Classify every element as isotropic or orthotropic-in-its-axes.

    Isotropic elements are detected from ``C`` directly. For the others,
    ``C`` is rotated into ``material.axes`` (identity if axes are absent) and must
    be orthotropic there; the nine engineering constants are computed from the
    rotated compliance, so they are exact for any rotated orthotropic ``C``.
    """
    C = np.asarray(material.C, dtype=float)
    M = C.shape[0]
    iso, E, nu = isotropic_parameters(C, rtol=rtol)
    iso_e = np.flatnonzero(iso)
    ort_e = np.flatnonzero(~iso)
    if material.axes is not None:
        axes = np.asarray(material.axes, dtype=float).reshape(M, 3, 3)[ort_e]
    else:
        axes = np.broadcast_to(np.eye(3), (len(ort_e), 3, 3)).copy()
    ortho = np.zeros((len(ort_e), 9))
    if len(ort_e):
        ok, eng = orthotropic_parameters(rotate_stiffness(C[ort_e], axes), rtol=rtol)
        if not np.all(ok):
            bad = ort_e[~ok]
            raise ValueError(
                f"{bad.size} elements (first: {bad[0]}) are neither isotropic nor orthotropic "
                "in Material.axes; FEBio has no general anisotropic linear material.")
        E1, E2, E3, G12, G13, G23, n12, n13, n23 = eng.T
        ortho = np.column_stack([E1, E2, E3, G12, G23, G13, n12, n23, n13 * E3 / E1])
    return FebioMaterialSplit(iso_e, ort_e, np.column_stack([E[iso_e], nu[iso_e]]),
                              ortho, axes)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def _case(problem: Problem, case: int | str) -> LoadCase:
    if isinstance(case, str):
        for c in problem.load_cases:
            if c.name == case:
                return c
        raise KeyError(f"no load case named {case!r}")
    return problem.load_cases[case]


def _xml_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s)


def _param(fh: TextIO, name: str, values: np.ndarray, map_name: str, maps: list) -> None:
    """Write a material parameter: constant value, or a reference to an ElementData map.

    Values equal to round-off (parameters recovered from C carry ULP noise) are
    written as one constant.
    """
    v = np.asarray(values, dtype=float)
    if v.size and np.allclose(v, v[0], rtol=1e-13, atol=0.0):
        fh.write(f"\t\t\t<{name}>{v[0]:.17g}</{name}>\n")
    else:
        fh.write(f'\t\t\t<{name} type="map">{map_name}</{name}>\n')
        maps.append((map_name, v))


def _write_element_data(fh: TextIO, name: str, elem_set: str, values: np.ndarray) -> None:
    fh.write(f'\t\t<ElementData name="{name}" elem_set="{elem_set}">\n')
    write_rows(fh, '\t\t\t<e lid="%d">%.17g</e>\n',
               np.column_stack([np.arange(1, len(values) + 1), values]))
    fh.write("\t\t</ElementData>\n")


def write_feb(problem: Problem, path: str | Path, case: int | str = 0, *,
              linear_solver: str | None = None, output_prefix: str | None = None,
              dtol: float = 1e-7, etol: float = 1e-10) -> Path:
    """Write one load case of `problem` as an FEBio 4 (spec 4.0) input file.

    Parameters
    ----------
    case : load-case index or name.
    linear_solver : FEBio linear solver type (e.g. ``"pardiso"``, ``"skyline"``);
        ``None`` keeps FEBio's default (pardiso in MKL builds).
    output_prefix : logfile data files are ``<prefix>_u.txt``, ``_s.txt``,
        ``_e.txt`` and ``_sed.txt`` next to the ``.feb`` (default: file stem).
    dtol, etol : Newton displacement / energy convergence tolerances. Full
        Newton reformation is used, so the solution reaches round-off within
        about three iterations; values much tighter than the defaults sit below
        the round-off floor and make FEBio report non-convergence.
    """
    path = Path(path)
    lc = _case(problem, case)
    mesh, mat = problem.mesh, problem.material
    N, M = mesh.n_nodes, mesh.n_elems
    split = split_materials(mat)
    rho = np.asarray(mat.mass_density, dtype=float).reshape(M)
    prefix = output_prefix or path.stem
    maps: list[tuple[str, str, np.ndarray]] = []   # (map name, elem set, values)

    with open(path, "w", newline="\n") as fh:
        fh.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        fh.write(f"<!-- craniofea: {N} nodes, {M} tet4 elements, load case "
                 f"{lc.name!r} -->\n")
        fh.write('<febio_spec version="4.0">\n\t<Module type="solid"/>\n')

        # Control
        fh.write("\t<Control>\n\t\t<analysis>STATIC</analysis>\n"
                 "\t\t<time_steps>1</time_steps>\n\t\t<step_size>1</step_size>\n"
                 "\t\t<plot_zero_state>0</plot_zero_state>\n"
                 '\t\t<time_stepper type="default">\n\t\t\t<max_retries>5</max_retries>\n'
                 "\t\t\t<opt_iter>10</opt_iter>\n\t\t\t<dtmin>0.01</dtmin>\n"
                 "\t\t\t<dtmax>1</dtmax>\n\t\t</time_stepper>\n"
                 '\t\t<solver type="solid">\n'
                 "\t\t\t<symmetric_stiffness>symmetric</symmetric_stiffness>\n"
                 f"\t\t\t<dtol>{dtol:g}</dtol>\n\t\t\t<etol>{etol:g}</etol>\n"
                 "\t\t\t<rtol>0</rtol>\n\t\t\t<lstol>0.9</lstol>\n"
                 "\t\t\t<min_residual>1e-30</min_residual>\n\t\t\t<max_refs>25</max_refs>\n"
                 "\t\t\t<diverge_reform>1</diverge_reform>\n"
                 "\t\t\t<reform_each_time_step>1</reform_each_time_step>\n"
                 '\t\t\t<qn_method type="BFGS">\n\t\t\t\t<max_ups>0</max_ups>\n'
                 "\t\t\t</qn_method>\n")
        if linear_solver:
            fh.write(f'\t\t\t<linear_solver type="{linear_solver}"/>\n')
        fh.write("\t\t</solver>\n\t</Control>\n")

        # Materials
        fh.write("\t<Material>\n")
        mat_id = 0
        if len(split.iso_elems):
            mat_id += 1
            fh.write(f'\t\t<material id="{mat_id}" name="iso" type="isotropic elastic">\n')
            pm: list = []
            _param(fh, "density", rho[split.iso_elems], "iso_density", pm)
            _param(fh, "E", split.iso[:, 0], "iso_E", pm)
            _param(fh, "v", split.iso[:, 1], "iso_v", pm)
            maps += [(n, "iso", v) for n, v in pm]
            fh.write("\t\t</material>\n")
        if len(split.ortho_elems):
            mat_id += 1
            fh.write(f'\t\t<material id="{mat_id}" name="ortho" type="orthotropic elastic">\n')
            pm = []
            _param(fh, "density", rho[split.ortho_elems], "ortho_density", pm)
            for k, pname in enumerate(ORTHO_PARAMS):
                _param(fh, pname, split.ortho[:, k], f"ortho_{pname}", pm)
            maps += [(n, "ortho", v) for n, v in pm]
            fh.write("\t\t</material>\n")
        fh.write("\t</Material>\n")

        # Mesh
        fh.write('\t<Mesh>\n\t\t<Nodes name="all_nodes">\n')
        write_rows(fh, '\t\t\t<node id="%d">%.17g,%.17g,%.17g</node>\n',
                   np.column_stack([np.arange(1, N + 1), mesh.nodes]))
        fh.write("\t\t</Nodes>\n")
        for part, elems in (("iso", split.iso_elems), ("ortho", split.ortho_elems)):
            if len(elems):
                fh.write(f'\t\t<Elements type="tet4" name="{part}">\n')
                write_rows(fh, '\t\t\t<elem id="%d">%d,%d,%d,%d</elem>\n',
                           np.column_stack([elems + 1, mesh.tets[elems] + 1]))
                fh.write("\t\t</Elements>\n")

        mask = problem.constrained_mask()
        bc_sets = []
        pattern = mask @ np.array([1, 2, 4])
        for code in range(1, 8):
            nodes_c = np.flatnonzero(pattern == code)
            if nodes_c.size:
                name = "fix_" + "".join(a for a, b in zip("xyz", (1, 2, 4), strict=True)
                                        if code & b)
                bc_sets.append((name, code))
                fh.write(f'\t\t<NodeSet name="{name}">\n')
                write_id_list(fh, nodes_c + 1, sep=",")
                fh.write("\t\t</NodeSet>\n")

        F = None
        if lc.nodal_forces is not None:
            F = np.asarray(lc.nodal_forces, dtype=float).reshape(N, 3)
            f_nodes = np.flatnonzero(np.any(F != 0, axis=1))
            if f_nodes.size:
                fh.write('\t\t<NodeSet name="loaded_nodes">\n')
                write_id_list(fh, f_nodes + 1, sep=",")
                fh.write("\t\t</NodeSet>\n")
            else:
                F = None

        surfaces = []
        for set_name, p in lc.pressures.items():
            if set_name not in mesh.surface_sets:
                raise KeyError(f"pressure on unknown surface set {set_name!r}")
            faces = np.asarray(mesh.surface_sets[set_name], dtype=np.int64).reshape(-1, 3)
            if not len(faces):
                continue
            sname = f"p_{_xml_name(set_name)}"
            pv = np.broadcast_to(np.asarray(p, dtype=float), (len(faces),))
            surfaces.append((sname, pv))
            fh.write(f'\t\t<Surface name="{sname}">\n')
            write_rows(fh, '\t\t\t<tri3 id="%d">%d,%d,%d</tri3>\n',
                       np.column_stack([np.arange(1, len(faces) + 1), faces + 1]))
            fh.write("\t\t</Surface>\n")
        fh.write("\t</Mesh>\n")

        fh.write("\t<MeshDomains>\n")
        if len(split.iso_elems):
            fh.write('\t\t<SolidDomain name="iso" mat="iso"/>\n')
        if len(split.ortho_elems):
            fh.write('\t\t<SolidDomain name="ortho" mat="ortho"/>\n')
        fh.write("\t</MeshDomains>\n")

        # Mesh data
        fh.write("\t<MeshData>\n")
        for name, elem_set, values in maps:
            _write_element_data(fh, name, elem_set, values)
        if len(split.ortho_elems):
            fh.write('\t\t<ElementData type="mat_axis" elem_set="ortho">\n')
            ax = split.axes
            write_rows(fh, '\t\t\t<e lid="%d"><a>%.17g,%.17g,%.17g</a>'
                           '<d>%.17g,%.17g,%.17g</d></e>\n',
                       np.column_stack([np.arange(1, len(ax) + 1), ax[:, 0], ax[:, 1]]))
            fh.write("\t\t</ElementData>\n")
        for sname, pv in surfaces:
            if not np.all(pv == pv[0]):
                fh.write(f'\t\t<SurfaceData name="{sname}_values" surface="{sname}">\n')
                write_rows(fh, '\t\t\t<face lid="%d">%.17g</face>\n',
                           np.column_stack([np.arange(1, len(pv) + 1), pv]))
                fh.write("\t\t</SurfaceData>\n")
        if F is not None:
            fh.write('\t\t<NodeData name="nodal_force_values" node_set="loaded_nodes" '
                     'data_type="vec3">\n')
            write_rows(fh, '\t\t\t<node lid="%d">%.17g,%.17g,%.17g</node>\n',
                       np.column_stack([np.arange(1, f_nodes.size + 1), F[f_nodes]]))
            fh.write("\t\t</NodeData>\n")
        fh.write("\t</MeshData>\n")

        # Boundary conditions
        fh.write("\t<Boundary>\n")
        for name, code in bc_sets:
            fh.write(f'\t\t<bc name="{name}" node_set="{name}" type="zero displacement">\n'
                     f"\t\t\t<x_dof>{int(bool(code & 1))}</x_dof>\n"
                     f"\t\t\t<y_dof>{int(bool(code & 2))}</y_dof>\n"
                     f"\t\t\t<z_dof>{int(bool(code & 4))}</z_dof>\n\t\t</bc>\n")
        fh.write("\t</Boundary>\n")

        # Loads (ramped by load controller 1 over the single time step)
        fh.write("\t<Loads>\n")
        for sname, pv in surfaces:
            fh.write(f'\t\t<surface_load name="{sname}" surface="{sname}" type="pressure">\n')
            if np.all(pv == pv[0]):
                fh.write(f'\t\t\t<pressure lc="1">{pv[0]:.17g}</pressure>\n')
            else:
                fh.write(f'\t\t\t<pressure lc="1" type="map">{sname}_values</pressure>\n')
            fh.write("\t\t\t<symmetric_stiffness>1</symmetric_stiffness>\n"
                     "\t\t\t<linear>1</linear>\n\t\t</surface_load>\n")
        if lc.gravity is not None and np.any(np.asarray(lc.gravity) != 0):
            g = np.asarray(lc.gravity, dtype=float)
            # FEBio adds -rho * force to the external load vector, so the body
            # force parameter is the negative of the gravitational acceleration.
            fh.write('\t\t<body_load name="gravity" type="body force">\n'
                     f'\t\t\t<force lc="1">{-g[0]:.17g},{-g[1]:.17g},{-g[2]:.17g}</force>\n'
                     "\t\t</body_load>\n")
        if F is not None:
            fh.write('\t\t<nodal_load name="nodal_forces" node_set="loaded_nodes" '
                     'type="nodal_force">\n'
                     '\t\t\t<value lc="1" type="map">nodal_force_values</value>\n'
                     "\t\t</nodal_load>\n")
        fh.write("\t</Loads>\n")

        fh.write('\t<LoadData>\n\t\t<load_controller id="1" name="ramp" type="loadcurve">\n'
                 "\t\t\t<interpolate>LINEAR</interpolate>\n\t\t\t<points>\n"
                 "\t\t\t\t<pt>0,0</pt>\n\t\t\t\t<pt>1,1</pt>\n\t\t\t</points>\n"
                 "\t\t</load_controller>\n\t</LoadData>\n")

        fh.write("\t<Output>\n\t\t<logfile>\n"
                 f'\t\t\t<node_data data="{";".join(DISP_VARS)}" delim="," '
                 f'file="{prefix}_u.txt"/>\n'
                 f'\t\t\t<element_data data="{";".join(STRESS_VARS)}" delim="," '
                 f'file="{prefix}_s.txt"/>\n'
                 f'\t\t\t<element_data data="{";".join(STRAIN_VARS)}" delim="," '
                 f'file="{prefix}_e.txt"/>\n'
                 f'\t\t\t<element_data data="sed" delim="," file="{prefix}_sed.txt"/>\n'
                 '\t\t</logfile>\n\t\t<plotfile type="febio">\n'
                 '\t\t\t<var type="displacement"/>\n\t\t\t<var type="stress"/>\n'
                 "\t\t</plotfile>\n\t</Output>\n</febio_spec>\n")
    return path


# --------------------------------------------------------------------------- #
# Reader
# --------------------------------------------------------------------------- #
def read_logfile_data(path: str | Path, n_items: int, n_values: int) -> np.ndarray:
    """Values of the LAST record of an FEBio logfile data file.

    Files consist of records ``*Step = .. / *Time = .. / *Data = ..`` followed by
    ``id,v1,v2,...`` lines. Returns (n_items, n_values) ordered by item id.
    """
    text = Path(path).read_text()
    last = text.rfind("*Data")
    if last < 0:
        raise BackendError(f"no data records in {path}")
    body = text[text.index("\n", last) + 1:]
    vals = np.array(body.replace(",", " ").split(), dtype=float)
    if vals.size % (n_values + 1):
        raise BackendError(f"malformed FEBio data file {path}")
    tab = vals.reshape(-1, n_values + 1)
    ids = tab[:, 0].astype(np.int64) - 1
    if len(ids) != n_items or np.any(np.sort(ids) != np.arange(n_items)):
        raise BackendError(f"{path}: expected {n_items} items, found {len(ids)}")
    out = np.empty((n_items, n_values))
    out[ids] = tab[:, 1:]
    return out


def read_solution(work_dir: str | Path, prefix: str, n_nodes: int, n_elems: int) -> Solution:
    """Assemble a :class:`Solution` from the logfile outputs of :func:`write_feb`."""
    w = Path(work_dir)
    u = read_logfile_data(w / f"{prefix}_u.txt", n_nodes, 3)
    stress = read_logfile_data(w / f"{prefix}_s.txt", n_elems, 6)
    strain = read_logfile_data(w / f"{prefix}_e.txt", n_elems, 6)
    strain[:, 3:] *= 2.0                       # tensor -> engineering shear
    sed = read_logfile_data(w / f"{prefix}_sed.txt", n_elems, 1)[:, 0]
    return Solution(u=u, strain=strain, stress=stress, sed=sed)


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
DEFAULT_FEBIO_CANDIDATES = ("febio4", "~/opt/febio/bin/febio4")


class FebioSolver:
    """Run a :class:`~craniofea.model.Problem` through FEBio 4.

    Parameters
    ----------
    work_dir : directory for inputs and results; ``None`` = temporary directory
        removed afterwards.
    febio : executable name or path. With the default, ``$FEBIO``, ``febio4`` on
        PATH and ``~/opt/febio/bin/febio4`` (``scripts/install_febio.sh``) are tried.
    n_threads : ``OMP_NUM_THREADS`` for FEBio.
    linear_solver : passed to :func:`write_feb` (``None`` = FEBio default:
        pardiso in MKL builds; builds without MKL select skyline themselves).
    equilibrium_tol : results are checked for nodal equilibrium against the
        linear load vector. FEBio's Cauchy stresses act on the deformed geometry,
        so the accepted residual is ``equilibrium_tol + 10 * max|strain|``.
    """

    name = "febio"

    def __init__(self, work_dir: str | Path | None = None, febio: str = "febio4",
                 n_threads: int = 4, linear_solver: str | None = None,
                 timeout: float | None = None, equilibrium_tol: float = 1e-3):
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.febio = febio
        self.n_threads = int(n_threads)
        self.linear_solver = linear_solver
        self.timeout = timeout
        self.equilibrium_tol = float(equilibrium_tol)

    def executable(self) -> str | None:
        """Resolved FEBio path, or None if not available."""
        if self.febio != "febio4":
            return find_executable([self.febio])
        return find_executable([os.environ.get("FEBIO", ""), *DEFAULT_FEBIO_CANDIDATES])

    def solve(self, problem: Problem) -> dict[str, Solution]:
        exe = self.executable()
        if exe is None:
            raise BackendError(f"FEBio executable {self.febio!r} not found "
                               "(build it with scripts/install_febio.sh)")
        if self.work_dir is None:
            with tempfile.TemporaryDirectory(prefix="craniofea_febio_") as tmp:
                return self._solve_in(problem, Path(tmp), exe)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self._solve_in(problem, self.work_dir, exe)

    def _run(self, exe: str, feb: Path, case_name: str) -> tuple[subprocess.CompletedProcess,
                                                                    Path]:
        log = feb.with_suffix(".stdout")
        env = dict(os.environ, OMP_NUM_THREADS=str(self.n_threads))
        with open(log, "w") as fh:
            try:
                proc = subprocess.run([exe, "-i", feb.name, "-silent", "-noconfig"],
                                      cwd=feb.parent, env=env, stdout=fh,
                                      stderr=subprocess.STDOUT, timeout=self.timeout)
            except subprocess.TimeoutExpired as err:
                raise BackendError(f"FEBio timed out on load case {case_name!r}") from err
        return proc, log

    def _solve_in(self, problem: Problem, work: Path, exe: str) -> dict[str, Solution]:
        t0 = time.perf_counter()
        mesh = problem.mesh
        out: dict[str, Solution] = {}
        for i, case in enumerate(problem.load_cases):
            t1 = time.perf_counter()
            stem = f"case{i:02d}_{_xml_name(case.name)[:40]}"
            feb = work / f"{stem}.feb"
            write_feb(problem, feb, i, linear_solver=self.linear_solver)
            proc, log = self._run(exe, feb, case.name)
            febio_log = feb.with_suffix(".log")
            text = febio_log.read_text(errors="replace") if febio_log.exists() else ""
            if "N O R M A L   T E R M I N A T I O N" not in text:
                raise BackendError(
                    f"FEBio failed on load case {case.name!r} (exit {proc.returncode}); see "
                    f"{febio_log}\n--- FEBio log (tail) ---\n{tail(febio_log) or tail(log)}")
            sol = read_solution(work, stem, mesh.n_nodes, mesh.n_elems)
            res = equilibrium_residual(problem, case, sol.stress)
            tol = self.equilibrium_tol + 10.0 * float(np.abs(sol.strain).max(initial=0.0))
            if not res <= tol:                           # also catches NaN results
                raise BackendError(
                    f"FEBio result for load case {case.name!r} violates nodal equilibrium "
                    f"(relative residual {res:.2e} > {tol:.1e}); see {work}")
            iters = re.findall(r"Total number of equilibrium iterations\s*\.*\s*:\s*(\d+)", text)
            solver_used = re.findall(r"Default linear solver\s*\.*\s*:\s*(\S+)", text)
            sol.info = {"solver": self.name, "executable": exe, "input": str(feb),
                        "work_dir": str(work),
                        "linear_solver": self.linear_solver or
                        (solver_used[-1] if solver_used else "default"),
                        "newton_iterations": int(iters[-1]) if iters else None,
                        "equilibrium_residual": res,
                        "time_case_s": time.perf_counter() - t1,
                        "time_s": time.perf_counter() - t0}
            out[case.name] = sol
        return out

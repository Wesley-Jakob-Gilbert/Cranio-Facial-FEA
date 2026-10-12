"""Core data model shared by every part of the pipeline.

These containers are the contract between the mesh builders, the material
model, the solver backends (JAX, CalculiX, FEBio) and the remodeling engine.
Everything is plain NumPy, SI units, 0-based indices.

Conventions
-----------
- Units: metres, pascals, newtons, kilograms, seconds (days only in remodeling).
- Voigt order for symmetric tensors: [xx, yy, zz, yz, xz, xy].
  Stress is stored as tensor components; strain uses ENGINEERING shear
  (gamma_ij = 2 * eps_ij), so that  sed = 0.5 * stress . strain  and
  stress = C @ strain  with C the 6x6 Voigt stiffness.
- Tetrahedra are linear (C3D4 / tet4), positively oriented
  (det[x1-x0, x2-x0, x3-x0] > 0).
- Surface sets are boundary triangles with OUTWARD normals by right-hand rule,
  i.e. cross(x1-x0, x2-x0) points out of the solid.
- A positive pressure pushes INTO the solid (acts along -outward_normal),
  which is the standard FE convention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")


@dataclass(frozen=True)
class AnatomicalFrame:
    """Orthonormal anatomical axes expressed in mesh coordinates.

    lr: patient left -> right, ap: posterior -> anterior,
    si: inferior -> superior. origin is a convenient reference point
    (e.g. the centroid of the mesh).
    """

    origin: np.ndarray
    lr: np.ndarray
    ap: np.ndarray
    si: np.ndarray

    def to_local(self, x: np.ndarray) -> np.ndarray:
        """Map (..., 3) mesh coordinates to (lr, ap, si) coordinates."""
        R = np.stack([self.lr, self.ap, self.si])
        return (np.asarray(x) - self.origin) @ R.T


@dataclass
class Mesh:
    nodes: np.ndarray                                   # (N, 3) float64, m
    tets: np.ndarray                                    # (M, 4) int64, 0-based
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)     # name -> node ids
    elem_sets: dict[str, np.ndarray] = field(default_factory=dict)     # name -> element ids
    surface_sets: dict[str, np.ndarray] = field(default_factory=dict)  # name -> (F, 3) node ids
    elem_fields: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (M, ...) arrays
    frame: AnatomicalFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_elems(self) -> int:
        return int(self.tets.shape[0])

    def save(self, path: str | Path) -> None:
        arrays: dict[str, np.ndarray] = {"nodes": self.nodes, "tets": self.tets}
        for k, v in self.node_sets.items():
            arrays[f"nset__{k}"] = np.asarray(v)
        for k, v in self.elem_sets.items():
            arrays[f"eset__{k}"] = np.asarray(v)
        for k, v in self.surface_sets.items():
            arrays[f"sset__{k}"] = np.asarray(v)
        for k, v in self.elem_fields.items():
            arrays[f"efld__{k}"] = np.asarray(v)
        if self.frame is not None:
            for k in ("origin", "lr", "ap", "si"):
                arrays[f"frame__{k}"] = getattr(self.frame, k)
        import json

        arrays["meta_json"] = np.array(json.dumps(self.meta))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Mesh":
        import json

        d = np.load(path, allow_pickle=False)
        frame = None
        if "frame__origin" in d:
            frame = AnatomicalFrame(*(d[f"frame__{k}"] for k in ("origin", "lr", "ap", "si")))
        return cls(
            nodes=d["nodes"],
            tets=d["tets"],
            node_sets={k[6:]: d[k] for k in d.files if k.startswith("nset__")},
            elem_sets={k[6:]: d[k] for k in d.files if k.startswith("eset__")},
            surface_sets={k[6:]: d[k] for k in d.files if k.startswith("sset__")},
            elem_fields={k[6:]: d[k] for k in d.files if k.startswith("efld__")},
            frame=frame,
            meta=json.loads(str(d["meta_json"])) if "meta_json" in d else {},
        )


@dataclass
class Material:
    """Per-element linear elastic stiffness.

    C: (M, 6, 6) Voigt stiffness in GLOBAL coordinates (already rotated).
    mass_density: (M,) kg/m^3, used for gravity.
    Optional descriptive fields let backends that cannot take a full C per
    element (CalculiX, FEBio) write an equivalent model:
      E, nu: (M,) isotropic parameters where applicable (NaN otherwise)
      ortho: (M, 9) [E1, E2, E3, G12, G13, G23, nu12, nu13, nu23] or None
      axes: (M, 3, 3) local material axes as rows (global coords) or None
    """

    C: np.ndarray
    mass_density: np.ndarray
    E: np.ndarray | None = None
    nu: np.ndarray | None = None
    ortho: np.ndarray | None = None
    axes: np.ndarray | None = None


@dataclass
class LoadCase:
    name: str
    # surface set name -> pressure in Pa: scalar, or (F,) per-face array
    pressures: dict[str, float | np.ndarray] = field(default_factory=dict)
    gravity: np.ndarray | None = None          # (3,) m/s^2, e.g. [0, 0, -9.81]
    nodal_forces: np.ndarray | None = None     # (N, 3) N, optional
    # Daily loading description used by the remodeling stimulus
    cycles_per_day: float = 1.0


@dataclass
class Problem:
    mesh: Mesh
    material: Material
    fixed_nodes: np.ndarray                     # node ids with all 3 dofs fixed (u = 0)
    load_cases: list[LoadCase]
    fixed_dofs: np.ndarray | None = None        # optional (N, 3) bool, ORed with fixed_nodes

    def constrained_mask(self) -> np.ndarray:
        """(N, 3) bool mask of homogeneous Dirichlet dofs."""
        mask = np.zeros((self.mesh.n_nodes, 3), dtype=bool)
        mask[np.asarray(self.fixed_nodes, dtype=np.int64)] = True
        if self.fixed_dofs is not None:
            mask |= np.asarray(self.fixed_dofs, dtype=bool)
        return mask


@dataclass
class Solution:
    u: np.ndarray            # (N, 3) displacement, m
    strain: np.ndarray       # (M, 6) Voigt, engineering shear
    stress: np.ndarray       # (M, 6) Voigt, Pa
    sed: np.ndarray          # (M,) strain energy density, J/m^3 = Pa
    info: dict = field(default_factory=dict)   # solver diagnostics (iterations, residual, time)


class Solver(Protocol):
    name: str

    def solve(self, problem: Problem) -> dict[str, Solution]:
        """Solve every load case of `problem`; keyed by LoadCase.name."""
        ...

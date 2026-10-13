"""Re-meshing of the CT bone geometry at prescribed element sizes (gmsh).

For mesh-convergence studies the boundary of the imported CT mesh is
re-parametrised and re-meshed at a target edge length:

1. the boundary triangulation is loaded into gmsh as a discrete surface,
2. ``classifySurfaces`` splits it into patches at sharp edges (and into
   disc-like patches suitable for reparametrisation),
3. ``createGeometry`` builds a parametrisation of every patch,
4. surfaces and volume are meshed with the target size (Frontal-Delaunay
   surfaces, Delaunay volume, Netgen-free optimisation).

Labels are then recomputed with the same rules and parameters as the source
mesh; the tooth-socket faces (which the re-meshed geometry no longer knows
from the PDL) are transferred from the source surface by nearest-face
lookup with normal agreement. The anatomical frame of the source is reused,
so all meshes of a convergence study share one coordinate system.

gmsh is an optional dependency (``pip install craniofea[remesh]``).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from craniofea.mesh.ct import compact
from craniofea.mesh.primitives import boundary_faces, face_normals_areas, orient_tets
from craniofea.mesh.surface import SurfaceQuery
from craniofea.model import Mesh

log = logging.getLogger(__name__)

MM = 1e-3


class RemeshError(RuntimeError):
    pass


def transfer_surface_set(source: Mesh, name: str, target: Mesh,
                         max_distance: float, min_cos: float = 0.5) -> np.ndarray:
    """Boundary faces of `target` whose nearest `source` boundary face is in set `name`.

    A face is transferred when its centroid lies within `max_distance` of
    the source surface and its normal agrees with the nearest source face
    (cosine > min_cos), which keeps the two sides of thin plates apart.
    """
    from craniofea.mesh.primitives import face_keys

    src = source.surface_sets["all"]
    flag = np.isin(face_keys(src), face_keys(source.surface_sets[name]))
    tgt = target.surface_sets["all"]
    d, idx = SurfaceQuery(source.nodes, src).query(target.nodes[tgt].mean(axis=1))
    n_src, _ = face_normals_areas(source.nodes, src)
    n_tgt, _ = face_normals_areas(target.nodes, tgt)
    agree = np.einsum("ij,ij->i", n_src[idx], n_tgt) > min_cos
    return tgt[flag[idx] & agree & (d < max_distance)]


def thickness_sizes(nodes: np.ndarray, tets: np.ndarray, depth: np.ndarray,
                    size_max: float, size_min: float,
                    elements_through_thickness: float = 1.0, k: int = 32) -> np.ndarray:
    """Target element size at every node, limited by the local wall thickness.

    The thickness near a node is estimated as twice the largest element depth
    (distance to the surface) among the k elements nearest to it; the size is
    clip(thickness / elements_through_thickness, size_min, size_max).
    Coarsening a plate beyond its thickness (alveolar walls and interdental
    septa are 0.5-2 mm thick) makes the re-meshed faces of its two sides
    intersect, which is why a uniform size fails on this geometry.
    """
    from scipy.spatial import cKDTree

    _, j = cKDTree(nodes[tets].mean(axis=1)).query(nodes, k=k)
    t = 2.0 * np.asarray(depth)[j].max(axis=1)
    return np.clip(t / elements_through_thickness, size_min, size_max)


def remesh_surface(nodes: np.ndarray, faces: np.ndarray, size: float,
                   angle_deg: float = 40.0, size_min_factor: float = 0.25,
                   background: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                   optimize: bool = True, threads: int = 2, algorithm2d: int = 6,
                   algorithm3d: int = 1,
                   verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralise the closed triangulated surface (nodes, faces) at edge length `size`.

    Coordinates are passed to gmsh unchanged. `background` = (nodes, tets,
    size per node) of a tet mesh covering the domain defines a size field
    (gmsh PostView field, evaluated in C++). Returns (nodes, tets).
    """
    try:
        import gmsh
    except ImportError as err:
        raise RemeshError("gmsh is not installed (pip install craniofea[remesh])") from err

    nodes_c, faces_c, _ = compact(np.asarray(nodes, float), np.asarray(faces, np.int64))
    gmsh.initialize(["-noenv"], readConfigFiles=False)
    try:
        gmsh.option.setNumber("General.Terminal", int(verbose))
        gmsh.option.setNumber("General.NumThreads", threads)
        gmsh.model.add("remesh")
        s = gmsh.model.addDiscreteEntity(2)
        tags = np.arange(1, len(nodes_c) + 1)
        gmsh.model.mesh.addNodes(2, s, tags.tolist(), nodes_c.ravel().tolist())
        gmsh.model.mesh.addElementsByType(s, 2, [], (faces_c + 1).ravel().tolist())
        t0 = time.time()
        gmsh.model.mesh.classifySurfaces(np.radians(angle_deg), True, True, np.pi)
        gmsh.model.mesh.createGeometry()
        surfaces = [t for _, t in gmsh.model.getEntities(2)]
        log.info("gmsh reparametrisation: %d patches in %.0f s", len(surfaces),
                 time.time() - t0)
        loop = gmsh.model.geo.addSurfaceLoop(surfaces)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size * size_min_factor)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm2d)
        gmsh.option.setNumber("Mesh.Algorithm3D", algorithm3d)
        gmsh.option.setNumber("Mesh.Optimize", int(optimize))
        if background is not None:
            bn, bt, bs = background
            x = bn[bt]                                            # (M, 4, 3)
            data = np.concatenate([x[:, :, 0], x[:, :, 1], x[:, :, 2], bs[bt]], axis=1)
            view = gmsh.view.add("size")
            gmsh.view.addListData(view, "SS", len(bt), data.ravel().tolist())
            del data, x
            field = gmsh.model.mesh.field.add("PostView")
            gmsh.model.mesh.field.setNumber(field, "ViewTag", view)
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        t0 = time.time()
        gmsh.model.mesh.generate(2)
        log.info("gmsh surface mesh: %.0f s", time.time() - t0)
        gmsh.model.mesh.generate(3)
        log.info("gmsh volume mesh: %.0f s", time.time() - t0)
        ntags, coords, _ = gmsh.model.mesh.getNodes()
        _, etags = gmsh.model.mesh.getElementsByType(4)
    except Exception as err:  # gmsh raises plain Exceptions
        raise RemeshError(f"gmsh failed: {err}") from err
    finally:
        gmsh.finalize()
    coords = np.asarray(coords).reshape(-1, 3)
    lookup = np.zeros(int(np.max(ntags)) + 1, dtype=np.int64)
    lookup[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
    tets = lookup[np.asarray(etags, dtype=np.int64).reshape(-1, 4)]
    if len(tets) == 0:
        raise RemeshError("gmsh produced no tetrahedra")
    out_nodes, out_tets, _ = compact(coords, tets)
    return out_nodes, out_tets


def remesh_ct(source: Mesh, size_mm: float, labels: dict | None = None,
              angle_deg: float = 40.0, min_size_mm: float = 0.4,
              elements_through_thickness: float = 1.0, **kwargs) -> Mesh:
    """Re-mesh a labeled CT mesh with maximum edge length `size_mm` and relabel it.

    The size is reduced in walls thinner than `size_mm` (`thickness_sizes`),
    down to `min_size_mm`.
    """
    from craniofea.mesh.labeling import label_ct_mesh

    t0 = time.time()
    src_mm = source.nodes / MM
    sizes = thickness_sizes(src_mm, source.tets, source.elem_fields["depth"] / MM, size_mm,
                            min_size_mm, elements_through_thickness)
    nodes_mm, tets = remesh_surface(src_mm, source.surface_sets["all"], size_mm,
                                    angle_deg=angle_deg, size_min_factor=min_size_mm / size_mm,
                                    background=(src_mm, source.tets, sizes), **kwargs)
    del sizes, src_mm
    nodes = nodes_mm * MM
    tets = orient_tets(nodes, tets)
    faces, _ = boundary_faces(nodes, tets)
    mesh = Mesh(nodes=nodes, tets=tets, surface_sets={"all": faces}, frame=source.frame,
                meta={"source": "Open-Full-Jaw (gmsh remesh)", "target_size_mm": size_mm,
                      "min_size_mm": min_size_mm,
                      "remesh_seconds": None, "parent": source.meta.get("source_file"),
                      "frame": {"method": "inherited from source mesh"}})
    socket = transfer_surface_set(source, "socket", mesh, max_distance=0.5 * size_mm * MM)
    if len(socket) == 0:
        raise RemeshError("no socket faces could be transferred to the re-meshed surface")
    mesh.surface_sets["socket"] = socket
    label_ct_mesh(mesh, labels)
    mesh.meta["remesh_seconds"] = round(time.time() - t0, 1)
    log.info("remesh %.2f mm: %d nodes, %d tets in %.0f s", size_mm, mesh.n_nodes,
             mesh.n_elems, mesh.meta["remesh_seconds"])
    return mesh

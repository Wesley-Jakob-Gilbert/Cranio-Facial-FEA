"""Off-screen 3D rendering of element fields on the maxilla (PyVista/VTK).

Element (cell) values are shown on the outer surface, colouring each boundary
triangle with the value of the tetrahedron that owns it. Cameras are placed in
the anatomical frame of the mesh, so "inferior" always looks up into the
palatal vault with the incisors at the top of the image, whatever the
orientation of the source scan.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from craniofea.mesh.primitives import boundary_faces
from craniofea.model import AnatomicalFrame, Mesh

# view name -> (direction from the model toward the camera, image-up axis)
_VIEWS = {
    "inferior": ("-si", "ap"),
    "superior": ("si", "ap"),
    "left": ("-lr", "si"),
    "right": ("lr", "si"),
    "anterior": ("ap", "si"),
    "posterior": ("-ap", "si"),
}


def _axis(frame: AnatomicalFrame, spec: str) -> np.ndarray:
    sign = -1.0 if spec.startswith("-") else 1.0
    return sign * np.asarray(getattr(frame, spec.lstrip("-")), dtype=float)


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("3D rendering needs the 'viz' extra: pip install craniofea[viz]") from exc
    pv.OFF_SCREEN = True
    return pv


def surface_polydata(mesh: Mesh, elem_values: dict[str, np.ndarray] | None = None):
    """Boundary surface as ``pyvista.PolyData`` with owner-element values as cell data."""
    pv = _require_pyvista()
    faces, owner = boundary_faces(mesh.nodes, mesh.tets)
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).ravel()
    poly = pv.PolyData(mesh.nodes, cells)
    for name, values in (elem_values or {}).items():
        poly.cell_data[name] = np.asarray(values)[owner]
    return poly


def volume_grid(mesh: Mesh, elem_values: dict[str, np.ndarray] | None = None):
    """Full tetrahedral mesh as ``pyvista.UnstructuredGrid`` with element values."""
    pv = _require_pyvista()
    import vtk

    cells = np.hstack([np.full((mesh.n_elems, 1), 4), mesh.tets]).ravel()
    types = np.full(mesh.n_elems, vtk.VTK_TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, mesh.nodes)
    for name, values in (elem_values or {}).items():
        grid.cell_data[name] = np.asarray(values)
    return grid


def _frame(mesh: Mesh) -> AnatomicalFrame:
    if mesh.frame is not None:
        return mesh.frame
    c = mesh.nodes.mean(axis=0)
    return AnatomicalFrame(c, np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])


def place_camera(plotter, mesh: Mesh, view: str, zoom: float = 1.0) -> None:
    frame = _frame(mesh)
    toward, up = _VIEWS[view]
    center = 0.5 * (mesh.nodes.min(axis=0) + mesh.nodes.max(axis=0))
    radius = 0.5 * np.linalg.norm(mesh.nodes.max(axis=0) - mesh.nodes.min(axis=0))
    plotter.camera.focal_point = center
    plotter.camera.position = center + _axis(frame, toward) * 4.0 * radius
    plotter.camera.up = _axis(frame, up)
    plotter.camera.parallel_projection = True
    plotter.reset_camera()
    plotter.camera.zoom(zoom)


def render_surface_field(
    mesh: Mesh,
    values: np.ndarray,
    path: str | Path | None = None,
    *,
    view: str = "inferior",
    clim: tuple[float, float] | None = None,
    cmap: str = "BrBG",
    label: str = "",
    title: str | None = None,
    window_size: tuple[int, int] = (900, 760),
    zoom: float = 1.05,
    show_scalar_bar: bool = True,
) -> np.ndarray:
    """Render an element field on the outer surface; returns the RGB image array."""
    pv = _require_pyvista()
    poly = surface_polydata(mesh, {"field": values})
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    sbar = dict(title=label, color="black", fmt="%.2f", vertical=False,
                position_x=0.2, position_y=0.04, width=0.6, height=0.06,
                title_font_size=18, label_font_size=15, n_labels=5)
    plotter.add_mesh(poly, scalars="field", cmap=cmap, clim=clim, smooth_shading=False,
                     show_scalar_bar=show_scalar_bar, scalar_bar_args=sbar,
                     ambient=0.25, diffuse=0.75, specular=0.05)
    if title:
        plotter.add_text(title, position="upper_edge", font_size=13, color="black")
    place_camera(plotter, mesh, view, zoom)
    img = plotter.screenshot(None if path is None else str(path), return_img=True)
    plotter.close()
    return img


def render_section(
    mesh: Mesh,
    values: np.ndarray,
    path: str | Path | None = None,
    *,
    plane: str = "lr",
    offset: float = 0.0,
    clim: tuple[float, float] | None = None,
    cmap: str = "BrBG",
    label: str = "",
    title: str | None = None,
    window_size: tuple[int, int] = (900, 700),
) -> np.ndarray:
    """Render an element field on a planar cut normal to a frame axis.

    ``plane="lr"`` gives a sagittal section (normal = left-right axis),
    ``"ap"`` a coronal section, ``"si"`` a horizontal section. ``offset`` is
    measured along that axis from the frame origin, in metres.
    """
    pv = _require_pyvista()
    frame = _frame(mesh)
    normal = _axis(frame, plane)
    origin = np.asarray(frame.origin) + offset * normal
    grid = volume_grid(mesh, {"field": values})
    cut = grid.clip(normal=normal, origin=origin, crinkle=True)
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    sbar = dict(title=label, color="black", fmt="%.2f", vertical=False,
                position_x=0.2, position_y=0.04, width=0.6, height=0.06)
    plotter.add_mesh(cut, scalars="field", cmap=cmap, clim=clim, show_scalar_bar=bool(label),
                     scalar_bar_args=sbar)
    if title:
        plotter.add_text(title, position="upper_edge", font_size=13, color="black")
    view = {"lr": "left", "ap": "anterior", "si": "superior"}[plane]
    # Look at the cut face, i.e. from the removed half toward the kept half.
    place_camera(plotter, mesh, view if plane != "lr" else "right")
    img = plotter.screenshot(None if path is None else str(path), return_img=True)
    plotter.close()
    return img


def side_by_side(images: list[np.ndarray], pad: int = 8) -> np.ndarray:
    """Concatenate equally sized RGB images horizontally with white padding."""
    h = max(im.shape[0] for im in images)
    padded = []
    for i, im in enumerate(images):
        if im.shape[0] < h:
            im = np.vstack([im, np.full((h - im.shape[0],) + im.shape[1:], 255, im.dtype)])
        padded.append(im)
        if i < len(images) - 1:
            padded.append(np.full((h, pad, im.shape[2]), 255, im.dtype))
    return np.hstack(padded)


def write_gif(frames: list[np.ndarray], path: str | Path, fps: float = 2.0,
              hold_last: int = 3) -> None:
    """Write an animated GIF (looping), holding the final frame a little longer."""
    import imageio.v3 as iio

    frames = list(frames) + [frames[-1]] * hold_last
    iio.imwrite(path, np.stack(frames), duration=1000.0 / fps, loop=0)

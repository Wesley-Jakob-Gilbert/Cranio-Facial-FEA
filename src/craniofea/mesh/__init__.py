"""Mesh construction, import and anatomical labeling.

Entry points (see docs/MESH.md):

- `craniofea.mesh.labeling.build_ct_mesh(config)`: Open-Full-Jaw import +
  frame + canonical labels
- `craniofea.mesh.template.build_template_mesh(params, labels)`: parametric
  adolescent maxilla with the same labels
- `craniofea.mesh.remesh.remesh_ct(mesh, size_mm)`: gmsh re-mesh of the CT
  geometry for convergence studies
- `craniofea.mesh.frame.detect_frame(mesh)`: automatic anatomical frame
- `craniofea.mesh.quality.quality_report(mesh)`: element quality statistics
"""

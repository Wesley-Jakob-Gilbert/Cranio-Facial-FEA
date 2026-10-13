#!/usr/bin/env python
"""Build the labeled maxilla meshes, their quality report and figures.

    python scripts/build_meshes.py template          # parametric template (no data needed)
    python scripts/build_meshes.py ct                # Open-Full-Jaw patient 3 (needs data/)
    python scripts/build_meshes.py remesh --sizes 2.0 1.5 1.0
    python scripts/build_meshes.py quality           # markdown table of all built meshes
    python scripts/build_meshes.py figures           # docs/figures/mesh_*.png
    python scripts/build_meshes.py all

Outputs go to data/derived/ (git-ignored: CT-derived meshes inherit the
CC BY-NC-SA 4.0 license of Open-Full-Jaw and must not be committed).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from craniofea.mesh.labeling import build_ct_mesh, load_config, summary_table  # noqa: E402
from craniofea.mesh.quality import format_report, quality_report  # noqa: E402
from craniofea.mesh.template import build_template_from_config  # noqa: E402
from craniofea.model import Mesh  # noqa: E402

log = logging.getLogger("build_meshes")
DERIVED = ROOT / "data" / "derived"
FIGURES = ROOT / "docs" / "figures"


def cmd_template(args) -> Mesh:
    cfg = load_config(ROOT / args.template_config)
    mesh = build_template_from_config(cfg)
    out = ROOT / cfg.get("output", "data/derived/template_maxilla.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(out)
    print(summary_table(mesh))
    log.info("saved %s", out)
    return mesh


def cmd_ct(args) -> Mesh:
    mesh = build_ct_mesh(ROOT / args.ct_config, root=ROOT)
    print(summary_table(mesh))
    return mesh


def _ct_source(args) -> tuple[Mesh, dict]:
    cfg = load_config(ROOT / args.ct_config)
    path = ROOT / cfg.get("output", "data/derived/patient3_maxilla.npz")
    if not path.exists():
        log.info("CT mesh not built yet; building it first")
        return cmd_ct(args), cfg
    return Mesh.load(path), cfg


def cmd_remesh(args) -> list[Path]:
    from craniofea.mesh.remesh import RemeshError, remesh_ct

    src, cfg = _ct_source(args)
    rcfg = cfg.get("remesh", {})
    sizes = args.sizes or rcfg.get("sizes_mm", [2.0, 1.5, 1.0])
    pattern = rcfg.get("output", "data/derived/patient3_maxilla_h{size}.npz")
    written = []
    for h in sizes:
        try:
            mesh = remesh_ct(src, float(h), cfg.get("labels"),
                             angle_deg=float(rcfg.get("angle_deg", 40.0)))
        except RemeshError as err:
            log.error("remesh at %.2f mm failed: %s", h, err)
            continue
        out = ROOT / pattern.format(size=f"{float(h):.1f}")
        mesh.save(out)
        written.append(out)
        print(summary_table(mesh))
    return written


def _built_meshes() -> dict[str, Path]:
    found = {"template": DERIVED / "template_maxilla.npz",
             "CT (Open-Full-Jaw P3)": DERIVED / "patient3_maxilla.npz"}
    for p in sorted(DERIVED.glob("patient3_maxilla_h*.npz"), reverse=True):
        found[f"CT remesh h={p.stem.split('_h')[-1]} mm"] = p
    return {k: v for k, v in found.items() if v.exists()}


def cmd_quality(args) -> dict:
    rows = {name: quality_report(Mesh.load(p)) for name, p in _built_meshes().items()}
    if not rows:
        log.error("no meshes built yet")
        return {}
    print(format_report(rows))
    (DERIVED / "mesh_quality.json").write_text(json.dumps(rows, indent=2))
    return rows


def cmd_figures(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from craniofea.mesh.plotting import label_figure

    FIGURES.mkdir(parents=True, exist_ok=True)
    meshes = _built_meshes()
    jobs = [("template", "mesh_template_labels.png",
             "Parametric adolescent maxilla template: canonical labels", 0.0, 10.0),
            ("CT (Open-Full-Jaw P3)", "mesh_labels.png",
             "Open-Full-Jaw patient 3 maxilla (bone only): canonical labels", 0.0, 10.0)]
    for name, fname, title, ap, lr in jobs:
        if name not in meshes:
            log.warning("%s not built; skipping %s", name, fname)
            continue
        fig = label_figure(Mesh.load(meshes[name]), title, coronal_ap_mm=ap, sagittal_lr_mm=lr)
        fig.savefig(FIGURES / fname, dpi=args.dpi)
        log.info("wrote %s", FIGURES / fname)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["template", "ct", "remesh", "quality", "figures", "all"])
    ap.add_argument("--ct-config", default="configs/mesh_ct.yaml")
    ap.add_argument("--template-config", default="configs/mesh_template.yaml")
    ap.add_argument("--sizes", type=float, nargs="+", help="remesh target sizes in mm")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    steps = {"template": [cmd_template], "ct": [cmd_ct], "remesh": [cmd_remesh],
             "quality": [cmd_quality], "figures": [cmd_figures],
             "all": [cmd_template, cmd_ct, cmd_remesh, cmd_quality, cmd_figures]}
    for step in steps[args.what]:
        step(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

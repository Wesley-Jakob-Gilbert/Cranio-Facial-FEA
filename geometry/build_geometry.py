#!/usr/bin/env python3
"""Build a toy cranio-maxillary geometry placeholder.

MVP intent:
- represent simplified maxilla/palate domain
- define palate ROI for tongue pressure loading

This stub writes metadata only; geometry generation is next step.
"""
from pathlib import Path
import json

OUT = Path(__file__).resolve().parent
meta = {
    "model": "toy_maxilla_v0",
    "status": "stub",
    "notes": [
        "Use simplified block/curved palate approximation",
        "Tag palate ROI for pressure loads",
        "Keep geometry coarse for MVP",
    ],
}

(OUT / "geometry_meta.json").write_text(json.dumps(meta, indent=2))
print("Wrote geometry/geometry_meta.json")

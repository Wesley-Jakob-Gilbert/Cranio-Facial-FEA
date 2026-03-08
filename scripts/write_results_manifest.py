#!/usr/bin/env python3
"""Write a reproducibility manifest for key result artifacts."""
from __future__ import annotations

from pathlib import Path
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "results" / "RESULTS_MANIFEST.md"

INCLUDE_EXT = {".csv", ".png", ".json", ".inp", ".dat", ".cvg", ".sta", ".frd", ".12d"}
EXCLUDE_NAMES = {"spooles.out"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


files = []
for p in sorted(RESULTS.rglob("*")):
    if not p.is_file():
        continue
    if p.name in EXCLUDE_NAMES:
        continue
    if p.suffix.lower() not in INCLUDE_EXT:
        continue
    rel = p.relative_to(ROOT)
    files.append((rel, p.stat().st_size, sha256_file(p)))

lines = []
lines.append("# Results Manifest\n")
lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
lines.append("\n")
lines.append("This manifest captures hashes for reproducibility/integrity checks.\n")
lines.append("\n")
lines.append("| File | Bytes | SHA256 |\n")
lines.append("|---|---:|---|\n")
for rel, size, digest in files:
    lines.append(f"| `{rel}` | {size} | `{digest}` |\n")

OUT.write_text("".join(lines), encoding="utf-8")
print(f"Wrote {OUT} ({len(files)} files)")

#!/usr/bin/env python3
"""Extract nodal displacement, element von Mises, and element SED from CalculiX .dat.

Outputs per case:
- results/<case>/node_u.csv
- results/<case>/elem_vm.csv
- results/<case>/elem_sed.csv  (strain energy density from ENER output)
"""
from __future__ import annotations
from pathlib import Path
import csv
import math
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOADS = yaml.safe_load((ROOT / "configs" / "loads.yaml").read_text())["load_cases"]
CASES = list(LOADS.keys())

node_re = re.compile(r"^\s*(\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)")
stress_re = re.compile(
    r"^\s*(\d+)\s+\d+\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)\s+([+-]?\d\.\d+E[+-]\d+)"
)
# ENER line: elem_id  integ_point  energy_density_value
ener_re = re.compile(
    r"^\s*(\d+)\s+\d+\s+([+-]?\d\.\d+E[+-]\d+)"
)


def mises(sxx, syy, szz, sxy, sxz, syz):
    return math.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )


for case in CASES:
    dat = ROOT / "results" / case / f"{case}.dat"
    if not dat.exists() or dat.stat().st_size == 0:
        print(f"Skip {case}: missing dat")
        continue

    nodes = {}
    elem_vm_acc = {}
    elem_sed_acc = {}

    in_disp = False
    in_stress = False
    in_ener = False

    for line in dat.read_text(encoding="utf-8", errors="ignore").splitlines():
        ls = line.strip().lower()
        if ls.startswith("displacements"):
            in_disp = True
            in_stress = False
            in_ener = False
            continue
        if ls.startswith("stresses"):
            in_stress = True
            in_disp = False
            in_ener = False
            continue
        if ls.startswith("internal energy density"):
            in_ener = True
            in_disp = False
            in_stress = False
            continue
        if not ls:
            continue

        if in_disp:
            m = node_re.match(line)
            if m:
                nid = int(m.group(1))
                ux, uy, uz = map(float, m.groups()[1:])
                umag = math.sqrt(ux * ux + uy * uy + uz * uz)
                nodes[nid] = (ux, uy, uz, umag)

        elif in_stress:
            s = stress_re.match(line)
            if s:
                eid = int(s.group(1))
                sxx, syy, szz, sxy, sxz, syz = map(float, s.groups()[1:])
                vm = mises(sxx, syy, szz, sxy, sxz, syz)
                acc = elem_vm_acc.setdefault(eid, [0.0, 0])
                acc[0] += vm
                acc[1] += 1

        elif in_ener:
            e = ener_re.match(line)
            if e:
                eid = int(e.group(1))
                sed_val = float(e.group(2))
                acc = elem_sed_acc.setdefault(eid, [0.0, 0])
                acc[0] += sed_val
                acc[1] += 1

    out_nodes = ROOT / "results" / case / "node_u.csv"
    with out_nodes.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "ux", "uy", "uz", "u_mag"])
        for nid in sorted(nodes):
            ux, uy, uz, umag = nodes[nid]
            w.writerow([nid, ux, uy, uz, umag])

    out_elem = ROOT / "results" / case / "elem_vm.csv"
    with out_elem.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "vm_avg_pa"])
        for eid in sorted(elem_vm_acc):
            ssum, cnt = elem_vm_acc[eid]
            w.writerow([eid, ssum / cnt if cnt else 0.0])

    out_sed = ROOT / "results" / case / "elem_sed.csv"
    with out_sed.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "sed_avg_pa"])
        for eid in sorted(elem_sed_acc):
            ssum, cnt = elem_sed_acc[eid]
            w.writerow([eid, ssum / cnt if cnt else 0.0])

    sed_info = f" + {out_sed}" if elem_sed_acc else ""
    print(f"Wrote {out_nodes} and {out_elem}{sed_info}")

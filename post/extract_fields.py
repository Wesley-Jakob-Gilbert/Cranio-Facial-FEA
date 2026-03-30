#!/usr/bin/env python3
"""Extract nodal displacement, element von Mises, SED, principal stresses, and full
stress tensor from CalculiX .dat output.

Outputs per case:
- results/<case>/node_u.csv
- results/<case>/elem_vm.csv
- results/<case>/elem_sed.csv            (CalculiX ENER; preferred)
- results/<case>/elem_stress_tensor.csv  (full 6-component tensor avg per element)
- results/<case>/elem_principal_stress.csv  (max/mid/min principal stresses)

The full stress tensor and principal stress outputs are used by bone_remodeling.py
to compute:
  - Fix 5: proper isotropic SED fallback: psi = (s_ij C^-1 s_ij)/2 (correct vs vm^2/2E)
  - Fix 3: tension-only suture softening via max principal stress (not Von Mises)
"""
from __future__ import annotations
from pathlib import Path
import csv
import math
import re
import yaml

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

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


def principal_stresses_numpy(sxx, syy, szz, sxy, sxz, syz):
    """Return (s1, s2, s3) sorted descending (max first) via numpy eigenvalues.

    Sign convention: positive = tension, negative = compression.
    """
    S = [[sxx, sxy, sxz],
         [sxy, syy, syz],
         [sxz, syz, szz]]
    eigs = np.linalg.eigvalsh(S)  # symmetric → real eigenvalues, ascending
    s3, s2, s1 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    return s1, s2, s3  # max, mid, min


def principal_stresses_analytic(sxx, syy, szz, sxy, sxz, syz):
    """Analytic 3×3 symmetric eigenvalue solve via characteristic polynomial.

    Returns (s1, s2, s3) with s1 >= s2 >= s3.
    Uses the standard cubic formula for a real symmetric matrix (Kopp 2008 style).
    Fallback when numpy unavailable.
    """
    # Invariants
    I1 = sxx + syy + szz
    I2 = (sxx*syy + syy*szz + szz*sxx) - (sxy**2 + sxz**2 + syz**2)
    I3 = (sxx*(syy*szz - syz**2)
          - sxy*(sxy*szz - syz*sxz)
          + sxz*(sxy*syz - syy*sxz))

    p = I1/3.0
    q = I1**2/9.0 - I2/3.0
    r = (I1**3/27.0 - I1*I2/6.0 + I3/2.0)

    # Clamp for numerical safety
    disc = max(0.0, q**3 - r**2)
    phi = math.atan2(math.sqrt(disc), r) / 3.0
    sq_q = math.sqrt(max(0.0, q))

    s1 = p + 2.0*sq_q*math.cos(phi)
    s2 = p - 2.0*sq_q*math.cos(math.pi/3.0 + phi)
    s3 = p - 2.0*sq_q*math.cos(math.pi/3.0 - phi)

    return tuple(sorted([s1, s2, s3], reverse=True))


def compute_principals(sxx, syy, szz, sxy, sxz, syz):
    if _HAS_NUMPY:
        return principal_stresses_numpy(sxx, syy, szz, sxy, sxz, syz)
    return principal_stresses_analytic(sxx, syy, szz, sxy, sxz, syz)


for case in CASES:
    dat = ROOT / "results" / case / f"{case}.dat"
    if not dat.exists() or dat.stat().st_size == 0:
        print(f"Skip {case}: missing dat")
        continue

    nodes = {}
    # Accumulators: elem_id → [sum, count]
    elem_vm_acc: dict[int, list] = {}
    elem_sed_acc: dict[int, list] = {}
    # Full stress tensor per element: [sxx, syy, szz, sxy, sxz, syz, count]
    elem_stress_acc: dict[int, list] = {}
    # Principal stresses: computed post-loop from averaged tensor (audit_3 fix)
    # elem_principal_acc removed — principals now derived from elem_stress_acc average.

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

                # Von Mises accumulation
                vm = mises(sxx, syy, szz, sxy, sxz, syz)
                acc = elem_vm_acc.setdefault(eid, [0.0, 0])
                acc[0] += vm
                acc[1] += 1

                # Full stress tensor accumulation (for isotropic SED fallback)
                if eid not in elem_stress_acc:
                    elem_stress_acc[eid] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]
                t = elem_stress_acc[eid]
                t[0] += sxx; t[1] += syy; t[2] += szz
                t[3] += sxy; t[4] += sxz; t[5] += syz
                t[6] += 1

                # (Principal stresses computed after loop from averaged tensor — see Fix below)

        elif in_ener:
            e = ener_re.match(line)
            if e:
                eid = int(e.group(1))
                sed_val = float(e.group(2))
                acc = elem_sed_acc.setdefault(eid, [0.0, 0])
                acc[0] += sed_val
                acc[1] += 1

    out_dir = ROOT / "results" / case
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- node_u.csv ---
    out_nodes = out_dir / "node_u.csv"
    with out_nodes.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "ux", "uy", "uz", "u_mag"])
        for nid in sorted(nodes):
            ux, uy, uz, umag = nodes[nid]
            w.writerow([nid, ux, uy, uz, umag])

    # --- elem_vm.csv ---
    out_elem = out_dir / "elem_vm.csv"
    with out_elem.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "vm_avg_pa"])
        for eid in sorted(elem_vm_acc):
            ssum, cnt = elem_vm_acc[eid]
            w.writerow([eid, ssum / cnt if cnt else 0.0])

    # --- elem_sed.csv (from CalculiX ENER; preferred) ---
    out_sed = out_dir / "elem_sed.csv"
    with out_sed.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "sed_avg_pa"])
        for eid in sorted(elem_sed_acc):
            ssum, cnt = elem_sed_acc[eid]
            w.writerow([eid, ssum / cnt if cnt else 0.0])

    # --- elem_stress_tensor.csv (Fix 5: proper isotropic SED fallback) ---
    # Contains the average stress tensor per element for downstream SED computation.
    # Allows bone_remodeling.py to compute psi_iso = (sij * Cijkl^-1 * skl)/2
    # correctly rather than using the incorrect vm^2/2E proxy.
    out_tensor = out_dir / "elem_stress_tensor.csv"
    with out_tensor.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "sxx_pa", "syy_pa", "szz_pa", "sxy_pa", "sxz_pa", "syz_pa"])
        for eid in sorted(elem_stress_acc):
            t = elem_stress_acc[eid]
            cnt = t[6]
            if cnt > 0:
                w.writerow([eid] + [t[i]/cnt for i in range(6)])

    # --- elem_principal_stress.csv (Fix 3: tension-only suture softening) ---
    # s1 = max principal (positive → tension), s3 = min principal (negative → compression).
    # Suture softening should be triggered by s1 (tensile opening), not Von Mises.
    #
    # Audit_3 Fix 1: Compute principal stresses from the AVERAGED stress tensor, not
    # by averaging per-integration-point principals. Averaging principals from different
    # orientations is mathematically incorrect: |max(s1_i)| >= s1(avg_tensor) always,
    # causing systematic overestimation of tension and spurious suture over-softening.
    # Correct order: (1) average tensor across IPs, (2) compute principals once.
    out_principal = out_dir / "elem_principal_stress.csv"
    with out_principal.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elem_id", "s1_max_pa", "s2_mid_pa", "s3_min_pa"])
        for eid in sorted(elem_stress_acc):
            t = elem_stress_acc[eid]
            cnt = t[6]
            if cnt > 0:
                # Average tensor components first (correct approach)
                sxx_avg = t[0]/cnt; syy_avg = t[1]/cnt; szz_avg = t[2]/cnt
                sxy_avg = t[3]/cnt; sxz_avg = t[4]/cnt; syz_avg = t[5]/cnt
                # Compute principals from averaged tensor
                s1, s2, s3 = compute_principals(sxx_avg, syy_avg, szz_avg,
                                                 sxy_avg, sxz_avg, syz_avg)
                w.writerow([eid, s1, s2, s3])

    sed_info = f" + {out_sed.name}" if elem_sed_acc else " (no ENER — tensor fallback available)"
    print(f"Wrote {out_nodes.name}, {out_elem.name}{sed_info}, "
          f"{out_tensor.name}, {out_principal.name}  [{case}]")

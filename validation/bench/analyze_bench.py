"""Pre-registered analysis of the pig-palate bench test (protocol section 7).

Inputs
  measured.csv  : specimen,gauge_id,trial,step,pressure_kPa,force_N,strain_ue,temp_C
                  (one row per grid per load step, baseline-corrected per trial:
                   strain_ue = mean(plateau) - mean(pre-load zero) with linear drift
                   correction between pre- and post-trial zeros)
  predicted.csv : specimen,gauge_id,model,pred_ue_per_kPa   (frozen BEFORE unblinding)

Steps
  1. Per grid and trial: OLS of strain on cup pressure -> measured sensitivity
     S_meas (ue/kPa), linearity R^2, intercept, hysteresis (from load/unload pairs).
     Average the trials per grid; grids failing QC (linear-fit residual rms >
     max(2.5 ue, 3 % of peak strain), zero shift > 5 ue, or shunt-cal error > 2 %) are
     excluded BEFORE comparison, with the reason logged.  (R^2 is not used for QC because
     it penalises small but valid signals.)
  2. Agreement S_meas vs S_pred, pooled over all grids/specimens, per model:
     OLS slope/intercept with specimen-cluster bootstrap 95 % CI, R^2, RMSE, NRMSE
     (RMSE / max|S_meas|), Bland-Altman bias and limits on log ratio,
     fraction of grids within +/-30 % (or +/-0.03 ue/kPa if |S| small),
     slope by specimen.
  3. Acceptance (pre-registered thresholds in ACCEPT below).

Usage
  python validation/bench/analyze_bench.py measured.csv predicted.csv [--model M1]
  python validation/bench/analyze_bench.py --demo         (synthetic data, shows output)
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ACCEPT = {                       # frozen in docs/validation/BENCH_TEST_PROTOCOL.md sec. 1
    "r2_min": 0.80,              # pooled S_meas vs S_pred
    "slope_ci_within": (0.75, 1.33),   # 95 % CI of slope (measured on predicted) inside
    "slope_point_within": (0.80, 1.25),
    "intercept_abs_max_frac": 0.10,    # |intercept| <= 10 % of max |S_meas|
    "nrmse_max": 0.20,
    "frac_within_30pct_min": 0.70,
    "specimens_slope_ok_min": 3,       # of 4 specimens with slope in (0.67, 1.5)
    "angle_err_max_deg": 15.0,         # rosettes with |e1| > 2|e2| (secondary)
}
FALSIFY_SLOPE = (0.5, 2.0)          # outside -> magnitude prediction rejected


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def per_grid_sensitivity(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["specimen"], r["gauge_id"], r["trial"])].append(
            (float(r["pressure_kPa"]), float(r["strain_ue"])))
    trials = defaultdict(list)
    for (s, g, t), v in by.items():
        p, e = np.array(v).T
        A = np.column_stack([p, np.ones_like(p)])
        (b, a), *_ = np.linalg.lstsq(A, e, rcond=None)
        pred = A @ [b, a]
        r2 = 1 - ((e - pred) ** 2).sum() / max(((e - e.mean()) ** 2).sum(), 1e-30)
        rms = float(np.sqrt(np.mean((e - pred) ** 2)))
        trials[(s, g)].append({"slope": b, "icpt": a, "r2": r2, "rms": rms,
                               "emax": float(np.max(np.abs(e)))})
    out = {}
    for k, ts in trials.items():
        sl = np.array([t["slope"] for t in ts])
        out[k] = {"S_meas": float(sl.mean()), "S_cv": float(sl.std(ddof=1) / abs(sl.mean()))
                  if len(sl) > 1 else float("nan"),
                  "r2_lin_min": float(min(t["r2"] for t in ts)),
                  "icpt_max_ue": float(max(abs(t["icpt"]) for t in ts)), "n_trials": len(ts),
                  # linearity QC: fit residual must be noise-like, not a fraction of signal
                  "lin_ok": all(t["rms"] <= max(2.5, 0.03 * t["emax"]) for t in ts),
                  "rms_max_ue": float(max(t["rms"] for t in ts))}
    return out


def agreement(x, y, spec_ids, n_boot=4000, seed=1):
    """y = measured, x = predicted."""
    x, y = np.asarray(x), np.asarray(y)
    A = np.column_stack([x, np.ones_like(x)])
    (b, a), *_ = np.linalg.lstsq(A, y, rcond=None)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    rng = np.random.default_rng(seed)
    specs = np.unique(spec_ids)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(specs, len(specs), replace=True)
        idx = np.concatenate([np.flatnonzero(spec_ids == s) for s in pick])
        if np.ptp(x[idx]) == 0:
            continue
        Ab = np.column_stack([x[idx], np.ones(len(idx))])
        boots.append(np.linalg.lstsq(Ab, y[idx], rcond=None)[0][0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    ratio = np.log(np.abs(y) / np.maximum(np.abs(x), 1e-12))
    ok = (np.abs(y - x) <= 0.30 * np.abs(x)) | (np.abs(y - x) <= 0.03)
    per_spec = {}
    for s in specs:
        m = spec_ids == s
        if m.sum() >= 3:
            per_spec[str(s)] = float(np.linalg.lstsq(np.column_stack([x[m], np.ones(m.sum())]),
                                                     y[m], rcond=None)[0][0])
    return {"n": int(len(x)), "slope": float(b), "slope_ci95": [float(lo), float(hi)],
            "intercept": float(a), "r2": float(r2), "rmse": rmse,
            "nrmse": rmse / float(np.max(np.abs(y))), "max_abs_meas": float(np.max(np.abs(y))),
            "bland_altman_log_ratio": {"bias": float(ratio.mean()),
                                        "loa": [float(ratio.mean() - 1.96 * ratio.std(ddof=1)),
                                                float(ratio.mean() + 1.96 * ratio.std(ddof=1))]},
            "frac_within_30pct": float(ok.mean()), "slope_by_specimen": per_spec}


def verdict(ag):
    c = {
        "r2": ag["r2"] >= ACCEPT["r2_min"],
        "slope_ci": ACCEPT["slope_ci_within"][0] <= ag["slope_ci95"][0]
        and ag["slope_ci95"][1] <= ACCEPT["slope_ci_within"][1],
        "slope_point": ACCEPT["slope_point_within"][0] <= ag["slope"] <= ACCEPT["slope_point_within"][1],
        "intercept": abs(ag["intercept"]) <= ACCEPT["intercept_abs_max_frac"] * ag["max_abs_meas"],
        "nrmse": ag["nrmse"] <= ACCEPT["nrmse_max"],
        "within30": ag["frac_within_30pct"] >= ACCEPT["frac_within_30pct_min"],
        "specimens": sum(0.67 <= v <= 1.5 for v in ag["slope_by_specimen"].values())
        >= ACCEPT["specimens_slope_ok_min"],
    }
    falsified = not (FALSIFY_SLOPE[0] <= ag["slope"] <= FALSIFY_SLOPE[1])
    return {"criteria": c, "all_pass": all(c.values()), "magnitude_falsified": falsified}


def main(measured, predicted, model):
    sens = per_grid_sensitivity(read_csv(measured))
    pred = {(r["specimen"], r["gauge_id"]): float(r["pred_ue_per_kPa"])
            for r in read_csv(predicted) if r["model"] == model}
    excluded, X, Y, S = [], [], [], []
    for k, v in sorted(sens.items()):
        if k not in pred:
            excluded.append((k, "no prediction"))
            continue
        if not v["lin_ok"]:
            excluded.append((k, f"nonlinear: residual rms {v['rms_max_ue']:.2f} ue"))
            continue
        X.append(pred[k]); Y.append(v["S_meas"]); S.append(k[0])
    ag = agreement(X, Y, np.array(S))
    res = {"model": model, "agreement": ag, "verdict": verdict(ag),
           "excluded": [f"{a}/{b}: {r}" for (a, b), r in excluded], "accept": ACCEPT}
    print(json.dumps(res, indent=1))
    return res


def demo():
    """Synthetic 4-specimen data set with a 15 % stiffness error and 1.5 ue noise."""
    rng = np.random.default_rng(0)
    work = Path(__file__).with_name("demo")
    work.mkdir(exist_ok=True)
    mrows, prows = [], []
    for s in ["S1", "S2", "S3", "S4"]:
        k_spec = rng.normal(1.15, 0.08)
        for gi in range(9):
            g = f"G{gi + 1}"
            S_pred = rng.choice([-1, 1]) * rng.uniform(0.2, 1.6)   # ue/kPa
            prows.append({"specimen": s, "gauge_id": g, "model": "M1", "pred_ue_per_kPa": S_pred})
            S_true = S_pred * k_spec * rng.normal(1, 0.12)
            for t in range(3):
                for step, p in enumerate([0, 10, 20, 35, 50, 70, 50, 35, 20, 10, 0]):
                    mrows.append({"specimen": s, "gauge_id": g, "trial": t, "step": step,
                                  "pressure_kPa": p, "force_N": p * 1.0,
                                  "strain_ue": S_true * p + rng.normal(0, 1.5), "temp_C": 22.0})
    for name, rows in (("measured.csv", mrows), ("predicted.csv", prows)):
        with open(work / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    res = main(work / "measured.csv", work / "predicted.csv", "M1")
    (work / "demo_result.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("measured", nargs="?")
    ap.add_argument("predicted", nargs="?")
    ap.add_argument("--model", default="M1")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    demo() if a.demo else main(a.measured, a.predicted, a.model)

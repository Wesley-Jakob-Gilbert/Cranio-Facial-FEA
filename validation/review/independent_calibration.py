"""Independent scalar checks; standard library only, NO craniofea imports.

Pressure-proportional strains assume identical load patterns and fixed stiffness.
The arbitrary modulus cancels in all baseline ratios and thresholds.
"""
import json
import math
from pathlib import Path


def stimulus(strains, counts, exponent):
    return math.fsum(n * abs(e) ** exponent for e, n in zip(strains, counts)) ** (1 / exponent)


def threshold(a_ref, b_ref, a_habit, b_habit, exponent, width):
    k = (1 + width) ** exponent
    num, den = k * a_ref - a_habit, b_habit - k * b_ref
    if num < 0:
        return 0.0
    return num / den if den > 0 else math.inf


def bisect_crossing(ratio, target, lo=0.0, hi=1e15):
    assert ratio(lo) < target < ratio(hi)
    for _ in range(180):
        mid = (lo + hi) / 2
        if ratio(mid) > target:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def main():
    modulus = 15e9
    p_swallow, p_rest, p_ref = 15000.0, 2000.0, 200.0
    n_swallow, width, exponent = 600.0, 0.1, 4
    e_sw, e_rest, e_ref = [p / modulus for p in (p_swallow, p_rest, p_ref)]
    # Energy units J/m^3 = Pa; E in Pa -> dimensionless strain.
    sed = p_swallow ** 2 / (2 * modulus)
    assert math.isclose(math.sqrt(2 * sed / modulus), e_sw)
    result = {"assumptions": "swallowing only; identical load patterns; no gravity; SI",
              "sed_Pa": sed, "effective_strain": e_sw}
    a = n_swallow * e_sw ** exponent
    br, bh = e_ref ** exponent, e_rest ** exponent
    ratio = lambda n: ((a + n * bh) / (a + n * br)) ** (1 / exponent)
    exact = threshold(a, br, a, bh, exponent, width)
    numeric = bisect_crossing(ratio, 1 + width)
    assert math.isclose(exact, numeric, rel_tol=1e-12)
    result["threshold"] = {
        "ignoring_reference": ((1 + width) ** exponent - 1) * n_swallow
                             * (p_swallow / p_rest) ** exponent,
        "with_reference": exact, "bisection": numeric,
        "threshold_in_1Hz_days": exact / 86400,
        "error_at_n1": ratio(1) - 1, "error_at_n86400": ratio(86400) - 1,
        "ratio_at_threshold": ratio(exact),
        "m2": threshold(n_swallow * e_sw**2, e_ref**2,
                        n_swallow * e_sw**2, e_rest**2, 2, width),
        "m8": threshold(n_swallow * e_sw**8, e_ref**8,
                        n_swallow * e_sw**8, e_rest**8, 8, width),
        "w_zero": threshold(a, br, a, bh, exponent, 0),
        "chewing_equals_49_times_swallow_term": threshold(50*a, br, 50*a, bh, exponent, width),
    }
    # All samples are internally homeostatic under their OWN reference history.
    psi_star = stimulus([e_sw, e_ref], [n_swallow, 1], exponent)
    assert stimulus([e_sw, e_ref], [n_swallow, 1], exponent) / psi_star - 1 == 0
    result["calibration"] = {
        "psi_star": psi_star, "reference_error": 0.0,
        "rate_cortical_per_day": 0.04 / 365,
        "rate_cancellous_per_day": 0.28 / 365,
        "monthly_dt_days": 365 / 12,
        "gf_per_cm2_in_Pa": 0.001 * 9.80665 / 0.0001,
        "minutes_for_1500_chews_at_1p57Hz": 1500 / 1.57 / 60,
        "chews_for_35p3min_at_1p57Hz": 35.3 * 60 * 1.57,
        "two_kPa_fraction_of_max_range": [2 / 41, 2 / 10],
        "mouth_breathing_at_1e7_error": (a / (a + 1e7 * br)) ** 0.25 - 1,
        "mouth_breathing_resorption_threshold": (
            a * (1 - (1-width)**exponent) / ((1-width)**exponent * br)),
    }
    # Define a separate threshold example with unequal cyclic stimuli and gravity.
    ar, ah, br2, bh2 = 2.0, 1.7, 0.01, 0.2
    general = threshold(ar, br2, ah, bh2, 4, 0.1)
    assert math.isclose(((ah + general*bh2)/(ar + general*br2))**.25, 1.1)
    assert math.isinf(threshold(1, 1, 1, 1, 4, .1))
    assert threshold(1, 1, 2, 1, 4, .1) == 0
    result["general_threshold_test"] = general
    # Spatial smoothing: raw onset at n=1 is NOT smoothed onset.
    # First element S=(1+n)^.25-1; second S=0; equal averaging.
    raw = (1.1**4 - 1)
    smoothed = (1.2**4 - 1)
    result["smoothing_example"] = {"raw_threshold": raw,
                                  "equal_average_threshold": smoothed,
                                  "threshold_multiplier": smoothed/raw}
    # A power-sum share is not a share of the root or an average of local shares.
    result["98pct_power_share"] = {
        "remaining_stimulus_ratio_if_chew_removed": .02**.25,
        "remaining_stimulus_ratio_if_other_removed": .98**.25,
    }
    path = Path(__file__).with_name("independent_calibration.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print("PASS: independent energy, threshold/bisection, homeostasis and edge assertions")


if __name__ == "__main__":
    main()

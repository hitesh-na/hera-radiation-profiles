"""Conservative candidate screening, not a claim of physical identification."""
import numpy as np
from scipy.signal import find_peaks
from .profiles import smooth_supported

DEFAULT_RULES = {"smoothing_bins": 3, "min_side_bins": 3,
                 "min_side_span_m": 1500.0, "min_drop_cps": 0.5,
                 "min_drop_fraction": 0.10, "competing_height_fraction": 0.08,
                 "min_competing_separation_m": 1500.0, "max_plateau_m": 2000.0,
                 "counting_sigma_multiplier": 2.0}


def screen_peak(profile, column="rate_cps", rules=None):
    r = {**DEFAULT_RULES, **(rules or {})}
    p = profile.reset_index(drop=True)
    s = smooth_supported(p, column, int(r["smoothing_bins"]))
    valid = np.flatnonzero(s.notna())
    result = {"status": "unresolved", "reason": "insufficient_supported_bins",
              "peak_altitude_m": None, "peak_altitude_km": None, "peak_cps": None,
              "candidate_altitude_m": None, "bin_m": float(p.bin_m.iloc[0]),
              "coverage_min_m": None, "coverage_max_m": None,
              "supported_bins": int(len(valid)), "smoothing_bins": int(r["smoothing_bins"]),
              "counting_uncertainty_available": bool(p.counting_se_cps.notna().any()) if column == "rate_cps" else False,
              "screening_error_basis": "observed_counting_SE" if column == "rate_cps" else "predicted_rate_Poisson_proxy_if_exposure_available",
              "interpretation": "screened candidate in observed coverage; requires physical and detector review"}
    if not len(valid):
        return result
    result["coverage_min_m"] = float(p.loc[valid, "observed_alt_min_m"].min())
    result["coverage_max_m"] = float(p.loc[valid, "observed_alt_max_m"].max())
    if len(valid) < 2 * r["min_side_bins"] + 1:
        return result
    # No interpolation over absent / too-sparse bins. Unknown regions can hide a peak.
    if np.any(np.diff(valid) != 1):
        result["reason"] = "gap_in_supported_altitude_coverage"
        return result
    y = s.iloc[valid].to_numpy(float)
    h = p.altitude_m.iloc[valid].to_numpy(float)
    span = float(np.ptp(y))
    if span < max(r["min_drop_cps"], r["min_drop_fraction"] * float(np.max(y))):
        result["reason"] = "flat_or_insufficient_prominence"
        return result
    near_top = np.flatnonzero(np.isclose(y, y.max(), atol=1e-10, rtol=1e-7))
    i = int(near_top[len(near_top) // 2])
    result["candidate_altitude_m"] = float(h[i])
    if near_top[0] == 0 or near_top[-1] == len(y) - 1:
        result.update(status="edge_limited", reason="maximum_touches_coverage_boundary")
        return result
    if h[near_top[-1]] - h[near_top[0]] > r["max_plateau_m"]:
        result.update(status="ambiguous", reason="broad_or_disconnected_top")
        return result
    k = max(int(r["min_side_bins"]), int(np.ceil(r["min_side_span_m"] / result["bin_m"])))
    if i < k or len(y) - 1 - i < k:
        result.update(status="edge_limited", reason="insufficient_altitude_on_both_sides")
        return result
    prominence = max(r["min_drop_cps"], r["min_drop_fraction"] * float(y[i]))
    candidates, _ = find_peaks(y, prominence=prominence)
    competitors = [int(j) for j in candidates if abs(h[j]-h[i]) >= r["min_competing_separation_m"]
                   and y[j] >= y[i] * (1-r["competing_height_fraction"])]
    if competitors:
        result.update(status="ambiguous", reason="competing_prominent_maxima")
        return result
    # Three bins near each observed edge form broad shoulders. Both must be
    # substantially below the candidate, avoiding a single low noisy neighbor.
    nside = int(r["min_side_bins"])
    left = np.arange(0, nside)
    right = np.arange(len(y)-nside, len(y))
    for flank in [left, right]:
        threshold = prominence
        if column == "rate_cps":
            errors = p.counting_se_cps.iloc[valid].to_numpy(float)
        elif "live_time_s" in p:
            errors = np.sqrt(p[column].clip(lower=0) / p.live_time_s.where(p.live_time_s > 0)).iloc[valid].to_numpy(float)
        else:
            errors = np.full(len(valid), np.nan)
        if np.isfinite(errors[i]) and np.isfinite(errors[flank]).all():
            # Conservative: use unsmoothed peak SE despite smoothing. This is a
            # screening heuristic, NOT a calibrated hypothesis test or peak CI.
            threshold = max(threshold, r["counting_sigma_multiplier"] *
                            np.sqrt(errors[i]**2 + np.sum(errors[flank]**2)/len(flank)**2))
        if y[i] - np.mean(y[flank]) < threshold:
            result["reason"] = "no_meaningful_drop_on_both_sides"
            return result
    result.update(status="identified", reason="interior_prominent_bracketed_candidate",
                  peak_altitude_m=float(h[i]), peak_altitude_km=float(h[i]/1000), peak_cps=float(y[i]))
    return result


def compare_peaks(measured, predicted):
    both = measured["status"] == predicted["status"] == "identified"
    signed = predicted["peak_altitude_m"] - measured["peak_altitude_m"] if both else np.nan
    return {"both_identified": both, "peak_error_m": signed, "abs_peak_error_m": abs(signed),
            "measured_status": measured["status"], "predicted_status": predicted["status"],
            "measured_peak_m": measured["peak_altitude_m"], "predicted_peak_m": predicted["peak_altitude_m"]}

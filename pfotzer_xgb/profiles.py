"""Measured and predicted profiles on one absolute-altitude grid."""
import numpy as np
import pandas as pd
from scipy.stats import chi2


def altitude_profile(df, bin_m=250, min_samples=3, prediction_col=None):
    if bin_m <= 0 or min_samples < 1 or df.empty:
        raise ValueError("Profile needs rows, a positive bin width and minimum sample count")
    if df.flight_id.nunique() != 1 or df.phase.nunique() != 1:
        raise ValueError("Construct profiles separately for each flight AND phase")
    d = df.copy()
    d["bin_index"] = np.floor(d["Altitude[m]"] / bin_m).astype(int)
    rows = []
    for index, g in d.groupby("bin_index", sort=True):
        exposure = "Live_Time_s" in g and g.Live_Time_s.notna().all() and (g.Live_Time_s > 0).all()
        counts = exposure and "Raw_Counts" in g and g.Raw_Counts.notna().all()
        weights = g.Live_Time_s.to_numpy(float) if exposure else np.ones(len(g))
        row = {"bin_index": index, "n_samples": len(g), "live_time_s": weights.sum() if exposure else np.nan,
               "raw_counts": g.Raw_Counts.sum() if counts else np.nan,
               "observed_alt_min_m": g["Altitude[m]"].min(), "observed_alt_max_m": g["Altitude[m]"].max(),
               "rate_cps": np.average(g.Counts_Per_Second, weights=weights),
               "aggregation": "sum_counts/sum_live_time" if counts else ("exposure_weighted_CPS" if exposure else "arithmetic_mean_CPS"),
               "counting_se_cps": np.nan, "counting_95_low_cps": np.nan, "counting_95_high_cps": np.nan}
        if counts:
            n, live = float(g.Raw_Counts.sum()), float(g.Live_Time_s.sum())
            row["rate_cps"] = n / live
            row["counting_se_cps"] = np.sqrt(n) / live
            row["counting_95_low_cps"] = 0.0 if n == 0 else chi2.ppf(.025, 2*n) / (2*live)
            row["counting_95_high_cps"] = chi2.ppf(.975, 2*(n+1)) / (2*live)
        if prediction_col is not None:
            row["predicted_cps"] = np.average(g[prediction_col], weights=weights)
        rows.append(row)
    p = pd.DataFrame(rows).set_index("bin_index")
    grid = pd.RangeIndex(d.bin_index.min(), d.bin_index.max() + 1, name="bin_index")
    p = p.reindex(grid).reset_index()
    p["n_samples"] = p.n_samples.fillna(0).astype(int)
    p["supported"] = p.n_samples >= min_samples
    p["altitude_m"] = (p.bin_index + .5) * bin_m
    p["bin_low_m"] = p.bin_index * bin_m
    p["bin_high_m"] = (p.bin_index + 1) * bin_m
    p["bin_m"] = bin_m
    p["flight_id"], p["phase"] = str(df.flight_id.iloc[0]), str(df.phase.iloc[0])
    p["source_kind"] = str(df.source_kind.iloc[0]) if "source_kind" in df else "unspecified"
    return p


def smooth_supported(profile, column, window=3):
    """Centered boxcar in contiguous supported bins; preserve every gap."""
    if window < 1 or window % 2 != 1:
        raise ValueError("Smoothing window must be a positive odd integer")
    y = profile[column].where(profile.supported).copy()
    runs = y.notna().ne(y.notna().shift(fill_value=False)).cumsum()
    out = pd.Series(np.nan, index=profile.index)
    for _, segment in y[y.notna()].groupby(runs[y.notna()]):
        out.loc[segment.index] = segment.rolling(window, center=True, min_periods=1).mean()
    return out


def profile_errors(profile):
    m = profile.supported & profile.rate_cps.notna() & profile.predicted_cps.notna()
    e = profile.loc[m, "predicted_cps"] - profile.loc[m, "rate_cps"]
    return {"profile_supported_bins": int(m.sum()),
            "profile_rmse_cps": np.sqrt(np.mean(e**2)) if m.any() else np.nan,
            "profile_mae_cps": np.mean(np.abs(e)) if m.any() else np.nan}


def overlap_table(profiles):
    """Exact common-grid intersection, by phase; no interpolation/extrapolation."""
    rows = []
    flights = sorted(profiles.flight_id.unique())
    for phase, g in profiles.groupby("phase"):
        common = g[g.supported].pivot(index="bin_index", columns="flight_id", values="rate_cps")
        common = common.reindex(columns=flights).dropna(how="any")
        if not common.empty:
            q = common.reset_index().melt(id_vars="bin_index", var_name="flight_id", value_name="rate_cps")
            q["phase"], q["bin_m"] = phase, float(g.bin_m.iloc[0])
            q["altitude_m"] = (q.bin_index + .5) * q.bin_m
            rows.append(q)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["bin_index", "flight_id", "rate_cps", "phase", "bin_m", "altitude_m"])

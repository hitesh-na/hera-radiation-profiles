import numpy as np
import pandas as pd

REQUIRED = ["flight_id", "Timestamp_s", "Counts_Per_Second", "Altitude[m]"]


def audit_table(df, max_match_error_s=2.5, altitude_limits=(-500, 60000)):
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    d = df.copy().reset_index(drop=True)
    d["source_row"] = np.arange(len(d)) if "source_row" not in d else d.source_row
    reasons = pd.Series("", index=d.index)
    def reject(mask, reason):
        reasons.loc[mask] += reason + ";"
    reject(d.flight_id.isna() | (d.flight_id.astype(str).str.strip() == ""), "missing_flight_id")
    d["flight_id"] = d.flight_id.astype(str)
    for c in REQUIRED[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        reject(~np.isfinite(d[c]), "nonfinite_" + c)
    reject(d.Counts_Per_Second < 0, "negative_rate")
    reject(~d["Altitude[m]"].between(*altitude_limits), "altitude_outside_configured_range")
    reject(d.duplicated(["flight_id", "Timestamp_s"], keep=False), "duplicate_flight_timestamp")
    if "preprocessing_valid" in d:
        valid = d.preprocessing_valid.astype(str).str.lower().isin(["true", "1"])
        reject(~valid, "preprocessing_invalid")
    if "match_time_error_s" in d:
        gap = pd.to_numeric(d.match_time_error_s, errors="coerce")
        reject(~np.isfinite(gap) | (gap > max_match_error_s) | (gap < 0), "unmatched_or_distant_sensor")
    pair = {"Raw_Counts", "Live_Time_s"}
    if "Raw_Counts" in d and "Live_Time_s" not in d:
        raise ValueError("Raw_Counts supplied without Live_Time_s: provide verified live time or omit counts.")
    if "Live_Time_s" in d:
        d["Live_Time_s"] = pd.to_numeric(d.Live_Time_s, errors="coerce")
        reject(~np.isfinite(d.Live_Time_s) | (d.Live_Time_s <= 0), "invalid_exposure")
    if pair.issubset(d.columns):
        d["Raw_Counts"] = pd.to_numeric(d.Raw_Counts, errors="coerce")
        reject(~np.isfinite(d.Raw_Counts) | (d.Raw_Counts < 0) |
               ~np.isclose(d.Raw_Counts, d.Raw_Counts.round()), "invalid_integer_counts")
        expected = d.Raw_Counts / d.Live_Time_s
        reject(~np.isclose(d.Counts_Per_Second, expected, rtol=1e-4, atol=1e-6), "rate_count_exposure_mismatch")
        d["Rate_Error"] = np.sqrt(d.Raw_Counts.clip(lower=0)) / d.Live_Time_s
    sensor_flags = []
    limits = {"Pressure[Pa]": (1, 120000), "Temperature[C]": (-120, 100), "Box_Temperature[C]": (-120, 100),
              "Latitude": (-90, 90), "Longitude": (-180, 180)}
    sensor_columns = ["Pressure[Pa]", "Temperature[C]", "Accel", "Mag", "Latitude", "Longitude"]
    if "Box_Temperature[C]" in d:
        sensor_columns.append("Box_Temperature[C]")
    for c in sensor_columns:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d else np.nan
        bad = ~np.isfinite(d[c])
        if c in limits:
            bad |= ~d[c].between(*limits[c])
        elif c in ["Accel", "Mag"]:
            bad |= d[c] < 0
        for flight, g in d.groupby("flight_id"):
            sensor_flags.append({"flight_id": flight, "column": c,
                                 "masked_or_missing_rows": int(bad.loc[g.index].sum())})
        d.loc[bad, c] = np.nan
    no_fix = (d.Latitude == 0) & (d.Longitude == 0)
    d.loc[no_fix, ["Latitude", "Longitude"]] = np.nan
    d["exclusion_reason"] = reasons.str.rstrip(";")
    excluded = d[d.exclusion_reason != ""].copy()
    clean = d[d.exclusion_reason == ""].drop(columns="exclusion_reason").copy()
    if clean.empty:
        raise ValueError("No valid rows remain; inspect units, clocks, exposure and missing values.")
    summary = []
    for flight, g in d.groupby("flight_id"):
        keep = clean[clean.flight_id == flight]
        ts = keep.Timestamp_s.sort_values()
        summary.append({"flight_id": flight, "input_rows": len(g), "valid_rows": len(keep),
                        "excluded_rows": len(g) - len(keep),
                        "altitude_min_m": keep["Altitude[m]"].min(),
                        "altitude_max_m": keep["Altitude[m]"].max(),
                        "largest_time_gap_s": ts.diff().max(),
                        "zero_count_rows": int((keep.get("Raw_Counts", pd.Series(dtype=float)) == 0).sum()),
                        "counts_and_exposure_available": pair.issubset(d.columns),
                        "gps_zero_pair_rows": int(no_fix.loc[g.index].sum())})
    return clean, excluded, pd.DataFrame(summary), pd.DataFrame(sensor_flags)

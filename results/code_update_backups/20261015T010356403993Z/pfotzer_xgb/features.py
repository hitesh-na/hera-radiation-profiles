"""Deterministic, sensor-only OFFLINE features. No fitted target transform."""
import numpy as np
import pandas as pd

SENSORS = {"Altitude[m]": "altitude_m", "Pressure[Pa]": "pressure_pa",
           "Temperature[C]": "temperature_c", "Accel": "acceleration",
           "Mag": "magnetic_field", "Latitude": "latitude", "Longitude": "longitude"}
BASIC = list(SENSORS.values())
EXTRA = ["log_pressure", "density_proxy_kg_m3", "vertical_speed_m_s",
         "pressure_change_pa_s", "phase_code"]
FEATURE_SETS = {"baseline": BASIC, "engineered": BASIC + EXTRA,
                "tuned_engineered": BASIC + EXTRA, "altitude_only": ["altitude_m"],
                "pressure_only": ["pressure_pa"],
                "altitude_pressure": ["altitude_m", "pressure_pa"]}


def engineer(df, max_gap_s=30, window=5, phase_threshold_m_s=0.5):
    """Centered windows use the completed sensor trajectory (not real-time)."""
    if window < 1 or window % 2 != 1:
        raise ValueError("Feature window must be an odd positive integer")
    d = df.sort_values(["flight_id", "Timestamp_s"]).copy().reset_index(drop=True)
    for source, dest in SENSORS.items():
        d[dest] = pd.to_numeric(d[source], errors="coerce") if source in d else np.nan
    # Use the same sensor validity rules during training and later inference.
    for col, bounds in {"pressure_pa": (1, 120000), "temperature_c": (-120, 100),
                        "latitude": (-90, 90), "longitude": (-180, 180)}.items():
        d.loc[~d[col].between(*bounds), col] = np.nan
    for col in ["acceleration", "magnetic_field"]:
        d.loc[d[col] < 0, col] = np.nan
    no_fix = (d.latitude == 0) & (d.longitude == 0)
    d.loc[no_fix, ["latitude", "longitude"]] = np.nan
    p = d.pressure_pa.where(d.pressure_pa > 0)
    tk = (d.temperature_c + 273.15).where(d.temperature_c > -273.15)
    d["log_pressure"] = np.log(p)
    d["density_proxy_kg_m3"] = p / (287.05 * tk)
    d["vertical_speed_m_s"] = np.nan
    d["pressure_change_pa_s"] = np.nan
    for _, g in d.groupby("flight_id", sort=False):
        gaps = g.Timestamp_s.diff()
        if (gaps <= 0).any():
            raise ValueError("Duplicate or decreasing flight timestamps")
        segments = (gaps > max_gap_s).cumsum()
        for _, s in g.groupby(segments):
            if len(s) < 2:
                continue
            t = s.Timestamp_s.to_numpy(float)
            for source, dest in [("altitude_m", "vertical_speed_m_s"),
                                 ("pressure_pa", "pressure_change_pa_s")]:
                sm = s[source].rolling(window, center=True, min_periods=1).median()
                vals = np.gradient(sm.to_numpy(float), t)
                d.loc[s.index, dest] = vals
    speed = d.vertical_speed_m_s
    inferred = np.select([speed > phase_threshold_m_s, speed < -phase_threshold_m_s,
                          speed.notna()], ["ascent", "descent", "stationary"], "unknown")
    if "phase" not in d:
        d["phase"] = inferred
    else:
        allowed = {"ascent", "descent", "stationary", "unknown"}
        if not set(d.phase.dropna()).issubset(allowed):
            raise ValueError("phase must be ascent/descent/stationary/unknown")
        d["phase"] = d.phase.fillna(pd.Series(inferred, index=d.index))
    d["phase_code"] = d.phase.map({"ascent": 1, "descent": -1, "stationary": 0})
    d[BASIC + EXTRA] = d[BASIC + EXTRA].replace([np.inf, -np.inf], np.nan)
    return d


def matrix(df, variant):
    """An explicit allowlist prevents labels, counts, IDs or peaks entering X."""
    return df[FEATURE_SETS[variant]].astype(float)

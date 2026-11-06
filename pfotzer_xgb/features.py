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

# Professor's clarification: logger temperature is INSIDE the payload and
# magnetic sensors were not calibrated. Acceleration units are still pending.
HERA_BASIC = ["altitude_m", "pressure_pa", "box_temperature_c", "latitude", "longitude"]
HERA_EXTRA = ["log_pressure", "vertical_speed_m_s", "pressure_change_pa_s", "phase_code"]
HERA_FEATURE_SETS = {**FEATURE_SETS, "baseline": HERA_BASIC,
                     "engineered": HERA_BASIC + HERA_EXTRA,
                     "tuned_engineered": HERA_BASIC + HERA_EXTRA}


def engineer(df, max_gap_s=30, window=5, phase_threshold_m_s=0.5,
             temperature_context="ambient"):
    """Centered windows use the completed sensor trajectory (not real-time)."""
    if window < 1 or window % 2 != 1:
        raise ValueError("Feature window must be an odd positive integer")
    d = df.sort_values(["flight_id", "Timestamp_s"]).copy().reset_index(drop=True)
    for source, dest in SENSORS.items():
        d[dest] = pd.to_numeric(d[source], errors="coerce") if source in d else np.nan
    if temperature_context not in {"ambient", "inside_box"}:
        raise ValueError("temperature_context must be ambient or inside_box")
    d["box_temperature_c"] = np.nan
    if temperature_context == "inside_box":
        source = "Box_Temperature[C]" if "Box_Temperature[C]" in d else "Temperature[C]"
        if source in d:
            d["box_temperature_c"] = pd.to_numeric(d[source], errors="coerce")
        d.loc[~d.box_temperature_c.between(-120, 100), "box_temperature_c"] = np.nan
        # Do not relabel instrument temperature as ambient or derive density.
        d["temperature_c"] = np.nan
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


def matrix(df, variant, feature_profile="standard"):
    """An explicit allowlist prevents labels, counts, IDs or peaks entering X."""
    if feature_profile not in {"standard", "hera_uncalibrated_v1"}:
        raise ValueError("Unknown feature_profile")
    sets = HERA_FEATURE_SETS if feature_profile == "hera_uncalibrated_v1" else FEATURE_SETS
    return df[sets[variant]].astype(float)

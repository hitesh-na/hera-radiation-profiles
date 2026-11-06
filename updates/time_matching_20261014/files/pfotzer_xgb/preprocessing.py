"""Explicit units/clock mappings. No silent STAR parser assumptions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .io import sha256, write_json
from .time_alignment import unwrap_logger_timer


def unit_seconds(unit):
    if unit not in {"s", "ms"}:
        raise ValueError("Set verified time units explicitly to 's' or 'ms'")
    return {"s": 1.0, "ms": .001}[unit]


def bin_cumulative_events(times_s, cumulative_dead_s, *, bin_seconds,
                          max_counter_gap_s, continuous_acquisition_verified):
    """Interpolate CUMULATIVE counters at bin boundaries, not whole inter-event
    exposures into the bin containing the final event. Interpolation is an
    approximation requiring confirmed continuous acquisition and bounded gaps.
    Full bins only; no extrapolation beyond the first/last counter record.
    """
    if continuous_acquisition_verified is not True:
        raise ValueError("Continuous acquisition must be verified, or supply an exposure table")
    t, dead = np.asarray(times_s, float), np.asarray(cumulative_dead_s, float)
    if len(t) < 2 or len(t) != len(dead) or not np.isfinite(t).all() or not np.isfinite(dead).all():
        raise ValueError("Invalid time/deadtime arrays")
    dt, dd = np.diff(t), np.diff(dead)
    if (dt <= 0).any() or (dead < 0).any() or (dd < -1e-9).any() or (dd > dt + 1e-8).any():
        raise ValueError("Clock/counter reset, duplicate timestamp, or impossible cumulative deadtime")
    if bin_seconds <= 0 or max_counter_gap_s <= 0:
        raise ValueError("Positive bin and gap limits are required")
    start = np.ceil(t[0] / bin_seconds) * bin_seconds
    stop = np.floor(t[-1] / bin_seconds) * bin_seconds
    edges = np.arange(start, stop + .5 * bin_seconds, bin_seconds)
    if len(edges) < 2:
        raise ValueError("Insufficient fully bracketed exposure for a bin")
    live = np.diff(edges) - np.diff(np.interp(edges, t, dead))
    # Every interval is [left, right), including the final bin.
    counts = np.diff(np.searchsorted(t, edges, side="left"))
    gap_bad = np.zeros(len(live), bool)
    for i in np.flatnonzero(dt > max_counter_gap_s):
        gap_bad |= (edges[:-1] < t[i + 1]) & (edges[1:] > t[i])
    valid = (live > 0) & (live <= np.diff(edges) + 1e-8) & ~gap_bad
    rate = np.divide(counts, live, out=np.full(len(live), np.nan), where=valid)
    return pd.DataFrame({"Timestamp_s": (edges[1:] + edges[:-1]) / 2,
                         "bin_start_s": edges[:-1], "bin_end_s": edges[1:],
                         "Raw_Counts": counts, "Live_Time_s": live,
                         "Counts_Per_Second": rate,
                         "Rate_Error": np.where(valid, np.sqrt(counts) / np.maximum(live, 1e-300), np.nan),
                         "preprocessing_valid": valid,
                         "preprocessing_reason": np.where(gap_bad, "counter_gap", np.where(valid, "", "invalid_live_time"))})


def bin_with_exposure(times_s, exposure):
    """Preferred when the instrument supplies independently verified exposure."""
    e = exposure.sort_values("bin_start_s").reset_index(drop=True).copy()
    cols = ["bin_start_s", "bin_end_s", "Live_Time_s"]
    if not set(cols).issubset(e):
        raise ValueError(f"Exposure table needs {cols}")
    x = e[cols].to_numpy(float)
    if not np.isfinite(x).all() or (e.bin_end_s <= e.bin_start_s).any():
        raise ValueError("Invalid exposure boundaries")
    if (e.bin_start_s.to_numpy()[1:] < e.bin_end_s.to_numpy()[:-1]).any():
        raise ValueError("Exposure intervals overlap")
    if (e.Live_Time_s <= 0).any() or (e.Live_Time_s > e.bin_end_s - e.bin_start_s + 1e-8).any():
        raise ValueError("Exposure must be positive and at most wall-clock duration")
    t = np.asarray(times_s, float)
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError("Event times must be finite and strictly increasing")
    e["Raw_Counts"] = np.searchsorted(t, e.bin_end_s, side="left") - np.searchsorted(t, e.bin_start_s, side="left")
    e["Timestamp_s"] = (e.bin_start_s + e.bin_end_s) / 2
    e["Counts_Per_Second"] = e.Raw_Counts / e.Live_Time_s
    e["Rate_Error"] = np.sqrt(e.Raw_Counts) / e.Live_Time_s
    e["preprocessing_valid"] = True
    e["preprocessing_reason"] = ""
    return e


def parse_clock(frame, config):
    mode = config["mode"]
    if mode == "numeric":
        t = pd.to_numeric(frame[config["column"]], errors="raise").to_numpy(float) * unit_seconds(config["unit"])
    elif mode == "numeric_wrapped":
        raw = pd.to_numeric(frame[config["column"]], errors="raise").to_numpy(float) * unit_seconds(config["unit"])
        t, _ = unwrap_logger_timer(raw, float(config["rollover_s"]))
    elif mode == "date_rtc":
        # The label EST alone is not enough: timezone and date format are required.
        dates = pd.to_datetime(frame[config["date_column"]], format=config["date_format"])
        sec = pd.to_timedelta(frame[config["time_column"]].astype(str)).dt.total_seconds()
        naive = dates + pd.to_timedelta(sec, unit="s")
        utc = naive.dt.tz_localize(config["timezone"], ambiguous="raise", nonexistent="raise").dt.tz_convert("UTC")
        t = utc.astype("int64").to_numpy(float) / 1e9
    elif mode == "rtc_only":
        s = pd.to_timedelta(frame[config["column"]].astype(str)).dt.total_seconds().to_numpy()
        if not np.isfinite(s).all() or ((s < 0) | (s >= 86400)).any():
            raise ValueError("Invalid time of day")
        delta = np.diff(s)
        if ((delta < 0) & (delta > -43200)).any():
            raise ValueError("RTC moves backwards other than a midnight rollover")
        t = s + np.r_[0, np.cumsum(delta < -43200)] * 86400
    else:
        raise ValueError("Clock mode must be numeric, numeric_wrapped, date_rtc or rtc_only")
    if not np.isfinite(t).all() or (np.diff(t) < 0).any():
        raise ValueError("Invalid or reset clock; split recording into verified segments")
    return t - float(config["reference_origin_s"]) + float(config["offset_s"])


def synchronize(cw, clue, tolerance_s):
    if tolerance_s <= 0:
        raise ValueError("A positive, finite synchronization tolerance is required")
    if not np.isfinite(tolerance_s):
        raise ValueError("Synchronization tolerance cannot be unlimited")
    c = clue.sort_values("Timestamp_s").copy()
    # Duplicate sensor timestamps can be normal. Aggregate same-time numeric
    # readings; counts/exposure always remain one row per CW interval.
    c = c.groupby("Timestamp_s", as_index=False).mean(numeric_only=True)
    c = c.rename(columns={"Timestamp_s": "sensor_timestamp_s"})
    d = pd.merge_asof(cw.sort_values("Timestamp_s"), c,
                      left_on="Timestamp_s", right_on="sensor_timestamp_s",
                      direction="nearest", tolerance=tolerance_s)
    d["match_time_error_s"] = (d.Timestamp_s - d.sensor_timestamp_s).abs()
    return d


def prepare(manifest_path, output_dir):
    manifest_path = Path(manifest_path).resolve()
    cfg = json.loads(manifest_path.read_text())
    if cfg.get("source_kind") != "real_hera" or cfg.get("verified") is not True:
        raise ValueError("Verify the raw-file manifest and set verified=true before preprocessing")
    root = manifest_path.parent
    outputs, audit = [], []
    for spec in cfg["flights"]:
        if spec.get("verified") is not True:
            raise ValueError(f"Unverified conventions for {spec['flight_id']}")
        cwcfg, cluecfg = spec["cosmicwatch"], spec["clue"]
        cwpath, clpath = root / cwcfg["path"], root / cluecfg["path"]
        events = np.loadtxt(cwpath, delimiter=cwcfg.get("delimiter"),
                            comments=cwcfg.get("comments", "#"), skiprows=cwcfg.get("skiprows", 0), ndmin=2)
        t = events[:, cwcfg["time_column"]] * unit_seconds(cwcfg["time_unit"])
        if "event_id_column" in cwcfg and cwcfg["event_id_column"] is not None:
            ids = events[:, cwcfg["event_id_column"]]
            if not np.isfinite(ids).all() or not np.allclose(np.diff(ids), 1):
                raise ValueError("Event counter gaps/resets: verify loss, selection or restart before counting rows")
        exposure_path = None
        if cwcfg["exposure_method"] == "exposure_table":
            exposure_path = root / cwcfg["exposure_path"]
            bins = bin_with_exposure(t, pd.read_csv(exposure_path))
        elif cwcfg["exposure_method"] == "cumulative_linear":
            dead = events[:, cwcfg["deadtime_column"]] * unit_seconds(cwcfg["deadtime_unit"])
            bins = bin_cumulative_events(t, dead, bin_seconds=cfg["time_bin_s"],
                max_counter_gap_s=cwcfg["max_counter_gap_s"],
                continuous_acquisition_verified=cwcfg["continuous_acquisition_verified"])
        else:
            raise ValueError("Explicit verified exposure_method required")
        shift = -float(cwcfg["reference_origin_s"]) + float(cwcfg["offset_s"])
        bins[["Timestamp_s", "bin_start_s", "bin_end_s"]] += shift
        raw = pd.read_csv(clpath)
        if cluecfg.get("filter_equals"):
            for col, val in cluecfg["filter_equals"].items():
                raw = raw[raw[col].astype(str) == str(val)].copy()
        clue = pd.DataFrame({"Timestamp_s": parse_clock(raw, cluecfg["clock"])})
        for dest, src in cluecfg["columns"].items():
            clue[dest] = pd.to_numeric(raw[src], errors="coerce").to_numpy() * float(cluecfg.get("scales", {}).get(dest, 1))
        for dest, axes in cluecfg.get("vectors", {}).items():
            v = raw[axes["columns"]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            offsets = np.asarray(axes.get("offsets", [0, 0, 0]), float)
            clue[dest] = np.linalg.norm((v - offsets) * float(axes["scale"]), axis=1)
        d = synchronize(bins, clue, cfg["match_tolerance_s"])
        d["flight_id"], d["detector_id"], d["source_kind"] = spec["flight_id"], spec["detector_id"], "real_hera"
        outputs.append(d)
        audit.append({"flight_id": spec["flight_id"], "event_rows": len(events),
                      "clue_rows": len(raw), "duplicate_sensor_times": int(clue.Timestamp_s.duplicated().sum()),
                      "binned_rows": len(d), "unmatched_rows": int(d.sensor_timestamp_s.isna().sum()),
                      "counted_events": int(d.Raw_Counts.sum()),
                      "events_outside_covered_intervals": int(len(events) - d.Raw_Counts.sum()),
                      "cosmicwatch_sha256": sha256(cwpath), "clue_sha256": sha256(clpath),
                      "exposure_sha256": sha256(exposure_path) if exposure_path else None})
    result = pd.concat(outputs, ignore_index=True)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "combined.csv").exists():
        raise ValueError("combined.csv already exists; use a new preprocessing directory")
    result.to_csv(out / "combined.csv", index=False)
    write_json(out / "provenance.json", {"manifest": cfg, "manifest_sha256": sha256(manifest_path),
        "files": audit, "data_sha256": sha256(out / "combined.csv"),
        "note": "Cumulative-counter interpolation is an approximation; review detector semantics before accepting real rates."})
    return out / "combined.csv"

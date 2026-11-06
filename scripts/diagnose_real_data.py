"""Inspect HERA recordings; produce a report, never training data or models.

All clock repairs and pressure fits are CANDIDATES, not verified calibration.
Requires the original project requirements and data/raw/HERA_sources.json.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import least_squares


VERSION = "1.0"
ROLLOVER_S = 2**32 / 1_000_000
STATUS = "DIAGNOSTICS_COMPLETE_TRAINING_PENDING"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_timer(raw):
    """Test one explicit wrap hypothesis; reject other backwards steps.

    A genuine reboot can mimic a wrap. Numerical success is not verification.
    """
    raw = np.asarray(raw, float)
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("Logger timer must contain finite values")
    if ((raw < 0) | (raw >= ROLLOVER_S)).any():
        raise ValueError("Timer values do not fit the proposed rollover range")
    jumps = np.flatnonzero(np.diff(raw) < 0) + 1
    if len(jumps) and not ((raw[jumps - 1] > .7 * ROLLOVER_S)
                          & (raw[jumps] < .3 * ROLLOVER_S)).all():
        raise ValueError("Backwards timer step is not a candidate rollover")
    unwrapped = raw + np.r_[0, np.cumsum(np.diff(raw) < 0)] * ROLLOVER_S
    if (np.diff(unwrapped) <= 0).any():
        raise ValueError("Candidate timer remains non-increasing")
    return unwrapped, jumps


def pressure_medians(time_s, pressure_pa):
    """Two-second medians used ONLY for sensor comparison, never event cuts."""
    z = pd.DataFrame({"t": time_s, "p": pressure_pa})
    z = z[np.isfinite(z.t) & np.isfinite(z.p) & (z.p > 0)].copy()
    z["bin"] = np.floor(z.t / 2)
    z = z.groupby("bin", sort=True)[["t", "p"]].median()
    return z.t.to_numpy(), z.p.to_numpy()


def map_to_logger(t, parameters, pivot_s):
    shift, ppm, _, _ = parameters
    return pivot_s + (1 + ppm / 1e6) * (np.asarray(t) - pivot_s) + shift


def fit_pressure_clocks(td, pd_, tl, pl):
    """Estimate sensor-only candidates, allowing pressure gain/bias differences.

    Fits use no detector counts, radiation rates, GPS altitude or ML scores.
    Pressure response differences can masquerade as clock error.
    """
    t, p = pressure_medians(td, pd_)
    u, q = pressure_medians(tl, pl)
    # Deliberately exclude ground plateaus, the lowest pressures, and edges.
    mask = (p > 5000) & (p < 90000) & (t > u[0] + 600) & (t < u[-1] - 600)
    x, y = t[mask][::2], p[mask][::2]
    if len(x) < 100:
        raise ValueError("Too few pressure samples for the diagnostic fit")
    pivot = float(np.median(x))
    turn = float(t[np.argmin(p)])
    lo, hi = [-300, -10000, .95, -1000], [300, 10000, 1.05, 1000]

    def fit(xx, yy, initial):
        def residual(z):
            predicted_p = np.interp(map_to_logger(xx, z, pivot), u, q)
            return (predicted_p - (z[2] * yy + z[3])) / 100
        opt = least_squares(residual, initial, bounds=(lo, hi), loss="soft_l1",
                            x_scale=[20, 1000, .01, 100], max_nfev=400)
        at = map_to_logger(xx, opt.x, pivot)
        idx = np.clip(np.searchsorted(u, at), 1, len(u) - 1)
        gaps = u[idx] - u[idx - 1]
        r = residual(opt.x) * 100
        return {"parameters": [float(v) for v in opt.x], "success": bool(opt.success),
                "at_parameter_bounds": [int(v) for v in opt.active_mask],
                "n_pressure_samples": len(xx),
                "detector_time_range_s": [float(xx.min()), float(xx.max())],
                "median_absolute_pressure_residual_pa": float(np.median(abs(r))),
                "p95_absolute_pressure_residual_pa": float(np.quantile(abs(r), .95)),
                "largest_logger_interpolation_gap_s": float(gaps.max()),
                "out_of_logger_time_support": int(((at < u[0]) | (at > u[-1])).sum())}

    joint = fit(x, y, [0, 0, 1, 0])
    fits = {"joint": joint}
    for name, keep in [("before_pressure_minimum", x < turn),
                       ("after_pressure_minimum", x > turn)]:
        if keep.sum() >= 50:
            fits[name] = fit(x[keep], y[keep], joint["parameters"])
    info = {"verified": False, "pivot_detector_s": pivot,
            "pressure_minimum_detector_s": turn,
            "parameter_order": ["shift_at_pivot_s", "relative_clock_ppm",
                                "pressure_gain", "pressure_bias_pa"],
            "equation_time": "t_logger = pivot + (1 + ppm/1e6)*(t_detector-pivot) + shift",
            "equation_pressure": "P_logger(t_logger) ~= gain*P_detector(t_detector) + bias",
            "fits": fits,
            "note": "Sensor-only fit; residual Pa is NOT timing accuracy or a confidence interval."}
    return info, (t, p, u, q, x, y)


def inspect_flight(fid, pair, out):
    detector_path, logger_path = pair["detector"], pair["logger"]
    header = []
    with detector_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header.append(line.rstrip())
    header_text = "\n".join(header)
    if not all(s in header_text for s in ["Time[s]", "Deadtime[s]", "Pressure[Pa]"]):
        raise ValueError(f"{fid}: unexpected detector units/header; inspect before adapting")
    events = np.loadtxt(detector_path, comments="#", ndmin=2)
    logger = pd.read_csv(logger_path)
    if events.shape[1] != 8 or not np.isfinite(events).all():
        raise ValueError(f"{fid}: expected eight finite detector columns")
    td, dead, pressure = events[:, 1], events[:, 5], events[:, 7]
    if (np.diff(td) <= 0).any():
        raise ValueError(f"{fid}: detector timestamps are not strictly increasing")
    timer_raw = pd.to_numeric(logger["Timer[S]"], errors="raise").to_numpy(float)
    tl, jumps = candidate_timer(timer_raw)
    lp = pd.to_numeric(logger["Pressure[Pa]"], errors="raise").to_numpy(float)
    dt, dd = np.diff(td), np.diff(dead)
    span = td[-1] - td[0]

    day = pd.to_datetime(logger["Date[MM/DD/YYYY]"], format="%m/%d/%Y", errors="coerce")
    tod = pd.to_timedelta(logger["RTC[H:M:S EST]"].astype(str), errors="coerce").dt.total_seconds()
    valid_date = day.notna() & (day.dt.year >= 2020) & tod.notna()
    rtc = day.astype("int64").to_numpy(float) / 1e9 + tod.to_numpy()
    valid_pairs = valid_date.to_numpy()[1:] & valid_date.to_numpy()[:-1]
    rtc_backward = int(((np.diff(rtc) < 0) & valid_pairs).sum())
    date_changes = np.flatnonzero((day.astype(str).to_numpy()[1:] != day.astype(str).to_numpy()[:-1])
                                 & valid_pairs) + 1
    transition_records = [{"data_row_1based": int(i + 1),
                           "date": str(logger["Date[MM/DD/YYYY]"].iloc[i]),
                           "displayed_time": str(logger["RTC[H:M:S EST]"].iloc[i])}
                          for i in date_changes]
    alt, lat, lon, sats = [pd.to_numeric(logger[k], errors="coerce").to_numpy(float)
                            for k in ["Altitude[m]", "Latitude", "Longitude", "Sats"]]
    gps = (np.isfinite(alt) & np.isfinite(lat) & np.isfinite(lon) & (sats >= 4)
           & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
           & ~((lat == 0) & (lon == 0)) & (alt >= -500) & (alt <= 60000))
    median_p = pd.Series(pressure, index=pd.to_timedelta(td, unit="s")).rolling(
        "10s", center=True, min_periods=3).median().to_numpy()
    spikes = np.abs(pressure - median_p) > np.maximum(1000, .05 * np.abs(median_p))
    accepted_gps = np.flatnonzero(gps)
    accepted_gaps = np.diff(tl[accepted_gps])
    gps_gap_records = []
    for i in np.flatnonzero(accepted_gaps > 30):
        left, right = accepted_gps[i], accepted_gps[i + 1]
        gps_gap_records.append({"start_candidate_logger_s": float(tl[left]),
                                "end_candidate_logger_s": float(tl[right]),
                                "duration_s": float(tl[right] - tl[left]),
                                "last_accepted_altitude_m": float(alt[left]),
                                "next_accepted_altitude_m": float(alt[right])})
    counters = {}
    for col in logger.columns:
        if col.startswith("Event"):
            c = pd.to_numeric(logger[col], errors="raise").to_numpy(float)
            delta = np.diff(c)
            counters[col] = {"first": float(c[0]), "last": float(c[-1]),
                             "negative_steps": int((delta < 0).sum()),
                             "steps_greater_than_one": int((delta > 1).sum())}
    # Ignore flagged PRESSURE readings in the diagnostic fit only. Events and
    # their timestamps, counts, amplitudes and original pressure remain intact.
    clock, plot_data = fit_pressure_clocks(td, np.where(spikes, np.nan, pressure), tl, lp)
    summary = {
        "flight_id": fid, "scientifically_verified": False,
        "event_rows": len(events), "logger_rows": len(logger),
        "detector_duration_s": float(span), "logger_candidate_duration_s": float(tl[-1] - tl[0]),
        "detector_event_id_nonunit_steps": int((np.diff(events[:, 0]) != 1).sum()),
        "detector_largest_interevent_gap_s": float(dt.max()),
        "deadtime_decreases": int((dd < -1e-9).sum()),
        "deadtime_increment_greater_than_elapsed": int((dd > dt + 1e-8).sum()),
        "negative_deadtime_values": int((dead < 0).sum()),
        "cumulative_deadtime_change_s": float(dead[-1] - dead[0]),
        "deadtime_fraction_if_interpretation_correct": float((dead[-1] - dead[0]) / span),
        "detector_coincidence_ones": int((events[:, 2] == 1).sum()),
        "detector_other_coincidence_values": int((~np.isin(events[:, 2], [0, 1])).sum()),
        "adc_at_or_above_header_limit": int((events[:, 3] >= 4095).sum()),
        "adc_max_observed": float(events[:, 3].max()),
        "detector_pressure_spike_flags": int(spikes.sum()),
        "candidate_timer_rollovers": len(jumps), "proposed_rollover_s": ROLLOVER_S,
        "rtc_backwards_steps_valid_date_pairs": rtc_backward,
        "invalid_or_old_date_rows": int((~valid_date).sum()),
        "date_transitions_with_valid_dates": transition_records,
        "logger_largest_candidate_gap_s": float(np.diff(tl).max()),
        "gps_basic_screen_rows": int(gps.sum()),
        "gps_basic_screen_max_altitude_m": float(alt[gps].max()) if gps.any() else None,
        "largest_gap_between_accepted_gps_readings_s": float(accepted_gaps.max()) if len(accepted_gaps) else None,
        "gps_gaps_above_30s": gps_gap_records,
        "logger_counters": counters, "candidate_clock_alignment": clock,
    }
    pd.DataFrame({"data_row_1based": jumps + 1, "previous_timer_s": timer_raw[jumps - 1],
                  "next_timer_s": timer_raw[jumps],
                  "candidate_interval_s": tl[jumps] - tl[jumps - 1]}).to_csv(
                      out / f"{fid}_timer_jumps.csv", index=False)
    flagged = np.flatnonzero(spikes)
    pd.DataFrame({"event_data_row_1based": flagged + 1, "event_id": events[flagged, 0],
                  "detector_time_s": td[flagged], "raw_pressure_pa": pressure[flagged],
                  "local_ten_second_median_pa": median_p[flagged]}).to_csv(
                      out / f"{fid}_pressure_flags.csv", index=False)

    t, p, u, q, x, y = plot_data
    params = clock["fits"]["joint"]["parameters"]
    pivot = clock["pivot_detector_s"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle(f"{fid}: REAL DATA DIAGNOSTICS — clock fits are provisional", fontsize=14)
    ax = axes[0, 0]
    ax.plot(td[::20] / 60, pressure[::20] / 1000, color="#bdd4e7", lw=.5, label="Detector raw (sampled)")
    ax.plot(t / 60, p / 1000, color="#286090", lw=1, label="Detector: 2 s medians")
    ax.plot(u / 60, q / 1000, color="#c16b20", lw=1, label="Logger: candidate timer")
    if len(flagged):
        ax.scatter(td[flagged] / 60, pressure[flagged] / 1000, marker="x", color="#b83e40", s=18,
                   label="Pressure spike flags")
    ax.set(xlabel="Each device's elapsed reading (min)", ylabel="Pressure (kPa)", title="Before pressure-based time alignment")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    r = np.interp(map_to_logger(x, params, pivot), u, q) - (params[2] * y + params[3])
    ax.scatter(x / 60, r, s=3, color="#286090", alpha=.65)
    ax.axhline(0, color="grey", lw=.7)
    ax.set_yscale("symlog", linthresh=100)
    ax.set(xlabel="Detector elapsed time (min)", ylabel="Pressure residual (Pa)",
           title="Joint fit residuals (linear within +/-100 Pa)\nResidual is not timing accuracy")
    ax = axes[1, 0]
    labels = {"joint": "Both flight portions", "before_pressure_minimum": "Before pressure minimum",
              "after_pressure_minimum": "After pressure minimum"}
    for name, f in clock["fits"].items():
        xx = np.linspace(*f["detector_time_range_s"], 200)
        ax.plot(xx / 60, map_to_logger(xx, f["parameters"], pivot) - xx, label=labels[name], lw=1.5)
    ax.set(xlabel="Detector elapsed time (min)", ylabel="Proposed logger minus detector (s)",
           title="Clock-fit sensitivity to flight portion")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(tl[::10] / 60, np.where(gps, alt / 1000, np.nan)[::10], color="#3c815b", lw=1)
    ax.set(xlabel="Candidate logger elapsed time (min)", ylabel="Reported GPS altitude (km)",
           title="GPS after a basic validity screen\nNot independent calibration")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(out / f"{fid}_diagnostics.png", dpi=140)
    plt.close(fig)
    return summary


def write_reports(out, summaries, provenance):
    rows = []
    for s in summaries:
        gps_max = s['gps_basic_screen_max_altitude_m']
        gps_text = f"{gps_max / 1000:.2f}" if gps_max is not None else "unavailable"
        rows.append(f"| {s['flight_id']} | {s['event_rows']:,} | {s['candidate_timer_rollovers']} | "
                    f"{s['rtc_backwards_steps_valid_date_pairs']:,} | "
                    f"{s['deadtime_fraction_if_interpretation_correct'] * 100:.3f}% | "
                    f"{gps_text} |")
    overview = ["# HERA real-data diagnostics", "", f"**Status: {STATUS}**", "",
        "These are measurements from the original real recordings. No model was trained. "
        "No training table, verified metadata, CPS result or Pfotzer peak was produced.", "",
        "| Flight | Detector events | Timer wraps (candidate) | RTC backwards steps | Deadtime/span* | GPS max (km)** |",
        "| --- | ---: | ---: | ---: | ---: | ---: |", *rows, "",
        "*Assumes the header's seconds and cumulative-deadtime interpretation. Numerical consistency "
        "does not establish continuous acquisition or all sources of lost exposure.", "",
        "**Basic GPS screen only; neither altitude accuracy nor peak coverage is established.", "",
        "## What the checks establish", "",
        "- The input files match the recorded SHA-256 fingerprints.",
        "- Full detector timestamps, event IDs and deadtime increments were inspected; exact issue counts are in summary.json.",
        "- Adding 4294.967296 seconds at each qualifying timer drop gives an increasing candidate logger clock. "
        "This is the 2^32-microsecond hypothesis; firmware or an independent clock must confirm it.",
        "- The displayed date/time moves backwards within recordings. In June the date advances near 20:00, "
        "so it must not be treated as a clean absolute clock without checking date/time conventions.",
        "- Both devices' pressure traces can be compared. Their fitted timing depends on the flight portion "
        "and pressure-response assumptions. Matching pressure curves is not proof of synchronization to 2.5 seconds.",
        "- A few isolated detector pressure readings are flagged. These are review flags, not deleted radiation events.",
        "- Logger event counters and detector event rows represent different measurements. "
        "Do not pool them or obtain a count rate by counting logger rows.", "",
        "## GPS coverage that needs attention", "",
        "The longest intervals between readings that pass the GPS screen are listed below. "
        "These durations use the candidate logger clock. A gap does not mean the detector stopped recording; "
        "it means those events lack GPS readings that pass this screen.", "",
        *[f"- {s['flight_id']}: " +
          (f"{s['largest_gap_between_accepted_gps_readings_s'] / 60:.2f} minutes."
           if s['largest_gap_between_accepted_gps_readings_s'] is not None else "insufficient accepted GPS readings.")
          for s in summaries], "",
        "October's roughly 46-minute interval runs from an accepted altitude near 15.10 km to the next near 31.42 km. "
        "It spans a substantial part of ascent. Do not interpolate that entire gap for training or claim a measured "
        "ascent peak there. Investigate the GPS flags or an independently matched tracker record. "
        "The valid descent data can be assessed separately. Exact intervals are in summary.json.", "",
        "## Needed to finish scientific preprocessing", "",
        "1. Logger source code/version (Timer[S], date/RTC construction, units), any shared start-time or synchronization record, "
        "and an independent tracker/GPS record for the October flight if available.",
        "2. CosmicWatch firmware/version/configuration used for these specific flights, plus known pauses, resets, losses or changes of recording mode. "
        "Confirm cumulative Deadtime[s] and what acquisition losses it covers.",
        "3. Detector identity, trigger threshold, geometry/orientation and singles/coincidence mode for each flight. "
        "Confirm logger acceleration/magnetic units and whether temperature is ambient or inside the payload.", "",
        "You can obtain these from the experiment team. The filename 'AxLab' by itself does not identify a calibrated detector. "
        "Existing project v2.1 slides support cumulative deadtime, but do not document each flight's settings.", "",
        "## Method and limits", "",
        "The pressure comparison uses medians in 2-second intervals, retains every second occupied detector interval, "
        "selects detector pressure strictly between 5,000 and 90,000 Pa, and excludes 600 seconds at either logger edge. "
        "These are diagnostic choices, not a validated sensor operating range. The joint fit uses both sides of the pressure minimum. "
        "Separate fits examine the portions before and after that minimum.", "",
        "The fit minimizes soft-L1 pressure residuals with a 100 Pa scale and parameters: clock offset at the pivot, "
        "relative clock slope (ppm), pressure gain and pressure bias. Bounds are +/-300 s, +/-10,000 ppm, "
        "0.95–1.05 and +/-1,000 Pa. No radiation counts, CPS, altitude peaks or ML score is used to fit clocks. "
        "Pressure gain/bias are nuisance parameters, not a new sensor calibration. Fit success is numerical only; "
        "bounds, overlap and gaps are recorded in summary.json. Do not copy these fitted numbers into a verified manifest.", "",
        "Pressure spike flags compare each event's pressure with its centered 10-second median, using "
        "a difference greater than max(1,000 Pa, 5% of that median). This is not a final cleaning rule. "
        "Flagged pressure readings are omitted from the diagnostic pressure fit only; no original value or event is altered. "
        "The GPS screen requires at least four satellites, "
        "finite coordinates within geographic bounds, a location other than (0,0), and altitude from -500 to 60,000 m. "
        "It does not detect every stale or inaccurate fix.", "",
        "No ADC value at the header's upper limit is not evidence that the analog electronics never saturated. "
        "Likewise, an event-time gap alone cannot distinguish a quiet interval from an acquisition pause.", "",
        "## After the outstanding details are resolved", "",
        "Build and review the synchronized training table, preserve gaps and exposures, then calculate measured profiles. "
        "Run XGBoost with entire flights held out; tune hyperparameters using only the remaining flights. "
        "Three flight groups support an exploratory experiment, with limited evidence for generalization. "
        "Many rows from one flight do not create many independent flights. Predictive relationships do not establish causation.", "",
        "For future flights, retain separate original files and flight settings, evaluate the frozen model first, "
        "and then create a new version for retraining. Do not mix data from a future test flight into tuning.", ""]
    md = "\n".join(overview)
    for s in summaries:
        fid = s["flight_id"]
        md += f"\n## {fid}\n\n![{fid} diagnostic plots]({fid}_diagnostics.png)\n"
    (out / "report.md").write_text(md, encoding="utf-8")
    # Self-contained HTML for a beginner to open without a Markdown extension.
    table_rows = "".join("<tr>" + "".join(f"<td>{html.escape(cell.strip())}</td>"
                         for cell in r.strip("|").split("|")) + "</tr>" for r in rows)
    body = "<h1>HERA real-data diagnostics</h1><p class='status'>Diagnostics complete. Training pending.</p>"
    body += "<p>The raw recordings were checked. No model was trained or verified training table created.</p>"
    body += "<table><tr><th>Flight</th><th>Events</th><th>Candidate wraps</th><th>RTC backwards steps</th><th>Deadtime/span*</th><th>GPS max km**</th></tr>" + table_rows + "</table>"
    body += "<p>*Conditional on cumulative deadtime in seconds. **Basic GPS screen, not calibration.</p>"
    for s in summaries:
        fid = s["flight_id"]
        b64 = base64.b64encode((out / f"{fid}_diagnostics.png").read_bytes()).decode("ascii")
        body += f"<h2>{html.escape(fid)}</h2><img alt='{html.escape(fid)} diagnostics' src='data:image/png;base64,{b64}'>"
    methods_start = overview.index("## What the checks establish")
    body += "<h2>Findings, methods and next steps</h2><pre>" + html.escape("\n".join(overview[methods_start:])) + "</pre>"
    page = "<!doctype html><html lang='en'><meta charset='utf-8'><title>HERA real-data diagnostics</title><style>body{max-width:1150px;margin:36px auto;padding:0 20px;font:16px/1.5 system-ui;color:#23323e}h1{margin-bottom:8px}.status{font-weight:700;color:#8b4a0d}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd5dd;padding:9px;text-align:left}img{width:100%;height:auto}pre{white-space:pre-wrap;font:15px/1.6 system-ui;background:#f3f6f8;padding:22px;border-radius:8px}</style><body>" + body + "</body></html>"
    (out / "report.html").write_text(page, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"status": STATUS, "verified": False,
        "provenance": provenance, "flights": summaries}, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="New results directory; existing paths are refused")
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = root / "data/raw/HERA_sources.json"
    if not (root / "pfotzer_xgb").is_dir():
        raise ValueError("Extract this add-on into the existing project folder")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs, inputs = {}, []
    for record in manifest["files"]:
        path = (root / record["relative_path"]).resolve()
        if not path.is_relative_to(root / "data/raw"):
            raise ValueError("Source manifest paths must remain inside data/raw")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Raw file missing or changed: {record['relative_path']}")
        pair = pairs.setdefault(record["flight_id"], {})
        if record["kind"] in pair:
            raise ValueError("Duplicate flight/kind entry in source manifest")
        pair[record["kind"]] = path
        inputs.append({k: record[k] for k in ["flight_id", "kind", "relative_path", "bytes", "sha256"]})
    if not pairs or any(set(pair) != {"detector", "logger"} for pair in pairs.values()):
        raise ValueError("Each flight needs one detector file and one logger file")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = (args.output if args.output else root / "results" / f"real_diagnostics_{stamp}").resolve()
    if out == root or any(out.is_relative_to(root / sub) for sub in ["data", "config", "pfotzer_xgb", "notebooks", "scripts"]):
        raise ValueError("Choose a new output directory under results")
    out.mkdir(parents=True, exist_ok=False)
    provenance = {"generated_utc": datetime.now(timezone.utc).isoformat(),
                  "diagnostic_version": VERSION, "script_sha256": sha256(__file__),
                  "manifest_sha256": sha256(manifest_path), "inputs": inputs,
                  "versions": {"python": platform.python_version(), "numpy": np.__version__,
                               "pandas": pd.__version__, "scipy": scipy.__version__,
                               "matplotlib": matplotlib.__version__}}
    summaries = []
    print(f"Verified {len(inputs)} input file fingerprints.", flush=True)
    for fid, pair in sorted(pairs.items()):
        print(f"Inspecting {fid} ...", flush=True)
        s = inspect_flight(fid, pair, out)
        summaries.append(s)
        print(f"  {s['event_rows']:,} events; {s['candidate_timer_rollovers']} candidate timer wraps; "
              f"{s['rtc_backwards_steps_valid_date_pairs']:,} backwards RTC steps.", flush=True)
    # Detect accidental source changes during a run, as well as before it.
    if any(sha256(root / r["relative_path"]) != r["sha256"] for r in inputs):
        raise ValueError("A source file changed during diagnostics; results are incomplete")
    write_reports(out, summaries, provenance)
    print(f"\n{STATUS}")
    print(f"Report: {out / 'report.html'}")
    print("Original recordings are unchanged. No training table or model was created.")
    print("Keep RUN_TRAINING = False. See the report for the instrument details still needed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"DIAGNOSTICS STOPPED: {error}", file=sys.stderr)
        print("Keep the recordings unchanged and share this message.", file=sys.stderr)
        sys.exit(1)

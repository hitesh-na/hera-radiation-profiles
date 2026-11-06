"""Software fixture only. These records are NOT HERA measurements."""
from pathlib import Path
import numpy as np
import pandas as pd
from .io import write_json, sha256


def make_demo(output_dir, seed=12026):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "synthetic_flights.csv").exists():
        raise ValueError("Demo data already exist; choose a new directory")
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for j, top in enumerate([28500, 27000, 11500]):
        flight = f"SYNTHETIC_{chr(65+j)}"
        # Comparable dense sample spacing, different ascent and descent speeds.
        up = np.arange(500, top, 30.0)
        down = np.arange(top, 500, -45.0)
        altitude = np.r_[up, down]
        t = np.arange(len(altitude), dtype=float) * 5.0 + 2.5
        phase = np.r_[np.repeat("ascent", len(up)), np.repeat("descent", len(down))]
        center = 15800 + j*350
        amplitude = 25*(1+.025*j)
        rate = 2.8 + amplitude*np.exp(-.5*((altitude-center)/4800)**2)
        rate *= np.where(phase == "descent", 1.025, 1.0)
        live = rng.uniform(4.72, 4.95, len(t))
        counts = rng.poisson(rate*live)
        frame = pd.DataFrame({"flight_id": flight, "Timestamp_s": t, "phase": phase,
            "Counts_Per_Second": counts/live, "Raw_Counts": counts, "Live_Time_s": live,
            "Rate_Error": np.sqrt(counts)/live, "Altitude[m]": altitude,
            "Pressure[Pa]": 101325*np.exp(-altitude/(7600+20*j)),
            "Temperature[C]": np.maximum(15-.0065*altitude, -56.5)+rng.normal(0,.4,len(t)),
            "Accel": 9.8+rng.normal(0,.08,len(t)), "Mag": 48+rng.normal(0,.35,len(t)),
            "Latitude": 40+j*.05+t*.000004, "Longitude": -77+j*.08+t*.000008,
            "match_time_error_s": rng.uniform(0,.8,len(t)), "detector_id": "SIMULATED_SAME_RESPONSE",
            "source_kind": "synthetic"})
        frames.append(frame)
        truth.append({"flight_id": flight, "generating_peak_altitude_m": center,
                      "top_altitude_m": top, "expected_coverage": "bracketed" if top>center+2000 else "edge_limited",
                      "note": "Generator truth is for software checks only and is never a predictor."})
    table = pd.concat(frames, ignore_index=True)
    path = out / "synthetic_flights.csv"
    table.to_csv(path, index=False)
    write_json(out / "synthetic_truth.json", {"source_kind": "synthetic", "seed": seed, "flights": truth})
    write_json(out / "provenance.json", {"source_kind": "synthetic", "seed": seed,
        "data_sha256": sha256(path), "purpose": "Software demonstration only; no real HERA accuracy or Pfotzer result"})
    return path

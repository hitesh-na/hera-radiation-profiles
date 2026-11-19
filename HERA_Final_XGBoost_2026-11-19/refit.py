"""Reproduce the final five-flight HERA models from the pinned repository."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

CHANNELS = ("CW12", "CW123")
FEATURE = "Pressure_Pa"
SOURCE_FEATURE = "Pressure[Pa]"
FINAL_SEED = 42
VARIANT = "pressure_only_xgboost"
EXPLORATORY_DIR = "HERA_Exploratory_Training_2026-11-06"
REPO_COMMIT = "38644fcb0f97c08641bea0835234b8b6d864c669"
EXPLORATORY_HASHES = {
    "train_exploratory.py": "41ba103206d2f79e34eb7a0888fc16aba264ba05961e3a005f54654413635c61",
    "inputs/INPUT_HASHES.json": "aca992b1d65f5d2cb5ccbea86f1d57f950bc2e3aab3ad33b0b487c293cc55991",
    "results/held_out_predictions_per_seed.csv": "93b385b4f34cf9bb37c21bb2dd28118fb062d15573f10a1f901623687c4510e7",
}
PEAK_CODE_HASHES = {
    "pfotzer_xgb/peaks.py": "855fd2bd6e55f8bc839fd32e5f015a0d768467becde51b5051f42cf0f0b1464d",
    "pfotzer_xgb/profiles.py": "ea6aa5b39cfab733290b7ce9dc3ef5c2d1eb351bf251d0bb7f25cd498ba17403",
    "config/research.json": "cbc4f4aebe26b26d77ef1c172bf546b8856ccda61f8be6b3b4fcfea26befa606",
}


def verify_file(path, expected):
    data = path.read_bytes()
    # Git on Windows may check out tracked text with CRLF.
    normalized = data.replace(b"\r\n", b"\n")
    if expected not in {hashlib.sha256(b).hexdigest() for b in (data, normalized)}:
        raise ValueError(f"Fingerprint mismatch: {path}")


def load_trainer(path):
    spec = importlib.util.spec_from_file_location("hera_exploratory_trainer", path)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    return trainer


def reproduce_validation(trainer, data, pressure, protocol, saved, output):
    predictions = []
    metrics = []
    max_difference = 0.0
    for channel in CHANNELS:
        target = data[f"Rate_{channel}_cps"]
        for flight in protocol["independent_flight_groups"]:
            held_out = data.flight_id.eq(flight)
            train = ~held_out
            for seed in protocol["seeds"]:
                model = trainer.fit_model(
                    pressure.loc[train], target.loc[train], data.loc[train, "flight_id"],
                    protocol["base_parameters"], seed,
                )
                observed = data.loc[held_out, ["flight_id", "bin_index", "phase",
                                              "Altitude[m]", SOURCE_FEATURE]].copy()
                observed["observed_rate_cps"] = target.loc[held_out]
                observed["predicted_rate_cps"] = np.maximum(
                    model.predict(pressure.loc[held_out].to_numpy(float)), 0
                )
                matching = saved.loc[
                    saved.channel.eq(channel) & saved.outer_test_flight.eq(flight)
                    & saved.seed.eq(seed)
                ].set_index("bin_index")
                if not matching.flight_id.eq(flight).all() or len(matching) != len(observed):
                    raise ValueError("Saved predictions do not match the held-out cohort")
                expected = matching.loc[observed.bin_index, "prediction_cps"].to_numpy(np.float32)
                actual = observed.predicted_rate_cps.to_numpy(np.float32)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
                max_difference = max(max_difference, float(np.max(np.abs(actual - expected))))
                observed["channel"] = channel
                observed["seed"] = seed
                predictions.append(observed)
                metrics.append({"channel": channel, "flight_id": flight, "seed": seed,
                                "intervals": len(observed),
                                **trainer.score(observed.observed_rate_cps, actual)})
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(output / "held_out_predictions.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "held_out_metrics_by_seed.csv", index=False)
    equal_seed = metrics.groupby(["channel", "flight_id"])[trainer.METRICS].mean().reset_index()
    equal_seed.to_csv(output / "held_out_metrics_by_flight.csv", index=False)
    summary = pd.concat([
        equal_seed.groupby("channel")[trainer.METRICS].mean().assign(seeds="11,42,73"),
        metrics.loc[metrics.seed.eq(FINAL_SEED)].groupby("channel")[trainer.METRICS]
        .mean().assign(seeds=str(FINAL_SEED)),
    ]).reset_index()
    summary.to_csv(output / "held_out_metrics_summary.csv", index=False)
    return predictions, max_difference


def audit_peaks(predictions, config, output):
    from pfotzer_xgb.peaks import compare_peaks, screen_peak
    from pfotzer_xgb.profiles import altitude_profile, profile_errors

    rows = []
    profiles = []
    selected = predictions[predictions.seed.eq(FINAL_SEED)]
    for (channel, flight, phase), group in selected.groupby(["channel", "flight_id", "phase"]):
        group = group.rename(columns={"observed_rate_cps": "Counts_Per_Second"})
        group = group.assign(source_kind="real_hera")
        for width in config["sensitivity_bin_m"]:
            # Every recorded bin has the same elapsed exposure. The arithmetic
            # mean is equivalent to elapsed-time weighting. Detector live time
            # is unavailable, so do not populate the live-time/Poisson columns.
            profile = altitude_profile(group, width, config["min_samples_per_bin"],
                                       prediction_col="predicted_rate_cps")
            if width == config["altitude_bin_m"]:
                profiles.append(profile.assign(channel=channel))
            for smoothing in config["sensitivity_smoothing_bins"]:
                rules = {**config["peak_rules"], "smoothing_bins": smoothing}
                measured = screen_peak(profile, "rate_cps", rules)
                predicted = screen_peak(profile, "predicted_cps", rules)
                rows.append({"channel": channel, "flight_id": flight, "phase": phase,
                             "bin_m": width, "smoothing_bins": smoothing,
                             "primary_setting": width == config["altitude_bin_m"]
                             and smoothing == config["peak_rules"]["smoothing_bins"],
                             "retained_intervals": len(group),
                             "observed_min_altitude_m": group["Altitude[m]"].min(),
                             "observed_max_altitude_m": group["Altitude[m]"].max(),
                             "observed_screening_reason": measured["reason"],
                             "predicted_screening_reason": predicted["reason"],
                             **profile_errors(profile), **compare_peaks(measured, predicted)})
    pd.concat(profiles, ignore_index=True).to_csv(output / "primary_altitude_profiles.csv", index=False)
    audit = pd.DataFrame(rows)
    audit.to_csv(output / "peak_screening.csv", index=False)
    primary = audit[audit.primary_setting]
    return {
        "primary_total_pairs": len(primary),
        "primary_pairs_with_both_peaks": int(primary.both_identified.sum()),
        "sensitivity_pairs_with_both_peaks": int(audit.both_identified.sum()),
        "sensitivity_total_pairs": len(audit),
        "primary_measured_screening_reasons": primary.observed_screening_reason.value_counts().to_dict(),
        "recorded_rate_aggregation": "equal elapsed exposure; detector live time unavailable",
        "validation_scope": "retrospective screening on seed 42 held-out predictions",
    }


def fit_final(trainer, data, pressure, protocol, output):
    models = output / "models"
    models.mkdir()
    metadata = {}
    for channel in CHANNELS:
        model = trainer.fit_model(pressure, data[f"Rate_{channel}_cps"],
                                  data.flight_id, protocol["base_parameters"], FINAL_SEED)
        native = model.get_booster()
        native.set_attr(hera_channel=channel, hera_feature=FEATURE, hera_feature_unit="Pa",
                        hera_target_unit="recorded counts per elapsed second")
        path = models / f"{channel}_pressure_only.ubj"
        native.save_model(path)
        reloaded = xgb.Booster(params={"nthread": 1}, model_file=path)
        expected = model.predict(pressure.to_numpy(float))
        actual = reloaded.inplace_predict(pressure.to_numpy(float))
        np.testing.assert_array_equal(expected, actual)
        if not np.isfinite(actual).all():
            raise ValueError(f"Nonfinite training-input predictions for {channel}")
        metadata[channel] = {"file": f"models/{path.name}",
                             "sha256": trainer.digest(path),
                             "target_column": f"Rate_{channel}_cps",
                             "training_input_reload_max_difference_cps": 0.0}
    manifest = {
        "status": "final refit of the five-flight recorded-rate experiment",
        "source_repository": "https://github.com/hitesh-na/hera-radiation-profiles",
        "source_commit": REPO_COMMIT,
        "feature": FEATURE,
        "feature_unit": "Pa",
        "prediction_unit": "recorded counts per elapsed second",
        "variant": VARIANT, "seed": FINAL_SEED,
        "parameters": {"objective": "reg:squarederror", "tree_method": "hist",
                       **protocol["base_parameters"]},
        "training_pressure_min_pa": float(pressure[FEATURE].min()),
        "training_pressure_max_pa": float(pressure[FEATURE].max()),
        "training_flights": data.groupby("flight_id").size().to_dict(),
        "training_rows_per_channel": len(data), "window_seconds": 30,
        "training_weights": "equal total weight per flight",
        "training_verified": False,
        "independent_final_model_test_completed": False,
        "models": metadata, "xgboost_version": xgb.__version__,
        "source_input_hashes": EXPLORATORY_HASHES,
        "peak_screening_source_hashes": PEAK_CODE_HASHES,
        "variant_selection": "lowest mean held-out RMSE in the prior five-flight study for both targets",
    }
    trainer.write_json(output / "model_manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="Pinned HERA GitHub checkout")
    parser.add_argument("--exploratory", type=Path, help="Override the checkout's Run 1 folder")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    if xgb.__version__ != "2.1.4":
        parser.error("Use XGBoost 2.1.4 to reproduce the original study")
    repo = args.repo.resolve()
    source = (args.exploratory or repo / EXPLORATORY_DIR).resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be new or empty")
    for name, expected in EXPLORATORY_HASHES.items():
        verify_file(source / name, expected)
    for name, expected in PEAK_CODE_HASHES.items():
        verify_file(repo / name, expected)
    # The original trainer verifies the frozen input hashes byte for byte.
    # Git checkout can change line endings, so restore canonical LF text if needed.
    inputs = source / "inputs"
    hashes = json.loads((inputs / "INPUT_HASHES.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        verify_file(inputs / name, expected)
    trainer = load_trainer(source / "train_exploratory.py")
    out.mkdir(parents=True, exist_ok=True)
    if any(trainer.digest(inputs / name) != expected for name, expected in hashes.items()):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            canonical = Path(temporary)
            for name in ["INPUT_HASHES.json", *hashes]:
                raw = (inputs / name).read_bytes()
                (canonical / name).write_bytes(raw.replace(b"\r\n", b"\n"))
            data, protocol, _, _ = trainer.verify_inputs(canonical)
    else:
        data, protocol, _, _ = trainer.verify_inputs(inputs)
    pressure = data[[SOURCE_FEATURE]].rename(columns={SOURCE_FEATURE: FEATURE})
    if not np.isfinite(pressure[FEATURE]).all() or not pressure[FEATURE].gt(0).all():
        raise ValueError("Training pressures must be finite and positive")
    saved = pd.read_csv(source / "results/held_out_predictions_per_seed.csv")
    saved = saved[saved.variant.eq(VARIANT)]
    predictions, max_difference = reproduce_validation(trainer, data, pressure, protocol, saved, out)
    fit_final(trainer, data, pressure, protocol, out)
    sys.path.insert(0, str(repo))
    config = json.loads((repo / "config/research.json").read_text(encoding="utf-8"))
    checks = audit_peaks(predictions, config, out)
    checks.update({"reproduced_held_out_fits": 30,
                   "reproduced_held_out_predictions": len(predictions),
                   "max_held_out_prediction_difference_cps": max_difference,
                   "all_data_models_reloaded": len(CHANNELS)})
    trainer.write_json(out / "validation.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()

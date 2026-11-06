"""Nested flight-held-out evaluation. The outer test target never enters tuning."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, ParameterSampler
from xgboost import XGBRegressor
from .features import engineer, matrix, FEATURE_SETS
from .quality import audit_table
from .profiles import altitude_profile, profile_errors, overlap_table
from .peaks import screen_peak, compare_peaks
from .io import new_output, write_json, sha256, versions, slug

BASE = {"n_estimators": 180, "learning_rate": .05, "max_depth": 3,
        "min_child_weight": 5, "subsample": .85, "colsample_bytree": .9,
        "gamma": 0, "reg_alpha": 0, "reg_lambda": 1}
SPACE = {"n_estimators": [100, 180, 300, 450], "learning_rate": [.025, .05, .09],
         "max_depth": [2, 3, 4, 5], "min_child_weight": [3, 8, 20],
         "subsample": [.7, .85, 1.], "colsample_bytree": [.75, .9, 1.],
         "gamma": [0, .1, .5], "reg_alpha": [0, .1, 1.], "reg_lambda": [1., 5., 15.]}


def metrics(y, pred):
    return {"rmse_cps": float(np.sqrt(mean_squared_error(y, pred))),
            "mae_cps": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred)) if len(y)>1 and np.var(y)>0 else np.nan}


def flight_weights(frame):
    sizes = frame.groupby("flight_id").Timestamp_s.transform("size")
    return len(frame) / (frame.flight_id.nunique() * sizes.to_numpy(float))


def make_model(params, seed, jobs=2):
    return XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=jobs,
                        random_state=seed, importance_type="gain", **params)


def nonnegative_predict(model, x):
    return np.maximum(model.predict(x), 0)


def grouped_splits(frame):
    if frame.flight_id.nunique() < 2:
        raise ValueError("Grouped validation requires at least two independent flights")
    return list(LeaveOneGroupOut().split(frame, groups=frame.flight_id))


def tune(training, variant, seed, config, outer_flight):
    """Accept ONLY outer-training rows. Fixed tree counts; no test early stopping."""
    candidates = [dict(BASE)] + list(ParameterSampler(
        config.get("search_space", SPACE), n_iter=config["search_candidates"],
        random_state=config.get("search_seed", 2026)))
    x, y = matrix(training, variant, config.get("feature_profile", "standard")), training.Counts_Per_Second
    splits = grouped_splits(training)
    trials, scores = [], []
    for candidate_id, params in enumerate(candidates):
        fold_scores = []
        for itr, iva in splits:
            tr, va = training.iloc[itr], training.iloc[iva]
            assert set(tr.flight_id).isdisjoint(set(va.flight_id))
            assert outer_flight not in set(training.flight_id)
            model = make_model(params, seed, config.get("n_jobs", 2))
            model.fit(x.iloc[itr], y.iloc[itr], sample_weight=flight_weights(tr))
            score = metrics(y.iloc[iva], nonnegative_predict(model, x.iloc[iva]))["rmse_cps"]
            fold_scores.append(score)
            trials.append({"outer_flight": outer_flight, "seed": seed, "candidate": candidate_id,
                "inner_train_flights": "|".join(sorted(tr.flight_id.unique())),
                "inner_validation_flight": str(va.flight_id.iloc[0]), "rmse_cps": score,
                "params_json": json.dumps(params, sort_keys=True)})
        scores.append(float(np.mean(fold_scores)))
    best = int(np.argmin(scores))
    for row in trials:
        row["selected"] = row["candidate"] == best
        row["mean_inner_flight_rmse_cps"] = scores[row["candidate"]]
    return candidates[best], trials


def read_and_check(data_path, config, metadata_path=None):
    d = pd.read_csv(data_path, dtype={"flight_id": str})
    mode = config["data_mode"]
    if mode == "synthetic":
        if "source_kind" not in d or set(d.source_kind) != {"synthetic"}:
            raise ValueError("Synthetic mode only accepts explicitly labeled synthetic records")
        meta = {"source_kind": "synthetic", "warning": "NO REAL HERA RESULTS"}
    elif mode == "real_hera":
        if metadata_path is None:
            raise ValueError("Real data require an audited metadata JSON; see config/combined_metadata.example.json")
        if (config.get("feature_profile") != "hera_uncalibrated_v1" or
                config.get("feature_engineering", {}).get("temperature_context") != "inside_box"):
            raise ValueError("Use the reviewed real-HERA feature profile and inside_box temperature context")
        meta = json.loads(Path(metadata_path).read_text())
        checks = ["timestamp_seconds", "altitude_m", "pressure_pa", "temperature_c",
                  "cps_valid", "synchronization_reviewed", "detector_config_reviewed"]
        if meta.get("source_kind") != "real_hera" or any(meta.get("verified", {}).get(k) is not True for k in checks):
            raise ValueError("Complete all source/unit/clock/detector verification fields first")
        if meta.get("data_sha256") != sha256(data_path):
            raise ValueError("Metadata hash does not match the exact input CSV")
        if "source_kind" in d and not set(d.source_kind.dropna()).issubset({"real_hera"}):
            raise ValueError("Real-data run contains non-HERA source labels")
        d["source_kind"] = "real_hera"
    else:
        raise ValueError("data_mode must be synthetic or real_hera")
    clean, excluded, quality, flags = audit_table(d, **config.get("quality", {}))
    clean = engineer(clean, **config.get("feature_engineering", {}))
    phases = config.get("phases", ["ascent", "descent"])
    phase_exclusions = clean[~clean.phase.isin(phases)].copy()
    phase_exclusions["exclusion_reason"] = "phase_not_in_training_protocol"
    excluded = pd.concat([excluded, phase_exclusions], ignore_index=True)
    clean = clean[clean.phase.isin(phases)].reset_index(drop=True)
    if clean.flight_id.nunique() < 3:
        raise ValueError("Nested flight validation needs at least THREE usable independent flights")
    if (clean.groupby("flight_id").size() < config.get("min_rows_per_flight", 30)).any():
        raise ValueError("A flight has too few valid rows for this protocol")
    return clean, excluded, quality, flags, meta


def circular_importance(model, test, variant, seed, repeats=3, feature_profile="standard"):
    """Descriptive group perturbation, not causal and never used for selection.
    Shift related columns together within phase, preserving their trajectories.
    """
    x = matrix(test, variant, feature_profile)
    base_score = metrics(test.Counts_Per_Second, nonnegative_predict(model, x))["rmse_cps"]
    groups = {"altitude_pressure_density": ["altitude_m", "pressure_pa", "log_pressure", "density_proxy_kg_m3"],
              "temperature": ["temperature_c"], "inside_box_temperature": ["box_temperature_c"],
              "motion": ["acceleration", "vertical_speed_m_s", "pressure_change_pa_s"],
              "magnetic": ["magnetic_field"], "position": ["latitude", "longitude"]}
    rng = np.random.default_rng(seed)
    result = []
    for name, columns in groups.items():
        cols = [c for c in columns if c in x]
        if not cols:
            continue
        for repeat in range(repeats):
            changed = x.copy()
            for _, g in test.groupby("phase", sort=False):
                positions = test.index.get_indexer(g.index)
                if len(positions) < 4:
                    continue
                shift = int(rng.integers(max(1,len(positions)//4), max(2,3*len(positions)//4)))
                changed.iloc[positions, changed.columns.get_indexer(cols)] = np.roll(x.iloc[positions][cols].to_numpy(), shift, axis=0)
            score = metrics(test.Counts_Per_Second, nonnegative_predict(model, changed))["rmse_cps"]
            result.append({"feature_group": name, "repeat": repeat, "delta_rmse_cps": score-base_score})
    return result


def measured_products(data, out, config):
    profiles, peaks, sensitivity = [], [], []
    for (flight, phase), g in data.groupby(["flight_id", "phase"]):
        for bin_m in config["sensitivity_bin_m"]:
            p = altitude_profile(g, bin_m, config["min_samples_per_bin"])
            for window in config["sensitivity_smoothing_bins"]:
                peak = screen_peak(p, rules={**config["peak_rules"], "smoothing_bins": window})
                sensitivity.append({"flight_id": flight, "phase": phase, "kind": "measured", **peak})
        p = altitude_profile(g, config["altitude_bin_m"], config["min_samples_per_bin"])
        profiles.append(p)
        peaks.append({"flight_id": flight, "phase": phase, **screen_peak(p, rules=config["peak_rules"])})
    profiles = pd.concat(profiles, ignore_index=True)
    profiles.to_csv(out / "profile_tables/measured_profiles.csv", index=False)
    pd.DataFrame(peaks).to_csv(out / "peak_tables/measured_peaks.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(out / "peak_tables/measured_sensitivity.csv", index=False)
    overlap_table(profiles).to_csv(out / "profile_tables/common_supported_overlap.csv", index=False)
    return profiles, pd.DataFrame(peaks)


def run(data_path, output_dir, config, metadata_path=None):
    data, excluded, quality, flags, metadata = read_and_check(data_path, config, metadata_path)
    out = new_output(output_dir)
    for name in ["quality_reports", "cv_metrics", "profile_tables", "peak_tables", "figures", "model_artifacts"]:
        (out/name).mkdir()
    write_json(out/"run_config.json", config)
    write_json(out/"provenance.json", {"source_kind": config["data_mode"], "input_sha256": sha256(data_path),
               "versions": versions(), "metadata": metadata,
               "flight_ids": sorted(data.flight_id.unique()),
               "early_stopping": "disabled: n_estimators selected only by inner grouped validation"})
    data.to_csv(out/"quality_reports/validated_features.csv", index=False)
    excluded.to_csv(out/"quality_reports/excluded_rows.csv", index=False)
    quality.to_csv(out/"quality_reports/flight_quality.csv", index=False)
    flags.to_csv(out/"quality_reports/sensor_missingness.csv", index=False)
    measured, measured_peaks = measured_products(data, out, config)
    # Measured products are made before fitting, and are NEVER passed to tune().
    all_metrics, predictions, histories, splits, peakrows, profiles, importances, permutations, sensitivities = [], [], [], [], [], [], [], [], []
    for train_idx, test_idx in grouped_splits(data):
        train, test = data.iloc[train_idx].copy(), data.iloc[test_idx].copy()
        heldout = str(test.flight_id.iloc[0])
        lo, hi = train["Altitude[m]"].min(), train["Altitude[m]"].max()
        assert set(train.flight_id).isdisjoint(set(test.flight_id))
        for seed in config["seeds"]:
            best, trials = tune(train, "tuned_engineered", seed, config, heldout)
            histories.extend(trials)
            for variant in config["variants"]:
                params = best if variant == "tuned_engineered" else BASE
                model = make_model(params, seed, config.get("n_jobs", 2))
                feature_profile = config.get("feature_profile", "standard")
                xtr, xte = matrix(train, variant, feature_profile), matrix(test, variant, feature_profile)
                model.fit(xtr, train.Counts_Per_Second, sample_weight=flight_weights(train))
                raw_prediction = model.predict(xte)
                prediction = np.maximum(raw_prediction, 0)
                stamp = {"flight_id": heldout, "seed": seed, "variant": variant, "source_kind": config["data_mode"]}
                test_prediction = test.copy()
                test_prediction["prediction_cps"] = prediction
                test_prediction["outside_training_altitude"] = ~test["Altitude[m]"].between(lo, hi)
                predcols = ["source_row", "flight_id", "Timestamp_s", "phase", "Altitude[m]", "Counts_Per_Second",
                            "prediction_cps", "outside_training_altitude"]
                predictions.append(test_prediction[predcols].assign(seed=seed, variant=variant, source_kind=config["data_mode"]))
                row = {**stamp, **metrics(test.Counts_Per_Second, prediction), "n_test": len(test),
                       "n_train": len(train), "train_altitude_min_m": lo, "train_altitude_max_m": hi,
                       "outside_training_altitude_fraction": test_prediction.outside_training_altitude.mean(),
                       "negative_raw_prediction_fraction": float(np.mean(raw_prediction < 0))}
                all_metrics.append(row)
                splits.append({**stamp, "train_flights": "|".join(sorted(train.flight_id.unique())), "n_train": len(train), "n_test": len(test)})
                prefix = f"{slug(heldout)}__{variant}__seed{seed}"
                model.save_model(out/f"model_artifacts/{prefix}.json")
                write_json(out/f"model_artifacts/{prefix}.metadata.json", {**stamp,
                    "train_flights": sorted(train.flight_id.unique()), "features": list(xtr.columns),
                    "hyperparameters": model.get_params(), "input_sha256": sha256(data_path),
                    "predictor_setting": "offline_completed_flight", "clip_predictions_below_zero": True})
                for feature, value in zip(xtr.columns, model.feature_importances_):
                    importances.append({**stamp, "feature": feature, "normalized_gain": float(value)})
                if variant == "tuned_engineered":
                    for item in circular_importance(model, test, variant, seed, config.get("permutation_repeats", 3), feature_profile):
                        permutations.append({**stamp, **item})
                for phase, g in test_prediction.groupby("phase"):
                    p = altitude_profile(g, config["altitude_bin_m"], config["min_samples_per_bin"], "prediction_cps")
                    profiles.append(p.assign(seed=seed, variant=variant))
                    mp = screen_peak(p, rules=config["peak_rules"])
                    pp = screen_peak(p, "predicted_cps", config["peak_rules"])
                    if pp["status"] == "identified" and not lo <= pp["peak_altitude_m"] <= hi:
                        pp.update(status="unresolved", reason="candidate_outside_training_altitude_range",
                                  peak_altitude_m=None, peak_altitude_km=None, peak_cps=None)
                    peakrows.append({**stamp, "phase": phase, **compare_peaks(mp, pp),
                        "measured_reason": mp["reason"], "predicted_reason": pp["reason"],
                        "measured_peak_cps": mp["peak_cps"], "predicted_peak_cps": pp["peak_cps"],
                        **profile_errors(p), **{f"phase_{k}": v for k,v in metrics(g.Counts_Per_Second, g.prediction_cps).items()}})
                    if variant == "tuned_engineered":
                        for width in config["sensitivity_bin_m"]:
                            sp = altitude_profile(g, width, config["min_samples_per_bin"], "prediction_cps")
                            for window in config["sensitivity_smoothing_bins"]:
                                pr = screen_peak(sp, "predicted_cps", {**config["peak_rules"], "smoothing_bins": window})
                                if pr["status"] == "identified" and not lo <= pr["peak_altitude_m"] <= hi:
                                    pr.update(status="unresolved", reason="candidate_outside_training_altitude_range",
                                              peak_altitude_m=None, peak_altitude_km=None, peak_cps=None)
                                sensitivities.append({**stamp, "phase": phase, **pr})
                print(f"{config['data_mode']} | held out {heldout} | seed {seed} | {variant} | RMSE {row['rmse_cps']:.3f} CPS", flush=True)
    metric_df, peak_df = pd.DataFrame(all_metrics), pd.DataFrame(peakrows)
    metric_df.to_csv(out/"cv_metrics/per_flight_seed.csv", index=False)
    pd.DataFrame(histories).to_csv(out/"cv_metrics/hyperparameter_search.csv", index=False)
    pd.DataFrame(splits).to_csv(out/"cv_metrics/split_audit.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(out/"cv_metrics/heldout_predictions.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(out/"profile_tables/heldout_profiles.csv", index=False)
    peak_df.to_csv(out/"peak_tables/heldout_peaks_and_profile_metrics.csv", index=False)
    pd.DataFrame(importances).to_csv(out/"cv_metrics/feature_gain.csv", index=False)
    pd.DataFrame(permutations).to_csv(out/"cv_metrics/grouped_circular_importance.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(out/"peak_tables/predicted_sensitivity.csv", index=False)
    # Seeds are repeated fits, NOT extra independent flights.
    seed_stats = metric_df.groupby(["variant", "flight_id"])[["rmse_cps", "mae_cps", "r2"]].agg(["mean", "std", "min", "max"])
    seed_stats.columns = ["_".join(c) for c in seed_stats.columns]
    seed_stats.reset_index().to_csv(out/"cv_metrics/seed_variability.csv", index=False)
    flight_means = metric_df.groupby(["variant", "flight_id"])[["rmse_cps", "mae_cps", "r2"]].mean()
    summary = flight_means.groupby("variant").agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["independent_flights"] = data.flight_id.nunique()
    summary.to_csv(out/"cv_metrics/summary_across_flights.csv", index=False)
    base = metric_df[metric_df.variant == "baseline"][["flight_id", "seed", "rmse_cps", "mae_cps"]]
    pairs = metric_df.merge(base, on=["flight_id", "seed"], suffixes=("", "_baseline"))
    pairs["delta_rmse_vs_baseline"] = pairs.rmse_cps - pairs.rmse_cps_baseline
    pairs["delta_mae_vs_baseline"] = pairs.mae_cps - pairs.mae_cps_baseline
    pairs.to_csv(out/"cv_metrics/paired_changes_vs_baseline.csv", index=False)
    from .plotting import make_figures, write_run_report
    make_figures(data, measured, measured_peaks, out, config)
    write_run_report(out, config, summary, peak_df, data)
    return out


def fit_final(data_path, output_dir, config, metadata_path=None, seed=42):
    """Refit a predeclared variant for FUTURE flights; this is not test evidence."""
    data, _, _, _, metadata = read_and_check(data_path, config, metadata_path)
    out = new_output(output_dir)
    params, history = tune(data, "tuned_engineered", seed, config, "FINAL_REFIT_NO_OUTER_TEST")
    x = matrix(data, "tuned_engineered", config.get("feature_profile", "standard"))
    model = make_model(params, seed, config.get("n_jobs", 2))
    model.fit(x, data.Counts_Per_Second, sample_weight=flight_weights(data))
    model.save_model(out/"model.json")
    pd.DataFrame(history).to_csv(out/"grouped_search.csv", index=False)
    write_json(out/"metadata.json", {"source_kind": config["data_mode"], "seed": seed, "variant": "tuned_engineered",
        "training_flights": sorted(data.flight_id.unique()), "features": list(x.columns),
        "hyperparameters": model.get_params(), "feature_engineering": config.get("feature_engineering", {}),
        "feature_profile": config.get("feature_profile", "standard"),
        "training_altitude_min_m": data["Altitude[m]"].min(), "training_altitude_max_m": data["Altitude[m]"].max(),
        "input_sha256": sha256(data_path), "versions": versions(), "source_metadata": metadata,
        "warning": "No independent test score for this refit. Use outer CV results for evaluation."})
    return out


def predict_new(model_dir, data_path, output_path):
    folder = Path(model_dir)
    meta = json.loads((folder/"metadata.json").read_text())
    d = pd.read_csv(data_path, dtype={"flight_id": str})
    required = {"flight_id", "Timestamp_s", "Altitude[m]"}
    if not required.issubset(d.columns):
        raise ValueError(f"Prediction needs {sorted(required)} and verified sensor units")
    if set(d.flight_id).intersection(meta["training_flights"]):
        raise ValueError("Future-flight prediction expects flight IDs absent from the final training set")
    if not np.isfinite(d[["Timestamp_s", "Altitude[m]"]].to_numpy(float)).all():
        raise ValueError("Finite time and altitude required")
    if meta["source_kind"] == "synthetic" and ("source_kind" not in d or set(d.source_kind) != {"synthetic"}):
        raise ValueError("Synthetic model may only be used on labeled synthetic fixtures")
    d = engineer(d, **meta["feature_engineering"])
    model = XGBRegressor()
    model.load_model(folder/"model.json")
    d["prediction_cps"] = nonnegative_predict(model, d[meta["features"]])
    d["outside_training_altitude"] = ~d["Altitude[m]"].between(meta["training_altitude_min_m"], meta["training_altitude_max_m"])
    d["model_source_kind"] = meta["source_kind"]
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Prediction output already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(output_path, index=False)
    return output_path

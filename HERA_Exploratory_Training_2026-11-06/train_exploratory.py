"""HERA recorded-rate study, following the 23 October frozen grouped protocol.

This separate entry point accepts screened, explicitly UNVERIFIED observations
only for an exploratory study. It does not alter the original training gate,
calibration status, original notebooks, data, or Google Drive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
FEATURE_MAP = {
    'Altitude_m': 'Altitude[m]',
    'Pressure_Pa': 'Pressure[Pa]',
    'Box_Temperature_C': 'Box_Temperature[C]',
    'Latitude_deg': 'Latitude',
    'Longitude_deg': 'Longitude',
    'Vertical_speed_m_s': 'sensor_only_vertical_speed',
    'Log_pressure': 'sensor_only_log_pressure',
    'Log_pressure_change_per_s': 'sensor_only_pressure_change',
}
VARIANTS = {
    'training_flight_balanced_constant_mean': [],
    'altitude_only_xgboost': ['Altitude_m'],
    'pressure_only_xgboost': ['Pressure_Pa'],
    'altitude_pressure_xgboost': ['Altitude_m', 'Pressure_Pa'],
    'restricted_sensor_xgboost': list(FEATURE_MAP),
    'restricted_sensor_tuned_xgboost': list(FEATURE_MAP),
}
METRICS = ['r2', 'mse_cps2', 'rmse_cps', 'mae_cps']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    def clean(v):
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, np.generic):
            return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, Path):
            return str(v)
        return v
    Path(path).write_text(json.dumps(clean(obj), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def equal_flight_weights(groups):
    """Equal total weight per flight, normalized to mean row weight one."""
    groups = pd.Series(np.asarray(groups))
    counts = groups.value_counts()
    return (len(groups) / (len(counts) * groups.map(counts))).to_numpy(float)


def feature_frame(data):
    """Offline sensor features; gradients never cross excluded-window gaps.

    Do not reuse preparation gradients, which were computed before all window
    exclusions. Recompute them on contiguous RETAINED runs only. A centered
    three-bin sensor median is permitted for this completed-flight analysis;
    these features are not advertised as real-time/causal forecasts.
    """
    f = pd.DataFrame(index=data.index)
    for alias, original in list(FEATURE_MAP.items())[:5]:
        f[alias] = data[original].to_numpy(float)
    f['Log_pressure'] = np.log(data['Pressure[Pa]'].to_numpy(float))
    f['Vertical_speed_m_s'] = np.nan
    f['Log_pressure_change_per_s'] = np.nan
    for _, group in data.groupby('flight_id', sort=True):
        g = group.sort_values('bin_index')
        runs = g.bin_index.diff().ne(1).cumsum()
        for _, run in g.groupby(runs, sort=False):
            if len(run) < 3:
                continue
            altitude = run['Altitude[m]'].rolling(3, center=True, min_periods=2).median()
            pressure = run['Pressure[Pa]'].rolling(3, center=True, min_periods=2).median()
            t = run['Timestamp_s'].to_numpy(float)
            f.loc[run.index, 'Vertical_speed_m_s'] = np.gradient(altitude.to_numpy(), t)
            f.loc[run.index, 'Log_pressure_change_per_s'] = np.gradient(np.log(pressure.to_numpy()), t)
    return f[list(FEATURE_MAP)]


def score(y, predicted):
    y = np.asarray(y, float)
    p = np.asarray(predicted, float)
    if not len(y):
        return {m: np.nan for m in METRICS}
    mse = mean_squared_error(y, p)
    # Undefined R2 stays undefined, rather than being replaced with zero or one.
    r2 = float(r2_score(y, p, force_finite=False)) if len(y) > 1 and np.var(y) > 0 else np.nan
    return {'r2': r2, 'mse_cps2': float(mse), 'rmse_cps': float(np.sqrt(mse)),
            'mae_cps': float(mean_absolute_error(y, p))}


def fit_model(features, target, groups, parameters, seed, threads=1):
    model = xgb.XGBRegressor(
        objective='reg:squarederror', tree_method='hist', n_jobs=threads,
        random_state=int(seed), verbosity=0, **parameters,
    )
    # Arrays avoid illegal XGBoost feature names containing brackets.
    model.fit(features.to_numpy(float), np.asarray(target, float),
              sample_weight=equal_flight_weights(groups))
    return model


def contribution_values(model, features):
    a = features.to_numpy(float)
    matrix = xgb.DMatrix(a, nthread=1)
    contributions = model.get_booster().predict(matrix, pred_contribs=True, approx_contribs=False)
    raw = model.get_booster().predict(matrix, output_margin=True)
    error = float(np.max(np.abs(contributions.sum(axis=1) - raw)))
    if not np.allclose(contributions.sum(axis=1), raw, rtol=1e-5, atol=1e-5):
        raise AssertionError(f'SHAP additivity failed: {error}')
    return contributions, raw, error


def verify_inputs(input_dir):
    hashes = json.loads((input_dir / 'INPUT_HASHES.json').read_text())
    for name, expected in hashes.items():
        if digest(input_dir / name) != expected:
            raise ValueError(f'Input fingerprint mismatch: {name}')
    data = pd.read_csv(input_dir / 'screened_30s_candidates_NOT_VERIFIED.csv')
    protocol = json.loads((input_dir / 'frozen_evaluation_protocol.json').read_text())
    split_plan = pd.read_csv(input_dir / 'planned_group_splits.csv')
    if len(data) != 918 or data.duplicated(['flight_id', 'bin_index']).any():
        raise ValueError('Expected 918 distinct original screened intervals')
    if not data.screening_pass.all() or data.training_verified.any():
        raise ValueError('This entry point expects the original screened, UNVERIFIED cohort')
    if not data.source_kind.eq('real_hera').all():
        raise ValueError('Real HERA source labels required')
    if sorted(data.flight_id.unique()) != protocol['independent_flight_groups']:
        raise ValueError('Flight groups differ from frozen protocol')
    for channel in ['CW12', 'CW123']:
        if not np.allclose(data[f'Rate_{channel}_cps'], data[f'Event_Count_{channel}'] / data.Elapsed_Exposure_s):
            raise ValueError('Recorded-rate definition mismatch')
    if not np.allclose(data.Elapsed_Exposure_s, 30):
        raise ValueError('Non-30-second observation')
    for row in split_plan.itertuples():
        outer_train = set(row.outer_train_flights.split('|'))
        inner_train = set(row.inner_train_flights.split('|'))
        if row.outer_test_flight in outer_train | inner_train:
            raise ValueError('Outer-flight leakage in restored split plan')
        if row.inner_validation_flight in inner_train or not inner_train.issubset(outer_train):
            raise ValueError('Inner-flight leakage in restored split plan')
    return data, protocol, split_plan, hashes


def select_candidate(candidate_id, parameters, channel, outer, data, features, protocol, split_plan, threads):
    """Only access inner training/validation targets; never outer-test targets."""
    rows = []
    folds = split_plan[split_plan.outer_test_flight.eq(outer)]
    for split in folds.itertuples():
        train_ids = split.inner_train_flights.split('|')
        train = data.flight_id.isin(train_ids)
        validation = data.flight_id.eq(split.inner_validation_flight)
        assert outer not in set(data.loc[train | validation, 'flight_id'])
        for seed in protocol['seeds']:
            model = fit_model(features.loc[train], data.loc[train, f'Rate_{channel}_cps'],
                              data.loc[train, 'flight_id'], parameters, seed, threads)
            predicted = np.maximum(model.predict(features.loc[validation].to_numpy(float)), 0)
            rows.append({'channel': channel, 'outer_test_flight': outer, 'candidate_id': candidate_id,
                         'inner_train_flights': split.inner_train_flights,
                         'inner_validation_flight': split.inner_validation_flight, 'seed': seed,
                         **score(data.loc[validation, f'Rate_{channel}_cps'], predicted)})
    # Four inner flights, three seeds per flight, with equal flight weight.
    frame = pd.DataFrame(rows)
    objective = float(frame.groupby('inner_validation_flight').rmse_cps.mean().mean())
    return candidate_id, objective, rows


def altitude_support_mask(train_data, test_data):
    """Within each phase's training min/max altitude span, not a density test."""
    support = np.zeros(len(test_data), bool)
    for phase, g in test_data.groupby('phase', sort=False):
        t = train_data[train_data.phase.eq(phase)]
        if not len(t):
            continue
        mask = test_data.phase.eq(phase).to_numpy()
        h = test_data.loc[mask, 'Altitude[m]']
        support[mask] = h.between(t['Altitude[m]'].min(), t['Altitude[m]'].max()).to_numpy()
    return support


def run(args):
    started = time.perf_counter()
    inputs = args.inputs.resolve()
    data, protocol, split_plan, hashes = verify_inputs(inputs)
    features = feature_frame(data)
    if not np.isfinite(features[['Altitude_m', 'Pressure_Pa']]).all().all():
        raise ValueError('Missing primary predictor')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be new or empty; existing results are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'models').mkdir()
    sampled = list(ParameterSampler(protocol['tuning_space'],
                                   n_iter=protocol['tuning_random_candidates'],
                                   random_state=protocol['tuning_search_seed']))
    write_json(out / 'run_plan_before_fitting.json', {
        'study_status': 'EXPLORATORY_UNVERIFIED_RECORDED_RATES', 'input_hashes': hashes,
        'script_sha256': digest(__file__), 'frozen_protocol_sha256': digest(inputs / 'frozen_evaluation_protocol.json'),
        'variants': VARIANTS, 'feature_map': FEATURE_MAP, 'candidate_parameters': sampled,
        'seeds': protocol['seeds'], 'inner_selection': protocol['tuning_selection_metric'],
        'prediction_policy': protocol['prediction_policy'], 'sample_weights': protocol['sample_weights'],
        'aggregation': protocol['aggregation'], 'workers': args.workers, 'threads_per_model': args.threads,
        'feature_derivatives': 'centered 3-bin sensor median and gradient within contiguous retained runs; no gap bridging',
        'permission': 'User requested exploratory training; original verification metadata and notebook training gate remain unchanged',
        'original_base_model_notebook': 'not supplied; recovered code is preparation code plus a frozen modeling protocol',
    })
    pd.concat([data[['source_row', 'flight_id', 'bin_index']], features], axis=1).to_csv(out / 'model_features.csv', index=False)
    predictions = []
    shap_frames = []
    scores = []
    tuning = []
    chosen = []
    validations = []
    folds = protocol['independent_flight_groups']
    for channel in ['CW12', 'CW123']:
        for outer in folds:
            print(f'{channel}: tuning with {outer} held out', flush=True)
            outer_train = data[~data.flight_id.eq(outer)]
            outer_test = data[data.flight_id.eq(outer)]
            # This assertion protects both channels and phases of the outer flight.
            assert set(outer_train.flight_id).isdisjoint(set(outer_test.flight_id))
            overlap = altitude_support_mask(outer_train, outer_test)
            objectives = []
            restricted = features[VARIANTS['restricted_sensor_tuned_xgboost']]
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                tasks = [executor.submit(select_candidate, i, p, channel, outer, data, restricted,
                                         protocol, split_plan, args.threads) for i, p in enumerate(sampled)]
                for k, task in enumerate(as_completed(tasks), 1):
                    i, objective, rows = task.result()
                    objectives.append((objective, i))
                    tuning.extend(rows)
                    if k % 8 == 0:
                        print(f'  {k}/24 candidates evaluated', flush=True)
            _, best_id = min(objectives)
            best_parameters = sampled[best_id]
            chosen.append({'channel': channel, 'outer_test_flight': outer,
                           'candidate_id': best_id, 'inner_macro_rmse_cps': min(objectives)[0],
                           'parameters': best_parameters})
            for variant, columns in VARIANTS.items():
                for seed in protocol['seeds']:
                    target_name = f'Rate_{channel}_cps'
                    train_mask = data.index.isin(outer_train.index)
                    test_mask = data.index.isin(outer_test.index)
                    metadata = {'channel': channel, 'variant': variant, 'outer_test_flight': outer,
                                'seed': seed, 'train_flights': '|'.join(sorted(outer_train.flight_id.unique()))}
                    model_name = f'{channel}__{variant}__test_{outer}__seed_{seed}'
                    if not columns:
                        constant = outer_train.groupby('flight_id')[target_name].mean().mean()
                        raw = np.full(len(outer_test), constant)
                        write_json(out / 'models' / f'{model_name}.json', {'constant_rate_cps': constant, **metadata})
                    else:
                        parameters = best_parameters if variant.endswith('_tuned_xgboost') else protocol['base_parameters']
                        model = fit_model(features.loc[train_mask, columns], data.loc[train_mask, target_name],
                                          data.loc[train_mask, 'flight_id'], parameters, seed, args.threads)
                        model.get_booster().save_model(out / 'models' / f'{model_name}.ubj')
                        contrib, raw, additive_error = contribution_values(model, features.loc[test_mask, columns])
                        saved = xgb.Booster(params={'nthread': args.threads},
                                            model_file=out / 'models' / f'{model_name}.ubj')
                        reloaded = saved.predict(xgb.DMatrix(features.loc[test_mask, columns].to_numpy(float), nthread=1),
                                                 output_margin=True)
                        np.testing.assert_allclose(raw, reloaded, rtol=1e-6, atol=1e-6)
                        validations.append({**metadata, 'model_reload_prediction_max_error_cps': float(np.max(abs(raw-reloaded))),
                                            'shap_additivity_max_error_cps': additive_error})
                        write_json(out / 'models' / f'{model_name}.metadata.json', {
                            **metadata, 'features': columns, 'parameters': parameters,
                            'status': 'EXPLORATORY_NOT_CALIBRATED_PHYSICAL_FLUX',
                            'negative_prediction_policy': 'clip to zero; SHAP values explain raw output',
                        })
                        sf = pd.DataFrame(contrib[:, :-1], columns=['shap_' + c for c in columns])
                        sf['bias_cps'] = contrib[:, -1]
                        sf['source_row'] = outer_test.source_row.to_numpy()
                        sf['raw_prediction_cps'] = raw
                        for k, v in metadata.items():
                            sf[k] = v
                        shap_frames.append(sf)
                    predicted = np.maximum(raw, 0)
                    for subset, mask in [('all', np.ones(len(outer_test), bool)), ('altitude_overlap', overlap)]:
                        scores.append({**metadata, 'subset': subset, 'n_bins': int(mask.sum()),
                                       **score(outer_test.loc[mask, target_name], predicted[mask])})
                    for phase, group in outer_test.groupby('phase'):
                        mask = outer_test.index.isin(group.index)
                        scores.append({**metadata, 'subset': phase, 'n_bins': int(mask.sum()),
                                       **score(group[target_name], predicted[mask])})
                    pred = outer_test[['source_row', 'flight_id', 'bin_index', 'phase', 'Altitude[m]', 'Pressure[Pa]', target_name]].copy()
                    pred = pred.rename(columns={target_name: 'observed_rate_cps'})
                    pred['raw_prediction_cps'] = raw
                    pred['prediction_cps'] = predicted
                    pred['altitude_in_training_phase_span'] = overlap
                    for k, v in metadata.items():
                        pred[k] = v
                    predictions.append(pred)
            # Checkpoint numerical results after each completed channel/flight fold.
            pd.DataFrame(scores).to_csv(out / 'metrics_per_flight_seed.csv', index=False)
            pd.DataFrame(tuning).to_csv(out / 'inner_tuning_evaluations.csv', index=False)
            write_json(out / 'chosen_parameters_by_outer_fold.json', chosen)
            print(f'Completed {channel} / {outer}', flush=True)
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(out / 'held_out_predictions_per_seed.csv', index=False)
    shap_values = pd.concat(shap_frames, ignore_index=True)
    shap_values.to_csv(out / 'held_out_shap_per_seed.csv', index=False)
    metric_frame = pd.DataFrame(scores)
    per_flight = metric_frame.groupby(['channel', 'variant', 'outer_test_flight', 'subset'], as_index=False)[METRICS].mean()
    per_flight.to_csv(out / 'metrics_seed_mean_per_flight.csv', index=False)
    summary = per_flight.groupby(['channel', 'variant', 'subset'])[METRICS].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out / 'metrics_equal_flight_summary.csv', index=False)
    seed_sd = metric_frame.groupby(['channel', 'variant', 'outer_test_flight', 'subset'])[METRICS].std().reset_index()
    seed_sd.to_csv(out / 'metrics_seed_standard_deviation.csv', index=False)
    keys = ['channel', 'variant', 'source_row', 'flight_id', 'bin_index', 'phase', 'Altitude[m]', 'Pressure[Pa]',
            'observed_rate_cps', 'altitude_in_training_phase_span']
    ensemble = pred.groupby(keys, as_index=False, dropna=False)[['prediction_cps', 'raw_prediction_cps']].mean()
    ensemble.to_csv(out / 'held_out_predictions_seed_ensemble.csv', index=False)
    ensemble_scores = []
    for (channel, variant, flight), group in ensemble.groupby(['channel', 'variant', 'flight_id']):
        ensemble_scores.append({'channel': channel, 'variant': variant, 'flight_id': flight,
                                **score(group.observed_rate_cps, group.prediction_cps)})
    pd.DataFrame(ensemble_scores).to_csv(out / 'secondary_seed_ensemble_metrics.csv', index=False)
    for (channel, variant, seed), group in pred.groupby(['channel', 'variant', 'seed']):
        if len(group) != len(data) or group.source_row.nunique() != len(data):
            raise AssertionError('Every interval must receive exactly one outer-test prediction per model/seed')
    packages = {}
    for package in ['numpy', 'pandas', 'scikit-learn', 'xgboost-cpu', 'xgboost', 'matplotlib']:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    validation = pd.DataFrame(validations)
    validation.to_csv(out / 'saved_model_and_shap_checks.csv', index=False)
    write_json(out / 'run_summary.json', {
        'status': 'COMPLETED_EXPLORATORY_REAL_DATA_STUDY', 'training_performed': True,
        'source_kind': 'real_hera', 'training_verified': False,
        'independent_flights': len(folds), 'distinct_30_second_intervals': len(data),
        'channels': ['CW12', 'CW123'], 'variants_per_channel': len(VARIANTS),
        'xgboost_variants_per_channel': 5, 'baseline_variants_per_channel': 1,
        'outer_xgboost_models': len(validations), 'inner_tuning_fits': len(tuning),
        'each_interval_outer_test_predictions_per_model_seed': 1,
        'max_shap_additivity_error_cps': validation.shap_additivity_max_error_cps.max(),
        'max_saved_model_reload_error_cps': validation.model_reload_prediction_max_error_cps.max(),
        'negative_raw_predictions': int((pred.raw_prediction_cps < 0).sum()),
        'elapsed_seconds_here': time.perf_counter() - started,
        'execution_environment_here_not_users_computer': {'python': platform.python_version(), 'packages': packages},
        'input_files_still_match': all(digest(inputs / n) == h for n, h in hashes.items()),
        'full_five_flight_production_model_fitted': False,
        'limitations': protocol['remaining_limits'],
    })
    print('Finished exploratory training. No production model was selected from test scores.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--exploratory', action='store_true', help='Explicitly acknowledge the supplied unverified recorded-rate cohort')
    args = parser.parse_args()
    if not args.exploratory:
        parser.error('Use --exploratory for the separately authorized, unverified recorded-rate study; original training flags stay unchanged')
    if args.workers < 1 or args.threads < 1:
        parser.error('Worker/thread counts must be positive')
    run(args)

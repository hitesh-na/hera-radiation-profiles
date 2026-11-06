"""Render the saved exploratory results; this script never fits a model."""
from pathlib import Path
import base64
import html
import io
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from train_exploratory import VARIANTS, FEATURE_MAP

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'
NAMES = {
    'training_flight_balanced_constant_mean': 'Constant baseline',
    'altitude_only_xgboost': 'Altitude only',
    'pressure_only_xgboost': 'Pressure only',
    'altitude_pressure_xgboost': 'Altitude + pressure',
    'restricted_sensor_xgboost': 'Restricted sensors',
    'restricted_sensor_tuned_xgboost': 'Restricted sensors, tuned',
}
LABELS = {
    'Altitude_m': 'GPS altitude', 'Pressure_Pa': 'Pressure', 'Box_Temperature_C': 'Box temperature',
    'Latitude_deg': 'Latitude', 'Longitude_deg': 'Longitude', 'Vertical_speed_m_s': 'Vertical speed',
    'Log_pressure': 'Log pressure', 'Log_pressure_change_per_s': 'Log-pressure change',
}
COLORS = ['#a0acba', '#1e608a', '#16806a', '#bd752a', '#5c609a', '#bf4d4f']


def save(fig, name):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160, bbox_inches='tight', facecolor='white')
    content = buffer.getvalue()
    with Image.open(io.BytesIO(content)) as im:
        im.load()
    temporary = FIGURES / (name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    with Image.open(temporary) as im:
        im.load()
    temporary.replace(FIGURES / name)
    plt.close(fig)


def embedded(name, alt):
    data = base64.b64encode((FIGURES / name).read_bytes()).decode('ascii')
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"></figure>'


def table(frame, index=False):
    return '<div class="table-wrap">' + frame.to_html(index=index, border=0, float_format=lambda x: f'{x:.3f}', na_rep='undefined', escape=True) + '</div>'


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelcolor': '#324459'})
    summary = json.loads((RESULTS / 'run_summary.json').read_text())
    metrics = pd.read_csv(RESULTS / 'metrics_equal_flight_summary.csv')
    per_flight = pd.read_csv(RESULTS / 'metrics_seed_mean_per_flight.csv')
    pred = pd.read_csv(RESULTS / 'held_out_predictions_seed_ensemble.csv')
    shap = pd.read_csv(RESULTS / 'held_out_shap_per_seed.csv')
    features = pd.read_csv(RESULTS / 'model_features.csv').set_index('source_row')
    cohort = pd.read_csv(ROOT / 'inputs' / 'screened_30s_candidates_NOT_VERIFIED.csv')
    primary = metrics[metrics.subset.eq('all')].copy()
    primary['Model'] = primary.variant.map(NAMES)
    primary.to_csv(RESULTS / 'primary_model_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, channel in zip(axes, ['CW12', 'CW123']):
        group = primary[primary.channel.eq(channel)].set_index('variant').loc[list(VARIANTS)]
        ax.barh(np.arange(6), group.rmse_cps_mean, color=COLORS)
        ax.set_yticks(np.arange(6), [NAMES[x] for x in group.index])
        ax.invert_yaxis()
        ax.set(title=f'{channel} · held-out-flight error', xlabel='Mean flight RMSE (counts / elapsed second)')
        for i, value in enumerate(group.rmse_cps_mean):
            ax.text(value, i, f'  {value:.3f}', va='center', fontsize=9)
        ax.set_xlim(0, group.rmse_cps_mean.max() * 1.18)
    fig.suptitle('Real HERA recordings · five flights · exploratory results', fontsize=14, weight='bold')
    fig.tight_layout()
    save(fig, 'model_comparison.png')
    importance = []
    for channel in ['CW12', 'CW123']:
        for variant, columns in VARIANTS.items():
            if not columns:
                continue
            s = shap[shap.channel.eq(channel) & shap.variant.eq(variant)]
            scores = {}
            for column in columns:
                value = s.assign(magnitude=s['shap_' + column].abs()).groupby('outer_test_flight').magnitude.mean().mean()
                scores[column] = value
                importance.append({'channel': channel, 'variant': variant, 'feature': column,
                                   'mean_abs_shap_cps_equal_flight_weight': value})
            order = sorted(columns, key=lambda c: scores[c], reverse=True)
            average = s.groupby('source_row')[[*['shap_' + x for x in columns], 'bias_cps', 'raw_prediction_cps']].mean()
            np.testing.assert_allclose(average[['shap_' + x for x in columns]].sum(axis=1) + average.bias_cps,
                                       average.raw_prediction_cps, atol=2e-5, rtol=1e-5)
            height = max(2.8, 0.48 * len(columns) + 1.7)
            fig, axes = plt.subplots(1, 2, figsize=(12, height), gridspec_kw={'width_ratios': [1, 1.5]})
            y = np.arange(len(order))
            axes[0].barh(y, [scores[c] for c in order], color='#16806a')
            axes[0].set_yticks(y, [LABELS[c] for c in order])
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Mean |SHAP| (counts / second)')
            axes[0].set_title('Importance · equal flight weight')
            rng = np.random.default_rng(2026)
            scatter = None
            for row, column in enumerate(order):
                values = features.loc[average.index, column].to_numpy()
                finite = np.isfinite(values)
                low, high = np.nanpercentile(values, [2, 98])
                colors = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
                colors[~finite] = .5
                scatter = axes[1].scatter(average['shap_' + column], row + rng.uniform(-.22, .22, len(average)),
                                          c=colors, cmap='coolwarm', vmin=0, vmax=1, s=9, alpha=.64, linewidths=0)
            axes[1].set_yticks(y, [LABELS[c] for c in order])
            axes[1].invert_yaxis()
            axes[1].axvline(0, color='#8292a2', lw=.8)
            axes[1].set_xlabel('SHAP contribution to raw prediction (counts / second)')
            axes[1].set_title('Held-out bins · three-seed mean')
            bar = fig.colorbar(scatter, ax=axes[1], pad=.025, fraction=.025)
            bar.set_ticks([0, 1], labels=['Low', 'High'])
            bar.set_label('Feature value')
            fig.suptitle(f'{channel} · {NAMES[variant]} · out-of-flight Tree SHAP', weight='bold')
            fig.tight_layout()
            save(fig, f'shap_{channel}_{variant}.png')
    pd.DataFrame(importance).to_csv(RESULTS / 'shap_feature_importance_equal_flight_weight.csv', index=False)
    for channel in ['CW12', 'CW123']:
        flights = sorted(cohort.flight_id.unique())
        fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
        for row, flight in enumerate(flights):
            for col, variant in enumerate(['altitude_pressure_xgboost', 'restricted_sensor_tuned_xgboost']):
                g = pred[pred.channel.eq(channel) & pred.flight_id.eq(flight) & pred.variant.eq(variant)]
                ax = axes[row, col]
                ax.scatter(g['Altitude[m]'] / 1000, g.observed_rate_cps, c='#8a97a4', s=11, alpha=.5, label='Observed')
                for phase, color in [('ascent', '#16806a'), ('descent', '#bf4d4f')]:
                    p = g[g.phase.eq(phase)]
                    ax.scatter(p['Altitude[m]'] / 1000, p.prediction_cps, s=12, color=color, alpha=.8, label=f'Predicted {phase}')
                ax.set_title(f'{flight} · {NAMES[variant]}', fontsize=10)
                ax.set_ylabel('Recorded rate (counts / s)')
                if row == 0 and col == 0:
                    ax.legend(fontsize=8, frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel('GPS altitude (km)')
        fig.suptitle(f'{channel} · observations and predictions on each entirely held-out flight', weight='bold')
        fig.tight_layout()
        save(fig, f'flight_profiles_{channel}.png')
    blocks = []
    for channel in ['CW12', 'CW123']:
        g = primary[primary.channel.eq(channel)].set_index('variant').loc[list(VARIANTS)]
        comparison = g[['Model', 'r2_mean', 'mse_cps2_mean', 'rmse_cps_mean', 'mae_cps_mean']].rename(columns={
            'r2_mean': 'R²', 'mse_cps2_mean': 'MSE (cps²)', 'rmse_cps_mean': 'RMSE (cps)', 'mae_cps_mean': 'MAE (cps)'})
        baseline = g.loc['training_flight_balanced_constant_mean', 'rmse_cps_mean']
        candidates = g.drop(index='training_flight_balanced_constant_mean')
        winner = candidates.rmse_cps_mean.idxmin()
        gain = (1 - candidates.loc[winner, 'rmse_cps_mean'] / baseline) * 100
        blocks.append(f'<h2>{channel}: held-out-flight results</h2><p>The lowest descriptive XGBoost RMSE is '
                      f'<strong>{html.escape(NAMES[winner])}</strong>: {candidates.loc[winner, "rmse_cps_mean"]:.3f} cps, '
                      f'{abs(gain):.1f}% {"below" if gain >= 0 else "above"} the constant baseline. This comparison is descriptive; '
                      'it is not a separately tested selection of a final winning model.</p>' + table(comparison))
        blocks.append(embedded(f'shap_{channel}_restricted_sensor_tuned_xgboost.png', f'{channel} tuned restricted sensor SHAP plot'))
        blocks.append('<p>Left: mean absolute contribution, weighting each flight equally. Right: one point per held-out interval, '
                      'averaging the three seeds. Colors show low to high values within each feature. Negative contributions lower '
                      'and positive contributions raise the raw prediction relative to the fold-specific model baseline. '
                      'Correlated features can share importance; these are model explanations, not physical causes.</p>')
        blocks.append(embedded(f'flight_profiles_{channel}.png', f'{channel} held-out-flight radiation profiles'))
        detail = per_flight[per_flight.channel.eq(channel) & per_flight.subset.eq('all')].copy()
        detail['Model'] = detail.variant.map(NAMES)
        detail = detail[['outer_test_flight', 'Model', 'r2', 'mse_cps2', 'rmse_cps', 'mae_cps']]
        blocks.append('<details><summary>Per-flight metrics for all six variants</summary>' + table(detail) + '</details>')
    coverage = cohort.groupby('flight_id').agg(intervals=('bin_index', 'size'),
                      altitude_min_m=('Altitude[m]', 'min'), altitude_max_m=('Altitude[m]', 'max')).reset_index()
    html_text = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HERA exploratory real-data results · 6 November 2026</title><style>
body{margin:0;background:#edf2f5;color:#213447;font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:auto;background:white;padding:44px}
h1{font-size:34px;line-height:1.15;margin:8px 0 16px}h2{margin-top:38px;font-size:24px}h3{font-size:18px}p{max-width:980px}
.eyebrow{font-size:12px;letter-spacing:1.7px;color:#16806a;font-weight:750}.notice{border-left:4px solid #bd752a;background:#fff6e8;padding:15px 20px}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px;margin:28px 0}.card{background:#eff6f5;padding:18px;border-radius:9px}.card strong{font-size:28px;display:block}
.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:14px;width:100%;margin:18px 0}th{text-align:left;background:#eaf1f5}th,td{padding:9px;border-bottom:1px solid #d9e2e9}
figure{margin:24px 0}img{width:100%;height:auto}code{background:#eef2f5;padding:2px 5px;font-size:14px}li{margin:7px 0}details{padding:15px;background:#f5f7f9;margin:20px 0}summary{cursor:pointer;font-weight:650}
footer{border-top:1px solid #d9e2e9;margin-top:36px;padding-top:20px;font-size:13px;color:#556575}a{color:#1e608a}@media(max-width:700px){main{padding:22px}.cards{grid-template-columns:1fr}h1{font-size:28px}}
</style><main><div class="eyebrow">HERA RESEARCH · 6 NOVEMBER 2026</div><h1>First exploratory models<br>on real logger recordings</h1>
<p>Count-rate prediction using five independent flight recordings. The original screened tables remain marked unverified.</p>
<div class="notice"><strong>Scope:</strong> recorded cumulative-counter increments per elapsed second. These results do not establish calibrated physical flux,
identical detector response, or a validated Pfotzer-peak altitude. Unknown flight-specific settings and clock assumptions remain part of the study.</div>
<div class="cards"><div class="card"><strong>918</strong>distinct 30-second intervals</div><div class="card"><strong>5</strong>independent flights · 7.65 hours</div><div class="card"><strong>2</strong>channels fitted separately</div></div>
'''
    html_text += embedded('model_comparison.png', 'Comparison of six model variants for both channels')
    html_text += '<h2>What was actually trained</h2><p>For each channel, five XGBoost variants and a flight-balanced constant baseline were evaluated on each held-out flight. '
    html_text += f'The run fitted <strong>{summary["outer_xgboost_models"]} outer-fold XGBoost models</strong> and <strong>{summary["inner_tuning_fits"]} inner tuning models</strong>. '
    html_text += 'Every interval received exactly one held-out prediction per variant and seed. Three random seeds repeat fits; they do not create additional independent flights.</p>'
    html_text += '<p>Tuning used 24 candidates sampled with seed 2026, evaluated using four inner held-out flights and seeds 11, 42 and 73. The outer test flight was excluded from parameter selection. '
    html_text += 'Training gave every flight equal total weight. Sensor derivatives were recomputed only inside contiguous retained intervals, without bridging excluded gaps. The analysis uses completed trajectories and is not a real-time forecast.</p>'
    html_text += '<p>Primary scores average seed metrics within each flight, then average the five flights equally. MSE is in squared cps; RMSE and MAE are in cps. '
    html_text += 'Averaging flight RMSE is different from taking the square root of average flight MSE. Negative R² means worse than that test flight’s own constant-mean reference. '
    html_text += 'The CSVs also report between-flight spread, within-flight seed variation, ascent/descent scores and errors restricted to each phase’s training altitude min/max span. '
    html_text += 'That span check does not prove dense training support throughout the range. Predictions below zero are clipped for scoring; SHAP explains the raw output.</p>'
    html_text += ''.join(blocks)
    html_text += '<h2>Data and integrity</h2>' + table(coverage)
    html_text += f'<p>All 38 original package fingerprints matched before fitting. All {summary["outer_xgboost_models"]} saved native models reproduced predictions after reload. '
    html_text += f'The largest SHAP reconstruction discrepancy was {summary["max_shap_additivity_error_cps"]:.8f} cps. '
    html_text += 'All bundled input fingerprints still match. Seven preparation tests and six training tests passed.</p>'
    html_text += '<h2>Recovered project location and computer setup</h2><p>The supplied preparation README records '
    html_text += '<code>V:\\Radiation_Profiles_Pfotzer_Peak_Estimation</code> and <code>.venv\\Scripts\\python.exe</code>. '
    html_text += 'Recovered base files are <code>logger_preparation.py</code> and <code>prepare_logger_data.py</code>, plus the frozen evaluation protocol. '
    html_text += 'The original XGBoost notebook was not included. Older-chat retrieval was disabled, so the user’s exact CPU, RAM, GPU and Python version could not be confirmed. '
    html_text += 'The execution environment recorded in run_summary.json belongs to this analysis environment, not the user’s computer. The supplied check_system.py reads local specifications and lists likely model notebooks without installing anything.</p>'
    html_text += '<h2>Remaining interpretation limits</h2><ul>'
    for limitation in summary['limitations']:
        html_text += '<li>' + html.escape(limitation) + '</li>'
    html_text += '<li>The professor’s screenshot describes intended upward orientation, usual CosmicWatch use with occasional GMC counters, settings usually unchanged, and some coincidence operation. '
    html_text += 'It does not supply exact settings or detector identities for every selected file. Counter labels are preserved without assuming inclusive channel populations.</li>'
    html_text += '<li>No final production model was chosen from these test comparisons. No independent external validation or peak estimate is claimed.</li></ul>'
    html_text += '<h2>Reproduce or inspect</h2><p>Open README.md for Windows commands, dependency checks and fresh-output instructions. '
    html_text += 'The results folder contains the metrics, held-out predictions, SHAP contributions, tuning evaluations, fitted models and their feature-order metadata. '
    html_text += 'Figures for every XGBoost variant are included. No Google Drive writes were performed.</p>'
    html_text += '<footer>Methods: <a href="https://xgboost.readthedocs.io/en/release_2.1.0/python/python_api.html">XGBoost prediction contributions</a>; '
    html_text += '<a href="https://scikit-learn.org/stable/modules/cross_validation.html">grouped cross-validation</a>; '
    html_text += '<a href="https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html">nested parameter selection</a>. '
    html_text += 'Source screening and tuning plan: the unchanged HERA preparation package dated 23 October 2026.</footer></main></html>'
    (ROOT / 'HERA_Real_Data_Results.html').write_text(html_text, encoding='utf-8')
    print('Created self-contained report and 13 scientific figures')


if __name__ == '__main__':
    main()

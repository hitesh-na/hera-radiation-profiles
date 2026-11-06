"""Explain and evaluate saved flight-held-out XGBoost models; never trains.

Usage: python scripts/explain_saved_models.py --run results/synthetic_demo
Output is a NEW directory, leaving the input run untouched.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


def scores(y, prediction):
    mse = float(mean_squared_error(y, prediction))
    return dict(mse_cps2=mse, rmse_cps=float(np.sqrt(mse)),
                mae_cps=float(mean_absolute_error(y, prediction)),
                r2=float(r2_score(y, prediction)) if len(y)>1 and np.var(y)>0 else np.nan)


def slug(value):
    return re.sub(r'[^A-Za-z0-9_-]', '_', str(value))


def contributions(booster, x):
    dm = xgb.DMatrix(x)
    values = booster.predict(dm, pred_contribs=True, approx_contribs=False)
    raw = booster.predict(dm, output_margin=True)
    if values.shape != (len(x), x.shape[1]+1):
        raise ValueError('Only single-output regression is supported')
    residual = float(np.max(np.abs(values.sum(axis=1)-raw)))
    if not np.allclose(values.sum(axis=1), raw, atol=2e-4, rtol=2e-5):
        raise ValueError('TreeSHAP additivity check failed')
    return values, raw, residual


def run(run_dir, output, sample_per_flight=250, display_seed=None):
    run_dir, output = Path(run_dir), Path(output)
    if sample_per_flight < 1:
        raise ValueError('sample-per-flight must be positive')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty output directory')
    cfg = json.loads((run_dir/'run_config.json').read_text())
    provenance = json.loads((run_dir/'provenance.json').read_text())
    data = pd.read_csv(run_dir/'quality_reports/validated_features.csv')
    predictions = pd.read_csv(run_dir/'cv_metrics/heldout_predictions.csv')
    mode = cfg['data_mode']
    if provenance['source_kind'] != mode or set(data.source_kind) != {mode}:
        raise ValueError('Inconsistent data provenance')
    display_seed = cfg['seeds'][0] if display_seed is None else display_seed
    if display_seed not in cfg['seeds']:
        raise ValueError('Display seed is not in the saved run')
    expected = {(str(f), int(seed), v) for f in data.flight_id.unique()
                for seed in cfg['seeds'] for v in cfg['variants']}
    metapaths = sorted((run_dir/'model_artifacts').glob('*.metadata.json'))
    metadata = [(p, json.loads(p.read_text())) for p in metapaths]
    keys = [(str(m['flight_id']), int(m['seed']), m['variant']) for _, m in metadata]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Saved models do not cover the declared flight/seed/variant grid')
    output.mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir()
    (output/'shap_values').mkdir()
    label = 'SYNTHETIC DEMONSTRATION — NO REAL HERA RESULTS' if mode=='synthetic' else 'REAL HERA — HELD-OUT FLIGHT EVALUATION'
    metric_rows, importance, checks, samples = [], [], [], {}
    for p, meta in metadata:
        flight, seed, variant = str(meta['flight_id']), int(meta['seed']), meta['variant']
        if flight in meta['train_flights'] or set(meta['train_flights']) != set(data.flight_id.unique())-{flight}:
            raise ValueError('Train/test flight separation failed')
        if meta['input_sha256'] != provenance['input_sha256'] or meta['source_kind'] != mode:
            raise ValueError('Model/input provenance mismatch')
        test = data[data.flight_id==flight].copy()
        x = test[meta['features']].astype(float)
        booster = xgb.Booster(params={'nthread':2})
        model_path = p.with_name(p.name.replace('.metadata.json', '.json'))
        booster.load_model(model_path)
        objective = json.loads(booster.save_config())['learner']['objective']['name']
        if objective != 'reg:squarederror':
            raise ValueError('CPS-scale explanation requires reg:squarederror')
        raw = booster.predict(xgb.DMatrix(x))
        prediction = np.maximum(raw, 0) if meta.get('clip_predictions_below_zero') else raw
        old = predictions[(predictions.flight_id==flight)&(predictions.seed==seed)&(predictions.variant==variant)]
        if old.source_row.duplicated().any() or set(old.source_row) != set(test.source_row):
            raise ValueError('Saved prediction rows do not match held-out rows')
        old = old.set_index('source_row').loc[test.source_row]
        if not np.allclose(prediction, old.prediction_cps, atol=2e-5, rtol=2e-5):
            raise ValueError('Reloaded model predictions differ from saved evaluation')
        if not np.allclose(test.Counts_Per_Second, old.Counts_Per_Second, atol=1e-8):
            raise ValueError('Saved targets differ')
        stamp = dict(flight_id=flight, seed=seed, variant=variant, source_kind=mode)
        metric_rows.append({**stamp, **scores(test.Counts_Per_Second,prediction),
                            'n_test':len(test), 'negative_raw_prediction_fraction':float((raw<0).mean())})
        values, raw_shap, residual = contributions(booster, x)
        if not np.allclose(raw,raw_shap,atol=2e-5,rtol=2e-5):
            raise ValueError('Raw prediction and margin differ')
        for j, feature in enumerate(x.columns):
            importance.append({**stamp,'feature':feature,'mean_abs_shap_cps':float(np.abs(values[:,j]).mean())})
        frame = test[['source_row','flight_id','Timestamp_s']].reset_index(drop=True)
        for j, feature in enumerate(x.columns):
            frame['shap__'+feature] = values[:,j]
        frame['base_value_cps'] = values[:,-1]
        frame['raw_prediction_cps'] = raw
        frame['evaluated_prediction_cps'] = prediction
        frame.to_csv(output/'shap_values'/f'{p.stem.replace(".metadata", "")}.csv.gz',index=False)
        checks.append({**stamp,'max_additivity_error_cps':residual,
                       'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest()})
        if seed==display_seed:
            ix = np.linspace(0,len(x)-1,min(sample_per_flight,len(x)),dtype=int)
            samples.setdefault(variant,[]).append((values[ix,:-1],x.iloc[ix]))
        print(f'Explained {flight} / {variant} / seed {seed}',flush=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output/'metrics_per_flight_seed.csv',index=False)
    names = ['r2','mse_cps2','rmse_cps','mae_cps']
    # Average seeds inside each flight first. Flights, not seeds, are the units.
    per_flight = metrics.groupby(['variant','flight_id'])[names].mean()
    per_flight.to_csv(output/'metrics_per_flight_seed_mean.csv')
    summary = per_flight.groupby('variant').agg(['mean','std','min','max'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary['independent_flights'] = data.flight_id.nunique()
    summary.to_csv(output/'model_comparison.csv')
    metrics.groupby(['variant','flight_id'])[names].std().to_csv(output/'seed_standard_deviation.csv')
    imp = pd.DataFrame(importance)
    imp.to_csv(output/'shap_importance_per_flight_seed.csv',index=False)
    balanced = imp.groupby(['variant','feature','flight_id']).mean(numeric_only=True).groupby(['variant','feature']).mean(numeric_only=True)
    balanced[['mean_abs_shap_cps']].to_csv(output/'shap_importance_equal_flight.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for variant in cfg['variants']:
        rank = balanced.loc[variant].mean_abs_shap_cps.sort_values()
        fig,ax=plt.subplots(figsize=(9, max(4,len(rank)*.35+2)))
        ax.barh(rank.index,rank.values,color='#257c9e')
        ax.set(xlabel='Mean |SHAP| (counts/second; before clipping)',title=f'{variant}: feature importance\nEqual weight per flight; averaged across seeds')
        fig.suptitle(label,fontsize=10,color='#9a4329');fig.tight_layout()
        fig.savefig(output/'figures'/f'shap_importance_{variant}.png',dpi=170);plt.close(fig)
        vals=np.concatenate([z[0] for z in samples[variant]])
        features=pd.concat([z[1] for z in samples[variant]],ignore_index=True)
        fig,ax=plt.subplots(figsize=(10,max(4,len(rank)*.38+2)))
        rng=np.random.default_rng(2026)
        for i,feature in enumerate(rank.index):
            j=features.columns.get_loc(feature);a=features[feature].to_numpy()
            finite=np.isfinite(a); lo=np.nanmin(a) if finite.any() else 0;hi=np.nanmax(a) if finite.any() else 0
            color=(a-lo)/(hi-lo) if hi>lo else np.full(len(a),.5)
            y=i+rng.uniform(-.23,.23,len(a))
            ax.scatter(vals[finite,j],y[finite],c=color[finite],cmap='coolwarm',vmin=0,vmax=1,s=6,alpha=.55,rasterized=True)
            if (~finite).any():ax.scatter(vals[~finite,j],y[~finite],color='gray',s=6,alpha=.55)
        ax.axvline(0,color='gray',lw=.8)
        ax.set(yticks=range(len(rank)),yticklabels=rank.index,xlabel='SHAP contribution to raw prediction (counts/second)',title=f'{variant}: held-out SHAP distribution\nSeed {display_seed}; up to {sample_per_flight} evenly spaced rows per flight')
        cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='coolwarm'),ax=ax,pad=.02)
        cb.set_ticks([0,1],labels=['Low','High']);cb.set_label('Feature value (within feature); missing = gray')
        fig.suptitle(label,fontsize=10,color='#9a4329');fig.tight_layout()
        fig.savefig(output/'figures'/f'shap_distribution_{variant}.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    variants=cfg['variants']; positions=np.arange(len(variants))
    for ax,name in zip(axes.flat,names):
        for i,v in enumerate(variants):
            y=per_flight.loc[v][name].to_numpy()
            ax.scatter(np.full(len(y),i),y,color='#86adbd',s=26)
            ax.scatter(i,np.nanmean(y),color='#123c56',marker='D',s=48)
        ax.set(xticks=positions,xticklabels=[v.replace('_','\n') for v in variants],ylabel=name,title='Higher is better' if name=='r2' else 'Lower is better')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle(label+'\nDots: individual held-out flights (seed means); diamonds: equal-flight mean',fontsize=12)
    fig.tight_layout();fig.savefig(output/'figures/model_error_comparison.png',dpi=170);plt.close(fig)
    (output/'validation.json').write_text(json.dumps({'source_kind':mode,'models_checked':len(checks),'checks':checks,'xgboost_version':xgb.__version__,'display_seed':display_seed,'trained_models':False},indent=2))
    (output/'README.md').write_text(f'''# Model explanations and errors

{label}

This command loaded {len(checks)} existing models; it did not fit or tune any model.
All evaluation rows belong to the model's held-out flight. Saved predictions were reproduced.

`model_comparison.csv`: equal-flight mean, sample standard deviation, min and max for R², MSE, RMSE and MAE. Seeds are averaged within a flight first; they are not independent flights. MSE units are (counts/second)². RMSE and MAE units are counts/second. R² may be negative and is undefined for constant targets. MSE here is ordinary mean squared error, not a standardized or normalized score. Flight variability is descriptive, not a confidence interval.

`figures/`: metric comparison, mean-absolute-SHAP importance and colored SHAP distribution plots for every variant. Distribution plots use seed {display_seed}, chosen by configuration rather than performance, with evenly spaced samples per flight. Bar plots use all held-out rows and all seeds, with equal flight weighting. Distribution points have deterministic vertical jitter for visibility.

Exact TreeSHAP comes from XGBoost pred_contribs=True, approx_contribs=False. It explains the raw square-error-regression output before the existing nonnegative clipping step. Contributions plus the base value reconstruct that raw output; additivity is checked. For clipped predictions the SHAP values do not sum to the clipped output. Negative-raw-prediction fractions are reported per model.

Positive SHAP increases a prediction relative to its model baseline; negative SHAP decreases it. Mean absolute SHAP measures contribution magnitude, not direction. Correlated altitude/pressure/derived variables can share attribution. These plots describe fitted model behavior, not causal physics. Different folds have different fitted baselines. Synthetic feature importance is not evidence about HERA instruments; synthetic magnetic/acceleration/density features are excluded from the reviewed real-data feature profile.

The baseline variant is an XGBoost model with basic sensor features. The other variants alter features or tune hyperparameters; these are not six different algorithm families. Whole-flight outer splits are preserved; tuning was performed inside training flights by the original pipeline. This post-hoc script never selects or tunes models using test SHAP values.

Sources: https://xgboost.readthedocs.io/en/stable/prediction.html and https://shap.readthedocs.io/en/stable/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html
''')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--sample-per-flight',type=int,default=250)
    parser.add_argument('--display-seed',type=int)
    args=parser.parse_args()
    output=args.output or args.run.parent/(args.run.name+'_shap_metrics')
    print(run(args.run,output,args.sample_per_flight,args.display_seed))

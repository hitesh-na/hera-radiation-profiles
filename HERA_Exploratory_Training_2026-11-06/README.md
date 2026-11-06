# HERA exploratory real-data training — 6 November 2026

The completed results are supplied. **You do not need to run training again to view them.** Open `HERA_Real_Data_Results.html` in a browser.

This separate add-on follows the evaluation protocol restored from `HERA_Logger_Preparation_2026-10-23.zip`. It contains a new trainer, not the missing original XGBoost notebook. It reads the unchanged 918 screened intervals, preserves their unverified status, and targets recorded rates per elapsed second. No original notebook, source recording, or Google Drive file was changed.

## Recovered base code and location

The attached preparation README records the project location as:

`V:\Radiation_Profiles_Pfotzer_Peak_Estimation`

It records the existing Python executable as:

`V:\Radiation_Profiles_Pfotzer_Peak_Estimation\.venv\Scripts\python.exe`

Recovered preparation code is located at:

- `HERA_Logger_Preparation_2026-10-23\logger_preparation.py`
- `HERA_Logger_Preparation_2026-10-23\prepare_logger_data.py`
- `HERA_Logger_Preparation_2026-10-23\results\frozen_evaluation_protocol.json`

The exact original modeling notebook name and the user's hardware/software specifications could not be recovered: older-chat retrieval returned “Personal context is disabled.” The rate-plot notebook names and DGX Spark instructions inside Annafy's reference README describe that reference material; they are not evidence of the user's own computer or original XGBoost code.

Extract this new ZIP into the recorded project folder. It creates a new `HERA_Exploratory_Training_2026-11-06` subfolder. Keep the previously supplied preparation package too.

## Read your actual local system specifications first

This command is read-only and uses Python's standard library. It reports OS, Python and package versions, Windows CPU/RAM/GPU names, and candidate modeling notebook/script filenames in the project. It does not install packages or read notebook contents.

```powershell
Set-Location 'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'
.\.venv\Scripts\python.exe .\HERA_Exploratory_Training_2026-11-06\check_system.py
```

To save a new local copy of that report, optionally run:

```powershell
.\.venv\Scripts\python.exe .\HERA_Exploratory_Training_2026-11-06\check_system.py --output .\HERA_Exploratory_Training_2026-11-06\system_check_run01.json
```

Choose a new filename if it already exists. The report can be shared back to check compatibility and identify the original notebook. These instructions were prepared from the recovered project path; they have not been executed on the user's computer.

## Requirements for an optional rerun

- A 64-bit Python environment; the supplied code targets Python 3.10 or newer.
- NumPy, pandas, scikit-learn, XGBoost, and Matplotlib.
- CPU processing is the default: two workers, one thread per model. `--workers 1` reduces concurrency.
- A GPU, CUDA, and the separate `shap` package are not required. SHAP values come directly from XGBoost's native exact tree contribution API.
- No numerical RAM minimum is claimed without checking the user's machine. There are 918 rows and at most eight predictors; original 78 MB raw recordings are not loaded by this trainer.

The completed models used **XGBoost 2.1.4**, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and Matplotlib 3.10.8. These are the execution environment's versions, not the user's confirmed versions. `requirements.txt` gives compatible ranges; `requirements_exact_run.txt` records the exact versions used for the completed run and requires a compatible Python version (3.12 was used here).

Check the local report before installing or upgrading anything. If dependencies are missing, `requirements.txt` can be installed in a separate environment to preserve the existing project environment. Native `.ubj` models and JSON feature-order metadata avoid the scikit-learn wrapper serialization incompatibility observed during the first attempted run.

## Completed study

- Five independent flight groups, 918 distinct 30-second intervals (7.65 hours).
- Channels CW12 and CW123 fitted separately; neither channel's counts/rates are predictors of the other.
- Five XGBoost feature variants and a training-flight-balanced constant baseline.
- Leave one entire flight out in each of five outer folds.
- For the tuned variant: four inner held-out flights, 24 pre-specified random parameter candidates, seeds 11/42/73. Selection uses mean inner-flight RMSE only.
- 2,880 inner tuning fits and 150 saved outer-fold XGBoost models. Repeated seeds and channels do not add independent flights.
- Scores: R², MSE, RMSE, MAE. Primary reporting averages seeds within flights, then flights equally. Per-flight, ascent/descent, altitude-span overlap, and seed variation tables are included.
- SHAP values are computed only after fitting, on the outer held-out rows. Contributions explain raw predictions. Scores use predictions clipped at zero; none were negative in this run.
- Seven preparation tests and six trainer tests passed. All 150 model reload checks passed with zero prediction discrepancy. Maximum SHAP reconstruction discrepancy was approximately 0.0000153 cps.
- All 38 original package fingerprints matched; all copied input fingerprints still match after training.

The pressure-only variant had the lowest descriptive mean held-out-flight RMSE for both channels. This does not establish a separately validated production-model selection. All fold models are supplied; no final model fitted on all five flights was selected or evaluated on an external sixth flight.

## Re-run only if needed

Choose a new empty output directory. The supplied `results` already contains the completed study and will not be overwritten.

```powershell
.\.venv\Scripts\python.exe .\HERA_Exploratory_Training_2026-11-06\test_training.py
.\.venv\Scripts\python.exe .\HERA_Exploratory_Training_2026-11-06\train_exploratory.py --exploratory --output .\results\hera_exploratory_run02 --workers 2 --threads 1
```

The `--exploratory` entry point records the user's authorization to study unverified recorded-rate candidates. Keep the old notebook's `RUN_TRAINING = False`; this add-on does not change that notebook or set `training_verified=True`.

The bundled `make_results_report.py` reads the completed `results` in this package and regenerates figures/report without training. The completed run took about 104 seconds here; this is not a time estimate for the user's machine. Small numerical differences can occur across software versions and hardware.

## Main outputs

- `HERA_Real_Data_Results.html`: self-contained report with metrics, SHAP plots and held-out profiles.
- `figures/`: all 13 scientific figures, including SHAP for every XGBoost variant/channel.
- `results/primary_model_comparison.csv`: primary scores, averaged equally by flight.
- `results/metrics_per_flight_seed.csv`, `metrics_seed_mean_per_flight.csv`, `metrics_equal_flight_summary.csv`, `metrics_seed_standard_deviation.csv`: full metric detail and spread.
- `results/held_out_predictions_per_seed.csv`: predictions exclusively from models excluding that interval's entire flight.
- `results/held_out_predictions_seed_ensemble.csv`, `secondary_seed_ensemble_metrics.csv`: secondary seed-ensemble results, distinct from primary mean seed metrics.
- `results/held_out_shap_per_seed.csv`, `shap_feature_importance_equal_flight_weight.csv`: raw contribution values and equal-flight-weight summaries.
- `results/inner_tuning_evaluations.csv`, `chosen_parameters_by_outer_fold.json`, `run_plan_before_fitting.json`: full tuning provenance and pre-fit plan.
- `results/models/*.ubj`, paired `*.metadata.json`: native boosters with exact feature order, parameters, channel, held-out flight and seed. Predict using native `xgboost.Booster` and inputs in the recorded feature order.
- `results/saved_model_and_shap_checks.csv`, `run_summary.json`, `training_tests.txt`: verification results.
- `inputs/`: unchanged screened tables, source manifest, frozen protocol and splits, and input fingerprints.

## Interpretation

These are uncalibrated recorded-count-rate models. Detector live time is unavailable. The timer wrap/scale assumptions and incomplete detector configuration records remain limitations. The professor described intended upward orientation, usually CosmicWatch detectors (occasionally GMC counters), generally unchanged settings and some coincidence operation; that reply does not identify exact settings for every file. Two channels remain separate recorded labels without an assumed inclusive relationship.

Centered sensor derivatives use only contiguous retained bins of completed flights. They are appropriate for this offline study, not a claim of causal real-time prediction. GPS altitude, pressure and log pressure are related predictors, and their SHAP importance is not evidence of independent physical causes. Peak estimation and physical-flux calibration remain separate work.

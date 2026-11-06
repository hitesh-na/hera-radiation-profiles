# HERA SHAP and model-error add-on

The included plots and scores are SYNTHETIC DEMONSTRATIONS ONLY. No real HERA model has been trained. No training setting needs to change.

This add-on adds one standalone analysis script. It loads existing saved model artifacts; it never trains models or edits raw data. No additional Python package beyond the project's existing requirements is needed. XGBoost itself calculates exact TreeSHAP.

## Install and run in Windows PowerShell

Extract this ZIP into V:\Radiation_Profiles_Pfotzer_Peak_Estimation, merging the scripts and docs folders. The example outputs are under examples\shap_metrics_synthetic_demo. They are ready to view without running anything.

To regenerate from your own existing synthetic run:

    Set-Location 'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'
    .\.venv\Scripts\python.exe .\scripts\explain_saved_models.py --run .\results\synthetic_demo --output .\results\synthetic_demo_shap_metrics --display-seed 42

The input run must contain run_config.json, provenance.json, validated_features.csv under quality_reports, heldout_predictions.csv under cv_metrics, and all saved model artifacts with metadata. If that run has not been generated on your computer, use the included example outputs; do not enable real training to make this command work. If the output directory already exists and contains files, choose a new output name.

For a future VERIFIED real-data run, supply its saved run directory to --run and a new directory to --output. Keep RUN_TRAINING=False and MODE='synthetic' until preparation is complete. This script's ability to read saved real models does not authorize or verify new training.

## Read the outputs

- model_comparison.csv: R², ordinary MSE, RMSE and MAE, averaged equally across held-out flights after averaging seeds within flights; also across-flight standard deviation, minimum and maximum.
- metrics_per_flight_seed.csv: all 54 synthetic fits (6 feature/tuning variants x 3 flights x 3 seeds).
- figures/model_error_comparison.png: comparison for all six variants.
- figures/shap_importance_*.png: contribution magnitude using all held-out rows, equal weighting per flight and averaging across seeds.
- figures/shap_distribution_*.png: feature value and signed contribution, seed 42, at most 250 evenly spaced rows per held-out flight.
- shap_values/*.csv.gz: exact contributions and base values for every held-out row.
- validation.json: saved-model hashes and SHAP reconstruction errors.

The six variants are all XGBoost: baseline sensor features, engineered features, tuned engineered features, altitude only, pressure only, and altitude + pressure. Extra features and tuning are not guaranteed to improve held-out performance.

SHAP explains predictions, not causal physical effects. Correlated altitude, pressure, temperature and derived features can share attribution. The synthetic temperature, density, magnetic and acceleration features shown in some examples do not change the restricted real-data feature policy.

SHAP contributions sum with the model baseline to the RAW prediction before nonnegative clipping. Evaluation metrics use the actual clipped predictions. The script checks that loaded models reproduce saved predictions and that SHAP values reconstruct raw predictions.

Validation performed: all 54 saved models reproduced held-out predictions; all train/test metadata checks passed; maximum SHAP reconstruction error was 0.00002671 counts/second. Plots were visually inspected. Synthetic data demonstrate the workflow only and must not be reported as real-flight model performance.

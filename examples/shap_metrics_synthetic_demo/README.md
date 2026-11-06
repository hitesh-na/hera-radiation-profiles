# Model explanations and errors

SYNTHETIC DEMONSTRATION — NO REAL HERA RESULTS

This command loaded 54 existing models; it did not fit or tune any model.
All evaluation rows belong to the model's held-out flight. Saved predictions were reproduced.

`model_comparison.csv`: equal-flight mean, sample standard deviation, min and max for R², MSE, RMSE and MAE. Seeds are averaged within a flight first; they are not independent flights. MSE units are (counts/second)². RMSE and MAE units are counts/second. R² may be negative and is undefined for constant targets. MSE here is ordinary mean squared error, not a standardized or normalized score. Flight variability is descriptive, not a confidence interval.

`figures/`: metric comparison, mean-absolute-SHAP importance and colored SHAP distribution plots for every variant. Distribution plots use seed 42, chosen by configuration rather than performance, with evenly spaced samples per flight. Bar plots use all held-out rows and all seeds, with equal flight weighting. Distribution points have deterministic vertical jitter for visibility.

Exact TreeSHAP comes from XGBoost pred_contribs=True, approx_contribs=False. It explains the raw square-error-regression output before the existing nonnegative clipping step. Contributions plus the base value reconstruct that raw output; additivity is checked. For clipped predictions the SHAP values do not sum to the clipped output. Negative-raw-prediction fractions are reported per model.

Positive SHAP increases a prediction relative to its model baseline; negative SHAP decreases it. Mean absolute SHAP measures contribution magnitude, not direction. Correlated altitude/pressure/derived variables can share attribution. These plots describe fitted model behavior, not causal physics. Different folds have different fitted baselines. Synthetic feature importance is not evidence about HERA instruments; synthetic magnetic/acceleration/density features are excluded from the reviewed real-data feature profile.

The baseline variant is an XGBoost model with basic sensor features. The other variants alter features or tune hyperparameters; these are not six different algorithm families. Whole-flight outer splits are preserved; tuning was performed inside training flights by the original pipeline. This post-hoc script never selects or tunes models using test SHAP values.

Sources: https://xgboost.readthedocs.io/en/stable/prediction.html and https://shap.readthedocs.io/en/stable/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html

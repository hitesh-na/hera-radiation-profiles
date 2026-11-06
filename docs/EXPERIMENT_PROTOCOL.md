# Experiment protocol

## Fixed experimental design

All predictive algorithms are `XGBRegressor` with `reg:squarederror` and histogram tree construction. The baseline uses the seven original sensor inputs. Engineered and tuned-engineered variants add sensor-only derived features. Altitude-only, pressure-only and altitude-plus-pressure variants test atmospheric predictor redundancy. No Random Forest, conventional Gradient Boosting, Gaussian Process or neural-network benchmark is trained.

For each outer held-out flight and each seed (11, 42, 73), inner leave-one-flight-out CV operates only on outer-training flights. Candidate scores are the arithmetic mean of per-validation-flight RMSE. The baseline hyperparameter candidate is explicitly included in every search. Fixed sampled candidate settings are reused across model seeds (`search_seed=2026`), so differences across seeds reflect stochastic fitting and resulting tuning selection, not an entirely different search grid.

The example tests six random candidates plus baseline; the research configuration tests 24 plus baseline. The grid explores `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha` and `reg_lambda`. Every inner fit and score is recorded. The outer model is refitted using selected settings and all outer-training rows. The outer target never enters search, stopping, feature construction or selection.

**Early stopping is disabled.** Tree count is itself an inner-CV hyperparameter. This avoids using the outer test flight as an evaluation set and avoids unclear tree-count refitting rules with only two outer-training flights.

Each training flight has equal total sample weight; rows within a flight have equal weight. This reduces domination by flight duration but is not an inverse-variance or Poisson likelihood fit. Negative CPS predictions are clipped to zero in both inner scoring and outer evaluation, with the raw negative fraction recorded. This rule is fixed before testing.

## Offline setting and preprocessing

Features are computed independently per physical flight using only sensors. Centered rolling medians/gradients use the completed trajectory; this is allowed for the declared offline reconstruction. No target values enter the feature allowlist. Derivatives do not cross gaps larger than 30 s by default. Phase is supplied from verified metadata or inferred from vertical speed using a 0.5 m/s threshold. Native XGBoost missing values avoid a globally fitted imputer.

Raw inputs require explicit units and clock origins. Detector bins are the left side of a nearest-time sensor join, keeping exactly one outcome row per bin. Duplicate environmental timestamps are numerically averaged and counted in provenance. An initial maximum match distance of 2.5 s is required and can be changed only with documented sampling justification. Counts are never duplicated by merging one bin to many sensor rows.

## Reporting without selection bias

- Report every held-out flight and seed. Do not label the best seed as the single result.
- Average seeds within a flight before summarizing across independent flights. The three seeds are not nine independent flights.
- Report RMSE, MAE and R2 for raw rows, equal-bin profile RMSE/MAE for each phase, and peak status/altitude results.
- Peak errors exist only for pairs where both peaks pass screening. Report how many pairs qualify, so low error on a selected subset is not mistaken for universal performance.
- Compare paired errors against baseline for matching flight/seed. A lower row RMSE does not guarantee a better peak location.
- Changes made after inspecting outer outcomes are exploratory. Fresh flights are needed for a confirmatory improvement claim.
- The final all-data refit uses the predeclared tuned-engineered variant and seed 42, not the best outer-test variant/seed. Its grouped search is recorded separately; it has no independent test score.

## Profiles and peaks

Measured CPS is sum(counts)/sum(exposure) where both are valid. Predicted CPS is averaged with the same exposure weights. Without exposure, both use arithmetic means; without counts, no Poisson intervals are reported. Ascent and descent are always separate in scientific profile products.

The primary grid uses 250 m bins with at least three observations. A 3-bin centered boxcar is used for peak screening. Smoothing never crosses missing or sparse bins, and the unsmoothed data remain in the output. The peak position is a bin midpoint: a precise-looking decimal is not a statement of physical precision.

Screening requires a contiguous supported profile, an interior maximum, at least three bins and 1.5 km of coverage on either side, a drop on both broad shoulders exceeding the larger of 0.5 CPS and 10% of the maximum, no comparably high prominent second maximum separated by at least 1.5 km, and no top wider than 2 km. When counts/exposure exist, a conservative two-counting-SE threshold supplements the prominence check. Predicted screening uses a Poisson proxy from predicted rate and exposure rather than the held-out count target. These are heuristics, not a calibrated significance test or a validated physical classifier.

Every 100/250/500 m and 1/3-bin setting is recorded in the sensitivity tables. Small bins may legitimately become unresolved through sparse sampling. Sensitivity ranges describe processing choices; they are not 95% confidence intervals. No calibration, block bootstrap or hierarchical scientific peak interval is claimed. Such intervals require instrument information and a separate justified uncertainty model.

## Feature interpretation

Normalized split gain is saved for every model. The tuned variant also has a descriptive grouped circular-shift perturbation analysis on each held-out flight. Related atmospheric features move together within phase; other feature groups are shifted separately. This preserves within-group trajectories but can still break between-group physical relationships. Values are diagnostic changes in CPS RMSE, not causal effects or formal permutation-test p-values. They do not select features or settings.

## Sources for implementation choices

- XGBoost's official sklearn interface documentation: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html
- Scikit-learn's LeaveOneGroupOut documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
- Scikit-learn's nested-validation discussion: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

The scientific acceptance rules are provisional project choices based on the handoff, not claims that those software manuals prescribe Pfotzer peak criteria.

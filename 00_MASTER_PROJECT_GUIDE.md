# Radiation Profiles & Pfotzer Peak Estimation

## Authoritative objective

> Map continuous atmospheric radiation curves to precisely locate the Pfotzer-Regener maximum and generate standardized profiles for cross-flight comparison.

This package implements the attached `Pfotzer_XGBoost_Handoff_For_Another_AI.md`. XGBoost is the only predictive model family used. The earlier STAR feature-importance work supplies context and data-format evidence; it is not the objective or a new benchmark.

## Scientific question

Can an empirical CPS-altitude profile and a model trained on other flights identify the same well-supported interior maximum? When the records do not bracket a maximum, the correct result is an explicit unresolved, ambiguous or edge-limited status.

## Implemented decisions

| Decision | Implementation and reason |
|---|---|
| Dependent variable | CosmicWatch detector CPS; preserve uncalibrated units |
| Coordinate | Supplied altitude in meters; definition/calibration must be documented |
| Time bins | Initial 5 s for raw input, subject to verified exposure and sampling |
| Primary altitude grid | Fixed absolute 250 m bins, at least 3 observations per bin |
| Flight phases | Separate ascent/descent; stationary/unknown rows logged and excluded by default |
| Profile aggregation | Sum counts divided by sum live time when both exist; otherwise explicitly labeled weighted/unweighted CPS aggregation |
| Feature derivation | Sensor-only, computed separately within each flight and continuous time segment |
| Missing sensors | Explicit masks and native XGBoost missing-value handling; no target-based imputation |
| Row objective | Squared-error CPS regression with equal total weight per training flight |
| Tuning objective | Mean RMSE across inner validation flights; fixed before outer scoring |
| Validation | Nested leave-one-flight-out; no random mixed-row scientific test |
| Repeats | Seeds 11, 42, 73; all saved and reported |
| Tree counts | Fixed per candidate, selected in inner CV; no early stopping on the outer test flight |
| Peak status | Explicit screening of coverage, interior location, shoulder drops, plateau and competing maxima |
| Sensitivity | 100/250/500 m grids and 1/3-bin smoothing; descriptive, not selected by test accuracy |
| Uncertainty | Counting-only Poisson intervals where supported; seed/sensitivity spread never called scientific confidence intervals |
| Prediction setting | Offline completed flight; same observed environmental trajectory as the measured profile |
| Final refit | Tuned-engineered, seed 42, predefined independently of the outer results; no new test claim |

The initial altitude bin, prominence thresholds and acquisition gap tolerances are **research protocol defaults**, not universal physical constants. Review their adequacy using actual sampling and instrument knowledge before confirmatory analysis, then freeze them. Document later changes as exploratory.

## Completion status

- Implemented: code, configuration, documented data adapter, data audit, XGBoost feature ablations, nested validation, repeated runs, profile reconstruction, peak rejection, sensitivity, standardized comparisons, saved models, notebook, tests and reporting.
- Executed: labeled synthetic three-flight demonstration and a separate synthetic final refit. One fixture intentionally ends below its generating peak.
- Pending: actual CosmicWatch and CLUE files, confirmed clock/exposure conventions, detector comparability, real training, empirical peak interpretation and defensible peak-location uncertainty.
- Not claimed: HERA accuracy, true physical peak precision, radiation dose, calibrated flux, or improvement from tuning on HERA data.

## Pipeline

Raw measurements are immutable inputs. A verified manifest maps their units and clocks into binned detector exposure and environmental readings. A nearest-timestamp join retains one row per detector bin with a strict tolerance. Quality checks preserve exclusion reasons. Measured profiles are generated before XGBoost fitting. Nested grouped experiments then produce predictions for entirely withheld flights, and the same profile rules compare observed and predicted rates. Final artifacts contain the evidence needed to trace each number.

## Next action

Obtain the six paired files listed in `docs/DATA_REQUIREMENTS.md`, plus detector/clock notes. More independent flights are particularly valuable: three flights give only two outer-training groups and one training group in each inner fold.

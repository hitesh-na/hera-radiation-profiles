# Radiation Profiles & Pfotzer Peak Estimation

> Map continuous atmospheric radiation curves to precisely locate the Pfotzer-Regener maximum and generate standardized profiles for cross-flight comparison.

## Current outcome

A reproducible XGBoost-only analysis workflow has been implemented. It includes auditable preprocessing, sensor-derived features, nested flight-level cross-validation, hyperparameter optimization, repeated seeds, profile reconstruction, conservative peak screening and standardized cross-flight comparison. The workflow has been executed on clearly labeled synthetic fixtures and its scientific failure cases have been checked.

**Real HERA model training and empirical Pfotzer peak estimates remain pending the original measurements.** The supplied files are prior notebooks, a research poster and an implementation handoff. Notebook table previews are truncated and cannot replace raw data. The synthetic model artifacts are software demonstrations and must not be used as trained HERA models.

## Prior research, separated from new results

The prior STAR poster reports within-flight Random Forest and conventional Gradient Boosting analyses with R2 around 0.98-0.99 and dominant altitude or pressure feature importance. These are reported prior findings, not scores obtained by this project. The new study evaluates XGBoost on completely withheld flights and prioritizes radiation profile and peak-location quality.

The previous notebooks reveal inconsistent seconds/milliseconds handling, event-interval exposure assigned wholly to an event's bin, and nearest-time joins without a maximum match distance. The new adapter requires explicit conventions, interpolates verified cumulative counters at bin boundaries only under stated assumptions, accepts independently verified exposure tables, and records rejected or unmatched rows.

## Software demonstration

The included example consists of three synthetic flights with known generator settings. Two span both sides of a broad count-rate maximum; the third ends below its generating maximum. The pipeline evaluates six XGBoost variants, three outer held-out flights and three seeds, saving all 54 outer-fold models and the inner search history. A separate all-fixture refit demonstrates how a future-flight model would be saved.

The complete demonstration metrics are in `../results/synthetic_demo/REPORT.md`. They should only be discussed as software verification. In this demonstration, feature engineering and tuning did not improve mean held-out RMSE relative to the untuned seven-sensor baseline. That negative result is retained rather than selecting a favorable seed or changing the protocol after testing. It says nothing about which variant will work best on actual HERA flights.

The incomplete fixture receives an edge-limited status and no accepted peak altitude. Tests also exercise flat curves, gaps, competing maxima, valid zero-count intervals, clock rollover, counter resets, exposure conservation, disjoint inner/outer groups and target-independent features.

## Interpretation and limits

Tree predictions reconstruct an empirical rate profile; they do not create a continuously measured physical field. Measured and predicted points are placed on the same observed altitude grid, and any smoothing is explicitly labeled. Three independent flights are sufficient for an exploratory workflow but provide limited evidence of population generalization.

The measured quantity is CosmicWatch CPS. Cross-flight detector gain, threshold, geometry, acceptance and live-time differences can confound absolute counts. The reported peak-screening status is evidence for an observed interior rate maximum, not proof by itself that the physical Pfotzer-Regener maximum was measured. Counting statistics, binning, calibration and model sensitivity all affect localization. No arbitrary accuracy target or scientific confidence interval is asserted.

## Empirical work still required

1. Supply the six original paired files for 2026-03-28, 2025-10-04 and 2025-06-01, plus additional flights if available.
2. Confirm authoritative detector stream, timestamp/deadtime units, acquisition intervals, clocks and altitude calibration.
3. Audit measured time/altitude/pressure/rate profiles and freeze data-quality and peak thresholds from sampling and instrument evidence.
4. Run the frozen nested experiment and interpret held-out profiles, unresolved statuses and paired error changes together.
5. Develop a justified uncertainty analysis if a numerical peak-location confidence interval is required; repeated seeds alone cannot provide it.

## Supervisor-reviewable explanation

This project reconstructs detector count-rate profiles from high-altitude balloon measurements and uses XGBoost trained on other flights to predict those profiles. The workflow estimates a Pfotzer-Regener peak only when an interior maximum is supported by measurements on both sides; otherwise it records why the peak is unresolved. Whole-flight validation, repeated runs and standardized altitude bins allow the study to compare prediction quality and profile shape while keeping detector calibration and uncertainty limits explicit. The software is complete as a tested research workflow; empirical conclusions await the original HERA flight measurements.

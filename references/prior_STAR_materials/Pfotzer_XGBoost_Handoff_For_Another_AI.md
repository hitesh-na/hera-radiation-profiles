# AI HANDOFF BRIEF — Radiation Profiles & Pfotzer Peak Estimation

**READ THIS ENTIRE FILE BEFORE STARTING IMPLEMENTATION.**  
**Purpose of this document:** Transfer the user's actual research request, the background learned from supplied STAR files, the desired implementation, the development plan, the end goal, and the rules for interpreting results to another AI model. This is a project specification, **not** a claim that the proposed science has already been completed.

**Status:** Project requirements clarified; prior STAR notebooks and poster reviewed; an optional preliminary XGBoost software scaffold exists, but the original HERA raw flight files have **not** been provided, so real-world XGBoost results and real Pfotzer peak estimates are **not yet established**.

---

## 1. Official assignment — preserve this wording

**Project:** Radiation Profiles & Pfotzer Peak Estimation

**Objective (exact wording supplied by the professor):**

> Map continuous atmospheric radiation curves to precisely locate the Pfotzer-Regener maximum and generate standardized profiles for cross-flight comparison.

**The professor's project title and objective are authoritative.** Do not silently rename or reframe this as a generic feature-importance, classification, or regression-accuracy project.

### What this means in practical language

Using high-altitude balloon data, build radiation detector **count-rate-versus-altitude profiles** and estimate the altitude at which the atmospheric secondary-radiation response reaches its maximum (the Pfotzer-Regener/Pfotzer maximum), **if the measurements genuinely include a well-supported maximum**. Then process different flights consistently so that their curves and peak estimates can be compared.

A successful result is not merely a high R². The main end products are **well-supported radiation profiles, peak-location estimates or explicitly unresolved statuses, and standardized cross-flight comparisons**.

The user has explicitly chosen **XGBoost** as the machine-learning technique. Implement and optimize **XGBoost regression**, not Gaussian Process Regression or neural networks, and do **not** introduce Random Forest and conventional Gradient Boosting as new benchmark algorithms unless the user later requests that comparison. It is appropriate to compare **different versions/settings of XGBoost** to prove whether feature engineering and tuning help.

---

## 2. User's preferences and requirements

The user wants an actual research project—not merely conceptual suggestions—with a logical, reproducible development process. Incorporate all of the following:

1. **XGBoost-only predictive modeling**, applied to radiation count rate from HERA balloon flights.
2. **Feature engineering:** design, justify, and test useful atmospheric and flight-derived predictors.
3. **Hyperparameter optimization:** properly tune XGBoost without leakage from test flights.
4. **Multiple training and testing runs:** record seed-to-seed variability; report distributions rather than cherry-picking the best run.
5. **Cross-validation:** use splits that measure generalization across independent flights, not just randomly mixed rows.
6. **Accuracy improvement:** document what changed and whether held-out prediction error and **peak-localization quality** actually improved.
7. **Radiation profile generation:** both measured and model-predicted CPS-versus-altitude profiles.
8. **Pfotzer peak estimation:** distinguish peaks that are supported by data from missing, ambiguous, or edge-limited peaks.
9. **Cross-flight standardization:** place different flights on compatible altitude bins and compare curve shapes as well as absolute count rates (subject to detector calibration).
10. **Deliverables:** organized code/notebooks, configuration, data dictionary, plots/tables, experimental results, documentation, and a final report/proposal/poster-ready explanation.

**Do not treat GPR, latent-variable modeling, or the user's earlier questions about math as project requirements.** Those were exploratory questions, subsequently superseded by an explicit XGBoost decision.

---

## 3. What the supplied previous STAR project actually did

The user supplied these research references:

- `Florini-STAR.ipynb` — main 2026 STAR notebook.
- `Florini-STAR-Misc.ipynb` — exploratory/scratch notebook, including LOWESS and magnetometer calibration attempts.
- `rateGroundPressure.ipynb` — an earlier detector-count/atmospheric-pressure analysis notebook.
- `STAR Poster Florini.pdf` — *Machine Learning Analysis of HERA Data*, Florini Pettiss and Dr. Christina Love, Drexel University.

**These are prior-work references, not this project's official objective.** Their useful contribution is the data provenance, sensor formats, preprocessing methods, candidate features, flight dates, and scientific context.

### Instrumentation and measurements

- **CosmicWatch:** detector event records; used to calculate `Counts_Per_Second` (CPS). A CosmicWatch count rate is **not automatically a calibrated radiation dose rate or total radiation flux**.
- **Adafruit CLUE data logger:** altitude, pressure, temperature, acceleration x/y/z, magnetic field x/y/z, latitude, longitude, and timestamps.
- Original STAR inputs to regression: `Altitude[m]`, `Pressure[Pa]`, `Temperature[C]`, `Accel` (vector magnitude), `Mag` (vector magnitude), `Latitude`, `Longitude`.
- The main notebook's later CosmicWatch parser typically bins particle events into **five-second intervals** and attempts to correct count rate for detector dead/live time. It joins detector and CLUE records using nearby timestamps.

### Three flights previously studied

- **March 28, 2026**
- **October 4, 2025**
- **June 1, 2025**

The original work fit **Random Forest** and **scikit-learn Gradient Boosting** separately within these flights. It used an approximately **80/20 random train/test split** on rows and reported R² values around **0.98–0.99**. Altitude received most tree-importance credit in two flights; pressure dominated the June 1 flight. **These are the previous researcher's reported numbers, NOT results from a new XGBoost experiment.** Correlated altitude/pressure features complicate individual importance interpretation. A high random-within-flight R² does not establish accuracy on a completely unseen flight.

The `Florini-STAR-Misc.ipynb` notebook explores LOWESS fitting of count rate versus altitude, then subtracting the fitted trend to examine residuals; it also experiments with magnetometer hard-iron offset correction. These are ideas, not finalized validated methods. The poster mentions residual analysis, restricted-altitude bands, ground measurements, and magnetic calibration as possible future work.

### Known caveats in the old preprocessing

The main notebook contains multiple versions of `getCosmicWatch_df`, including **inconsistent seconds-versus-milliseconds assumptions**. It also uses nearest-timestamp matching without a maximum allowable time difference. Some exploratory cells are out of execution order. **Adapt, review, and validate the prior code; do not copy its assumptions blindly.** Inspect raw headers, column conventions, time bases, cumulative-dead-time interpretation, and sensor clock offsets before training.

---

## 4. Scientific target: what is the predicted quantity?

The starting observable is detector particle event rate, measured in **counts per second (CPS)**. If raw counts and valid detector live time are known:

\[
\mathrm{CPS}_i = \frac{N_i}{T_{\mathrm{live},i}},
\]

where \(N_i\) is the observed count in interval \(i\) and \(T_{\mathrm{live},i}\) is valid detector exposure time. For a simple Poisson counting approximation, the counting-error scale is approximately \(\sqrt{N_i}/T_{\mathrm{live},i}\) when applicable. Handle zero counts, zero/negative live time, shutdown periods, missing intervals, and detector-specific rules explicitly.

The model learns a relation between **flight/environmental measurements and the observed CPS**. The ultimate scientific display is **count rate versus altitude**, not arbitrary uncalibrated units labeled as dose or flux. If calibrated radiation measurements are available, ask the professor whether those should replace or supplement CPS.

**A continuous-looking profile needs careful construction.** XGBoost tree predictions are not inherently smooth continuous functions. A defensible profile is obtained by ordering/aggregating observed and predicted values onto an altitude grid, with clearly documented optional smoothing **within supported altitude coverage only**. Label the result as an empirical/model-based reconstructed profile, not a directly measured continuous instrument signal.

**Critical condition:** A maximum cannot be assigned reliably if the flight does not record adequate data on **both sides** of a peak. An endpoint maximum is not automatically the Pfotzer-Regener maximum. When measurements are insufficient, return `unresolved` or a similarly explicit reason, not a fabricated precise altitude.

---

## 5. Desired project architecture and data flow

**Raw CosmicWatch events + raw CLUE logs**  
→ validate formats/units/flight metadata  
→ calculate event counts, live time, CPS and uncertainties  
→ synchronize detector/CLUE times  
→ produce auditable combined per-flight data  
→ engineer physically plausible sensor-derived features  
→ train/tune XGBoost using flight-grouped validation  
→ generate held-out CPS predictions  
→ reconstruct measured/predicted altitude profiles  
→ screen/estimate peaks + robustness  
→ standardize and compare flight profiles  
→ export reproducible scientific figures/tables/report.

### Minimum processed data schema

| Field | Meaning |
|---|---|
| `flight_id` | Unique flight identifier |
| `Timestamp_s` | Synchronized seconds from a documented reference |
| `Counts_Per_Second` | Observed detector rate (target) |
| `Altitude[m]` | Altitude, meters |
| `Pressure[Pa]` | Atmospheric pressure, pascals |
| `Temperature[C]` | Temperature, degrees Celsius |
| `Accel` | Magnitude of acceleration x/y/z, with verified input units |
| `Mag` | Magnitude of magnetometer x/y/z, with verified input units |
| `Latitude`, `Longitude` | Position |
| `Raw_Counts` | Raw integer particle counts per interval (preferable) |
| `Live_Time_s` | Verified exposure time per interval (preferable) |
| `Rate_Error` | Estimated CPS counting error (optional) |
| `match_time_error_s` | CW/CLUE synchronization gap (recommended) |

Keep raw files unchanged. Record which instrument contributed each column, time adjustments, excluded data, missingness, units, detector ID/calibration, and flight date. Track ascending/descending portions explicitly where possible.

**Data availability warning:** The user previously supplied notebooks and a poster, **not** the raw CosmicWatch/CLUE measurement files referenced inside them. Do not claim to have trained on real HERA data until those files (or an auditable combined flight table) are actually available. Example data and synthetic testing are acceptable only if clearly labeled as such.

---

## 6. Detailed implementation plan

### Phase A — data acquisition and quality audit

**Goal:** Before any ML, establish which measurements are usable and whether candidate flights bracket a physical peak.

1. Request original CosmicWatch `.txt` and CLUE `.csv` files for the three flights, plus additional flights if available.
2. Validate timestamp units (seconds vs ms), detector cumulative deadtime, bin exposure, sensor RTC rollover/offsets, alignment tolerance, altitude/pressure units, missing values, duplicate rows, outliers, and detector hardware/threshold settings.
3. Plot observed CPS vs time, altitude vs time, pressure vs altitude, and measured CPS vs altitude for each flight.
4. Determine altitude coverage, ascent/descent phases, gaps, and whether there is evidence of **rising and falling** count rates around an interior maximum.
5. Produce a data quality report. Document failures and exclusions instead of silently discarding them.

**Acceptance gate:** Do not assert that a particular flight has a measured Pfotzer maximum unless the coverage and detector behavior support it.

### Phase B — construct a first measured radiation profile

1. Calculate valid rate with exposure correction where supported by the raw data.
2. Aggregate measurements by a **common altitude grid** (illustrative starting bin size: **250 m**, subject to sampling/coverage).
3. Record count/exposure and sample coverage per altitude bin. Consider Poisson/statistical uncertainty and temporal dependence.
4. Plot measured CPS against altitude. Keep gaps rather than filling absent measurements with invented data.
5. Analyze ascent and descent separately or justify pooling; differing conditions can distort a combined profile.

**Goal:** Establish a defensible observed reference before training any XGBoost model.

### Phase C — initial XGBoost baseline

Use `XGBRegressor` with basic sensor features and a rate-regression objective (e.g. `reg:squarederror` initially). Do not automatically use a standard random row split as the primary scientific test. Record data, hyperparameters, seed, and output artifacts.

**Initial predictors:** altitude, pressure, temperature, acceleration magnitude, magnetic magnitude, latitude, longitude. Depending on flight physics and missingness, start with an altitude-only or altitude-plus-pressure **XGBoost variant** as a diagnostic; this is still an XGBoost-only project.

**Initial metrics:** held-out CPS RMSE, MAE and R², plus profile accuracy and peak-status/peak-altitude results.

### Phase D — feature engineering

Engineer candidates from **sensor inputs and flight metadata**, with no leakage from the held-out CPS target:

- Log pressure (for positive pressure only).
- Pressure/temperature atmospheric-density proxy, with units and validity checked.
- Vertical speed and ascent/descent indicator from altitude/time (prefer robust differences or physically sensible smoothing; define whether inference is offline vs live).
- Pressure change rate and optionally atmospheric-depth-inspired features where justified.
- Consistently calibrated acceleration/magnetic magnitudes; avoid claiming magnetometer calibration is finished when it is not.
- Optional sensor quality, valid-exposure and trajectory metadata when legitimate at prediction time.

Run **feature ablations** using XGBoost: original inputs, engineered inputs, altitude-only, pressure-only, and selected combinations. Test whether additions improve held-out profiles and peak estimates, not merely training accuracy. Avoid leakage from target-derived trend curves, observed held-out peaks, or future information unavailable in the declared prediction setting.

### Phase E — hyperparameter optimization

Use `RandomizedSearchCV`, Optuna, or a comparably reproducible tuning method **inside training flights only**. Reasonable parameters to explore:

`n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`.

Use a documented search budget, sensible bounds, a fixed selection metric, and explicit early-stopping/validation logic if applicable. Never tune on the outer held-out flight or report the best test seed as if it were a fair single result.

### Phase F — cross-validation and repeated experiments

**Primary design:** **leave-one-flight-out** outer evaluation. Train on all but one flight, test on the entirely withheld flight, and rotate the held-out flight. For hyperparameter tuning, use **inner grouped CV on training flights only**; with only three flights total, there are only two training groups in each outer fold, so explain that tuning/generalization estimates will be statistically limited. Add more independent flights if possible.

**Multiple runs:** e.g., repeat with predeclared random seeds `11`, `42`, `73` (and more if computationally feasible). Report all relevant statistics, central tendency, spread, and per-flight outcomes. Optional chronological within-flight blocked validation provides an **additional diagnostic**, not a substitute for testing on unseen flights.

**Comparison allowed/expected:** baseline XGBoost vs engineered XGBoost vs tuned engineered XGBoost; changes across different hyperparameter sets, selected features, altitude bins and smoothing settings. **Do not spend project scope on training Random Forest/ordinary Gradient Boosting to compare algorithms; the user explicitly rejected that scope.**

### Phase G — radiation profile reconstruction

For each **held-out** flight, generate:

1. Measured CPS vs altitude profile from actual observations.
2. XGBoost-predicted CPS vs altitude profile using that flight's measured environmental/trajectory inputs but **not** its CPS target as a predictor.
3. Matching altitude bins with clear missing-data masks, support counts and optional uncertainty/error information.
4. Clearly labeled smoothing settings if used to make the profile visually continuous.

Do **not** manufacture sensor-input combinations by freely varying altitude while keeping inconsistent pressure/temperature values. An observed-flight-supported profile or a physically validated atmospheric input trajectory is preferable. Do not extrapolate curves past valid sampled altitude ranges.

### Phase H — Pfotzer peak extraction and quality control

Build a documented peak estimator operating on the altitude profile:

- Require an interior maximum rather than the first/last supported bin.
- Check adequate samples on both sides and a meaningful amount of altitude coverage.
- Set/prominently report minimum prominence and continuity rules; handle noise, gaps, and competing maxima.
- Return **peak altitude (m and km), peak CPS, coverage, and a status** such as `identified`, `ambiguous`, `edge_limited`, or `unresolved`.
- Compare predicted and observed peak altitudes only if **both** pass the same scientifically defensible identification rules.
- Test sensitivity to reasonable binning and smoothing settings (illustratively 100, 250, 500 m; thresholds must be justified against data density).
- Distinguish repeat-seed variability from **true scientific uncertainty**. Peak-altitude confidence intervals require a separate justified method incorporating count statistics, data sampling, calibration and model sensitivity.

**Priority:** An honest `unresolved` result is better than a falsely precise estimate.

### Phase I — standardized cross-flight comparison

- Resample/aggregate every flight to the **same altitude coordinate and bin spacing**.
- Retain original absolute CPS where detector normalization/calibration permits comparison.
- Optionally divide by a **valid observed/estimated per-flight peak** to compare shapes, but don't use unvalidated endpoint maxima as if they were Pfotzer peaks; clearly describe any other normalization used.
- Compare only overlapping **observed** altitude coverage. Mark missing regions; do not silently extrapolate.
- Plot overlays and tables of peak altitude, shape/width where defensible, error bars/quality flags, and differences across flights.
- Discuss detector hardware, atmospheric conditions, geomagnetic/latitude changes, and data-quality confounding.

### Phase J — final reporting and reproducibility

Produce a well-explained research report or poster documenting how the raw files led to every number. Separate **prior STAR findings**, **new HERA measurements**, **new XGBoost results**, and **interpretations/hypotheses**. Include caveats where the dataset cannot answer a question.

---

## 7. What success should look like

### Scientific deliverables (highest priority)

- A graph of **actual measured CPS versus altitude** for every usable flight.
- A graph showing **held-out XGBoost-predicted CPS versus altitude**, aligned with observed measurements.
- A **per-flight Pfotzer peak table**: altitude, peak rate, quality flag, observed/predicted distinction, coverage, and uncertainty/sensitivity (as available).
- A **standardized cross-flight altitude-profile table** and overlay figure.
- A clear conclusion about which peaks are genuinely identifiable and which require more altitude coverage or better data.

### ML/engineering deliverables

- Documented HERA preprocessing and quality checks.
- XGBoost baseline + engineered + tuned variants.
- Repeated flight-held-out CV metrics with no test leakage.
- Hyperparameter search history and chosen configurations.
- Feature engineering/ablation report, feature importances and preferably out-of-flight permutation importance or SHAP (interpret associations, **not causation**).
- Profile-level and peak-level metrics in addition to aggregate CPS prediction metrics.
- Reproducible scripts/notebook, dependency versions, saved model(s) as appropriate, configuration, documented tests, and a research log.

### Minimum example output table (EMPTY until real HERA data exist)

| Flight | Measured altitude coverage | Measured peak status/height | Held-out XGB peak status/height | Peak altitude difference (if justified) | Held-out CPS RMSE | Notes |
|---|---|---|---|---|---|---|
| 2026-03-28 | Pending raw data | Pending | Pending | Not yet computed | Pending | Verify altitude coverage |
| 2025-10-04 | Pending raw data | Pending | Pending | Not yet computed | Pending | Verify altitude coverage |
| 2025-06-01 | Pending raw data | Pending | Pending | Not yet computed | Pending | Verify altitude coverage |

**Accuracy goal:** Improve cross-flight predictive and/or peak-localization performance compared with an **untuned XGBoost baseline** if possible, and transparently report if improvements do not occur. Do not invent an arbitrary R², accuracy, or peak-precision guarantee. A lower overall RMSE can still coexist with a worse estimated peak location; report both.

---

## 8. Proposed repository structure to implement/deliver

```text
Radiation_Profiles_Pfotzer_Peak_Estimation/
  00_MASTER_PROJECT_GUIDE.md       # single source of truth
  README.md                        # installation and run commands
  PROJECT_PROPOSAL.md              # scientific brief for supervisor
  requirements.txt or pyproject.toml
  config/
    flights.example.json           # file locations, detector/clock conventions
  data/
    raw/                           # original files (not invented)
    processed/                     # traceable synced CPS/flight tables
  src/ or pfotzer_xgb/
    preprocessing.py               # CosmicWatch/CLUE parsing and synchronization
    quality.py                     # measurements, coverage, unit checks
    features.py                    # flight-aware feature engineering
    train.py / experiments.py      # XGBoost variants, tuning, grouped CV, repeats
    profiles.py                    # common-altitude profiles and visualization
    peaks.py                       # defensible peak status and altitude estimates
    sensitivity.py                 # binning, smoothing, robustness
  notebooks/
    Exploration_and_Results.ipynb
  tests/
    test_preprocessing.py
    test_profiles_peaks.py
    test_validation.py
  results/
    quality_reports/
    cv_metrics/
    profile_tables/
    peak_tables/
    figures/
    model_artifacts/
  docs/
    DATA_DICTIONARY.md
    EXPERIMENT_PROTOCOL.md
    RESEARCH_LOG.md
    FINAL_REPORT.md
  references/
    prior_STAR_materials/         # only if user uploads originals
```

A prior assistant assembled an **optional preliminary prototype** named `Radiation_Profiles_Pfotzer_Peak_XGBoost_Project.zip`, with `00_MASTER_PROJECT_GUIDE.md`, preprocessing, features, profiles, experiments, quality/sensitivity scripts, documentation and tests. **If that ZIP is supplied, inspect and reuse/repair it as appropriate rather than assuming it is perfect; independently verify the methods and test results.** If it is not supplied, this handoff document alone should be sufficient to plan and build the project from scratch using the user's raw data when available.

---

## 9. Ground rules and limitations — nonnegotiable

1. **No fabricated empirical results.** Do not treat synthetic demonstration plots as real HERA findings.
2. **No falsely identified maxima.** An upper-altitude endpoint maximum does not prove the Pfotzer peak was observed.
3. **No unstated units.** CPS is not automatically dose, flux, or calibrated intensity. Check pressure and detector times carefully.
4. **No leakage.** Held-out CPS values and derived peaks cannot influence fitted predictors, hyperparameters, stopping, or feature selection for that held-out test.
5. **No automatic extrapolation claims.** Tree models generally extrapolate poorly beyond training/observed physical regimes.
6. **No treating correlated feature importances as causal proof.** Altitude and pressure may convey overlapping atmospheric information.
7. **No assumption that all flight detectors are directly comparable.** Gain, threshold, geometry, efficiency and live-time changes may affect measured CPS.
8. **A small number of flights limits inference.** Three-flights-only CV supports an exploratory study, not strong population-level generalization claims.
9. **Preserve uncertainty.** Binning and smoothing affect peak position; repeated seeds alone are not scientific confidence intervals.
10. **XGBoost only for the new ML modeling task, unless user changes scope.** No gratuitous Random Forest/GBR benchmark development.

---

## 10. Questions for the professor or missing facts (do not guess)

- Is the desired dependent variable **CosmicWatch counts per second**, calibrated flux, or dose? Which radiation detector/configuration is considered authoritative?
- Should the horizontal coordinate be **altitude** (GPS/geometric/geopotential), **atmospheric depth** (g/cm²), or both?
- Which balloon flights are available, and do they span **both sides** of a candidate Pfotzer maximum?
- Should ascent and descent profiles be separated? What about averaging repeated crossings?
- Are the sensor hardware settings, clocks, altitude calibration and detector efficiency known across flights?
- What peak-altitude accuracy or uncertainty would count as “precise” enough for the assignment?
- What final format, deadline and review milestones does the professor want?

Ask for these important details when necessary, but make progress on the design and testable code scaffold with clearly labeled assumptions rather than stalling.

---

## 11. Explicit request to the next AI model

**Please build or refine a complete, scientifically defensible, reproducible XGBoost-only project titled "Radiation Profiles & Pfotzer Peak Estimation" for the professor's exact objective above.** Use the previous Florini STAR materials as references for CosmicWatch/CLUE data formats and existing findings if those files are available. Do not confuse the old STAR feature-importance objective with the professor's new peak/profile objective.

**Start by:** (1) assessing what raw data are available, (2) validating/preparing measured rate-versus-altitude profiles, (3) implementing an XGBoost pipeline with feature engineering, grouped cross-validation, inner hyperparameter optimization and repeated runs, (4) building conservative peak extraction and standardized flight profiles, and (5) delivering organized source code, documentation, tests, plots, results tables and an honest final interpretation.

**If raw data are absent:** provide the complete project scaffold, exact data requirements and runnable test/synthetic examples, **without claiming real Pfotzer peak results or actual HERA accuracy**. Clearly distinguish implemented components from future work. Preserve the user's XGBoost-only requirement.

**The end goal:** a supervisor-reviewable and eventually poster/paper-ready workflow that can answer *where a defensible measured/model-supported Pfotzer-Regener maximum occurs in each HERA flight, how stable that estimate is, and how standardized radiation-versus-altitude profiles compare across flights*, with correct scientific uncertainty and quality flags.

---

**End of AI handoff.**

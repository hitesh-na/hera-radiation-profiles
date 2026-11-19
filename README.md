# Radiation Profiles & Pfotzer Peak Estimation

Study how flight measurements relate to recorded atmospheric radiation count rates and evaluate whether flight profiles support an estimate of the Pfotzer peak altitude.

## Final models and results

The [final XGBoost package](HERA_Final_XGBoost_2026-11-19/) contains two saved models, the plots, and a script that predicts CW1&2 and CW1&2&3 count rates from air pressure in pascals. Both models were fitted on 918 complete 30-second intervals from five HERA balloon flights.

| Recorded channel | Pressure-only mean held-out RMSE |
|---|---:|
| CW1&2 | 0.512 counts/s |
| CW1&2&3 | 0.497 counts/s |

Pressure alone gave the lowest average error among the configurations tested, including altitude alone and models with additional sensor readings. In both channels, the modeled count rate rose as pressure dropped, reached a broad maximum, then fell at the lowest pressures. Each score comes from the earlier whole-flight tests that trained on four flights and predicted the fifth. Independent testing of the two final all-flight models remains open. The Pfotzer peak altitude remains unresolved because the individual altitude profiles had gaps or too few supported bins.

- [Run predictions with a pressure CSV](HERA_Final_XGBoost_2026-11-19/README.md#make-predictions)
- [View the final plots](HERA_Final_XGBoost_2026-11-19/figures/model_profiles.png)
- [Inspect the exploratory experiment and data](HERA_Exploratory_Training_2026-11-06/README.md)

## Original notebook and synthetic example

The original notebook shows the workflow on labeled synthetic flights and supports separate experiments on verified datasets. Start with [Exploration and Results](notebooks/Exploration_and_Results.ipynb). You can inspect the saved example without retraining.

## Included

- XGBoost baseline, engineered, tuned-engineered, altitude-only, pressure-only and altitude-plus-pressure variants.
- Leave-one-flight-out outer validation, training-flight-only inner tuning and seeds 11, 42, 73.
- Explicit raw-file units and clock mapping, live-time checks, bounded synchronization and row-exclusion logs.
- Measured and held-out predicted CPS profiles, separate ascent/descent, conservative peak statuses and bin/smoothing sensitivity.
- Cross-flight overlap and valid-peak shape comparisons, search histories, saved models and reproducibility metadata.
- Unit tests for exposure, synchronization, peak rejection, leakage and model serialization.
- A completed **synthetic** example with 54 outer-fold fits, plus a separate final demonstration refit for new synthetic data.

## Install locally

Python 3.10 or newer is required; the included run used Python 3.12. Extract the ZIP, open a terminal in this folder, then:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or `.venv\Scripts\activate` on Windows. Then:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m jupyterlab notebooks/Exploration_and_Results.ipynb
```

CPU training is sufficient. No GPU, external account or data download is required for the included example. On supported Linux systems, `xgboost-cpu` is a smaller alternative providing the same Python `xgboost` API; do not install both distributions in one environment. `requirements-tested.txt` records the versions used here.

## Google Colab

Upload the complete ZIP to Colab's Files panel, unzip it there, and open the notebook from the extracted `notebooks` directory. In a setup cell run `!pip install -r /content/Radiation_Profiles_Pfotzer_Peak_Estimation/requirements.txt` and `!pip install -e /content/Radiation_Profiles_Pfotzer_Peak_Estimation --no-deps`. Set `PROJECT_ROOT` in the notebook to that extracted folder if automatic detection does not find it. Do not upload only the notebook: it uses the accompanying modules and results.

## Rerun the synthetic experiment

The bundled demonstration data already exist. Each run needs a new output directory so previous evidence is retained.

```bash
python -m pfotzer_xgb run --data data/synthetic/synthetic_flights.csv --config config/demo.json --output results/my_synthetic_run
```

To regenerate the fixed fixture in a different directory:

```bash
python -m pfotzer_xgb demo-data --output data/my_synthetic_fixture
```

## Train on actual HERA flights

1. Obtain the paired detector and environmental files listed in `docs/DATA_REQUIREMENTS.md`.
2. Verify raw header layouts, timestamp units, clocks, deadtime semantics, active intervals and detector settings. The old notebooks disagree on seconds versus milliseconds.
3. Fill in a copy of `config/flights.example.json`. The template is intentionally **unverified** and cannot silently assume units. Paths are relative to the manifest file.
4. Process raw files:

```bash
python -m pfotzer_xgb prepare --manifest config/flights.json --output data/processed/run01
```

5. Review `combined.csv` and `provenance.json`. Create a metadata file using `config/combined_metadata.example.json`; fill the exact CSV SHA-256 and verified conventions. The preprocessing provenance supplies the hash and raw source records. If you already have an auditable combined CSV, start here instead.
6. With at least three usable independent flights:

```bash
python -m pfotzer_xgb run --data data/processed/run01/combined.csv --metadata data/processed/run01/verified_metadata.json --config config/research.json --output results/hera_run01
```

Never mark conventions verified merely to bypass a check. Unknown exposure cannot be converted into a valid zero rate. A combined table containing CPS without raw counts/exposure is accepted only with a documented CPS derivation; counting uncertainty then remains unavailable.

## Refit the original workflow with verified data

After completing evaluation, the predeclared tuned-engineered variant can be fitted to all supplied flights. It is tuned by grouped CV again and then trained on all rows; it has **no independent test score of its own**.

```bash
python -m pfotzer_xgb fit-final --data data/processed/run01/combined.csv --metadata data/processed/run01/verified_metadata.json --config config/research.json --output results/hera_final
python -m pfotzer_xgb predict --model results/hera_final --data data/processed/new_flight_sensors.csv --output results/new_flight_predictions.csv
```

Prediction requires the same verified sensor units, flight IDs and timestamps. The target CPS is not required. This is an **offline completed-flight reconstruction**, because centered sensor derivatives use neighboring time points. A model trained on synthetic data refuses unlabeled or real-data prediction input. The final demo model is not usable as a HERA-trained model.

## Where to look

| File or folder | What it answers |
|---|---|
| `docs/DATA_REQUIREMENTS.md` | Exact files and conventions needed next |
| `docs/EXPERIMENT_PROTOCOL.md` | How leakage, tuning and repeated runs are handled |
| `results/synthetic_demo/REPORT.md` | Completed example results, explicitly synthetic |
| `results/synthetic_demo/figures/heldout_profiles.png` | Measured fixture curves and withheld-flight predictions |
| `results/synthetic_demo/cv_metrics/per_flight_seed.csv` | All row prediction scores, without seed selection |
| `results/synthetic_demo/peak_tables/heldout_peaks_and_profile_metrics.csv` | Peak statuses, conditional peak errors and profile errors |
| `results/synthetic_demo/model_artifacts/` | All 54 outer models and exact feature/configuration metadata |
| `results/synthetic_final/` | Separate all-fixture refit, for software demonstration only |
| `docs/FINAL_REPORT.md` | Supervisor-ready status, limitations and next empirical work |

The observable is detector counts per second. It is not calibrated flux or dose. `identified` means an interior peak candidate passes the configured screening rules; physical interpretation still requires detector and flight review. No fabricated endpoint peak, extrapolated curve or scientific confidence interval is supplied.

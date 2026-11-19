# HERA final XGBoost models

Open **OPEN_FINAL_RESULTS.html** for the project results, plots and peak screening. The final refit predicts recorded counts per elapsed second for CW1&2 and CW1&2&3 from pressure alone. Each channel has a separate saved model trained on the same 918 complete 30-second intervals from five flights, with equal total weight per flight.

| Channel | Earlier mean whole-flight test RMSE | Earlier mean whole-flight test R² |
|---|---:|---:|
| CW1&2 | 0.512 counts/s | 0.986 |
| CW1&2&3 | 0.497 counts/s | 0.983 |

These scores come from fitting on four flights and predicting the fifth, averaged across seeds and flights. Pressure only was selected after reviewing those comparisons. The two delivered models were then fitted to all five flights with the original pressure-only settings and seed 42. Independent testing of these final fits remains open. The project peak screening found no qualifying measured and predicted peak pair at the primary setting or at the tested sensitivity settings.

## Make predictions

With Python 3.11 or 3.12, open a terminal in the extracted folder and install the pinned dependencies in a project environment.

```powershell
python -m pip install -r requirements.txt
python predict.py example_pressure.csv predictions.csv
```

Replace `example_pressure.csv` with prepared flight inputs. Supply exactly one column named `Pressure_Pa` or `Pressure[Pa]`, in pascals. For comparable predictions, prepare the same kind of screened 30-second logger intervals and use the same counter definitions. The output preserves the input rows and adds predicted rates for both channels and a pressure status. Rows with invalid pressure or pressure outside the training span of about 1,198 to 94,357 Pa have blank predictions. The span describes the minimum and maximum observed pressures.

## Reproduce the refit

Use the [GitHub project snapshot](https://github.com/hitesh-na/hera-radiation-profiles/tree/38644fcb0f97c08641bea0835234b8b6d864c669) with its `HERA_Exploratory_Training_2026-11-06` folder. Run the command below from the extracted model folder and replace the repository path with your checkout. Use a new output directory.

```powershell
python refit.py --repo "V:\Radiation_Profiles_Pfotzer_Peak_Estimation" --output reproduced_results
```

The refit uses the uploaded logger experiment and the original repository's altitude profile and peak screen. It checks the source fingerprints and reproduces the pressure-only validation runs before fitting and reloading the two final models. June 1 data were reserved for the separate comparison and are outside this training cohort.

- `results/models/` contains the two XGBoost boosters.
- `results/model_manifest.json` records the feature, pressure span, target units, recipe, source commit and model fingerprints.
- `results/held_out_metrics_*.csv` and `results/held_out_predictions.csv` contain validation evidence from the four-flight fits.
- `results/peak_screening.csv` records the peak status for each flight, phase, channel and grid setting.
- `results/validation.json` records the prediction reproduction and peak audit checks.

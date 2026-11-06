# Your next step: inspect the real recordings

Your `PASS: 6/6` result confirmed that all six original files arrived correctly.
The next step is to inspect timing and measurement quality before creating the
table that XGBoost will learn from.

This small add-on contains a diagnostic script, a report already run on those
same six files, and these instructions. It contains no new raw data and no
trained real-data model. Keep your existing project and Python environment.

## Run this in the VS Code PowerShell terminal

1. Download `HERA_Real_Diagnostics_Addon.zip` into
   `V:\Radiation_Profiles_Pfotzer_Peak_Estimation`.
2. Run the following commands. Extract once; do not add `-Force` if existing
   files are reported. Instead, check whether you already extracted this add-on.

```powershell
Set-Location 'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'
Expand-Archive -LiteralPath '.\HERA_Real_Diagnostics_Addon.zip' -DestinationPath '.'
.\.venv\Scripts\python.exe .\scripts\diagnose_real_data.py
```

Use the terminal, not a notebook cell. This command uses your existing `.venv`
and the packages you already installed. It checks the input fingerprints,
inspects all three recordings and creates a new timestamped results folder.
Running it again creates another folder, so earlier results are preserved.

Successful completion ends with:

```text
DIAGNOSTICS_COMPLETE_TRAINING_PENDING
Report: V:\Radiation_Profiles_Pfotzer_Peak_Estimation\results\real_diagnostics_<timestamp>\report.html
Original recordings are unchanged. No training table or model was created.
Keep RUN_TRAINING = False. See the report for the instrument details still needed.
```

The timestamp will be different on your PC. Share the terminal output after the
command finishes. If it reports an error, share the error text; you do not need
to upload all the data again.

## Open the report

In VS Code's Explorer, expand `results`, then the new `real_diagnostics_...`
folder. Right-click `report.html`, choose **Reveal in File Explorer**, then
double-click the file to open it in your browser. It works offline.

You can also open the report already prepared here:

```powershell
Start-Process '.\results\real_diagnostics_reference_20261012\report.html'
```

The included reference report was generated from the same six fingerprints.
It demonstrates the expected findings. Your run records your own package
versions and input fingerprints in `summary.json`.

## What to understand

- **Raw-file check:** confirms which bytes you have.
- **Clock alignment:** makes sure a detector event is paired with the altitude
  and other sensor readings from the same moment.
- **Deadtime:** time the detector was unable to register another event. A rate
  uses counts divided by usable observation time, after the meaning and
  coverage of the recorded deadtime are confirmed.
- **Provisional fit:** a possible explanation supported by the sensor curves;
  it is not yet verified timing or calibration.
- **Hyperparameter optimization:** comes after the training table is checked.
  It chooses XGBoost settings using training flights only. It cannot fix
  incorrect timestamps or incompatible detector measurements.

The main issues found are repeated timer rollovers, backwards displayed
date/time steps, and pressure-based timing fits that vary between portions of
the flight. A small pressure residual does not prove that the time is correct
to the project's 2.5-second matching tolerance.

The October recording also has about 46 minutes between GPS readings that
pass the basic screen, covering ascent from roughly 15.10 to 31.42 km. We must
review the GPS flags or a separately matched tracker record before using that
interval. Filling the whole gap with invented altitude values would give a
misleading training table. The descent data can be assessed separately.

Keep the notebook settings `RUN_TRAINING = False` and `REFIT_ALL_DATA = False`.
Keep the real manifest unverified. Do not copy fitted clock parameters into it
just to enable training. The diagnostic script deliberately creates no
`combined.csv` or `verified_metadata.json`.

## Information needed from the experiment team

You do not have to infer these details yourself. A link to the original code,
configuration or flight notes is useful. For **June 1, 2025; October 4, 2025;
and March 28, 2026**, obtain:

1. **Logger code and clock information:** the source code/version that wrote
   `Timer[S]`, date and RTC; units and rollover behavior; any common start
   marker or known detector/logger offset. In June, the calendar date changes
   near a displayed 20:00, so the date and time may use different conventions.
   An independent GPS/APRS record for the October Claire flight would also
   help investigate its long GPS gap.
2. **Detector acquisition information:** exact CosmicWatch firmware and
   configuration, whether `Deadtime[s]` is cumulative, what it includes, and
   any acquisition pauses, resets, dropped data or changes of recording mode.
   The v2.1 workshop slides already support the cumulative interpretation;
   they do not establish the history of these particular recordings.
3. **Measurement comparability:** which detector and threshold were used,
   detector arrangement/orientation, and whether the intended measurement is
   all triggers or coincidence events. Also confirm logger acceleration and
   magnetic units and whether temperature is ambient or inside the enclosure.

See `docs/HERA_INSTRUMENT_EVIDENCE.md` for the project sources already examined.
No message has been sent to anyone on your behalf.

## Once those details are resolved

We can finish clock correction and exposure handling, review the measured
profiles, save the processed table with its evidence, and run the existing
XGBoost experiments. Entire flights remain separate during testing and tuning.
Three usable flights would allow an exploratory study, with limited evidence
about performance on future flights. Do not claim causation from a predictive
relationship or treat thousands of rows as thousands of independent flights.

For future data, keep separate raw files and detector/flight notes. Evaluate
the saved model on a new flight before using that flight to retrain it.

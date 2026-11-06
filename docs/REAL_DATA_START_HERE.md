# Add the real HERA recordings to your project

This add-on supplies six real source files: one CosmicWatch event file and one
environmental logger file for each of the three original flight dates. It adds
files to your existing project. It does not replace your notebook, Python
environment, original code, configurations, or saved synthetic results.

**These are unchanged raw recordings, not a validated training dataset.**
No real XGBoost model, empirical peak estimate, or real-data accuracy result is
included. The source-file identities and preliminary issues are documented in
`docs/HERA_Data_Check_Report.md`.

## Install the add-on on your Windows PC

1. Save `HERA_Real_Data_Addon.zip` inside your existing project folder:
   `V:\Radiation_Profiles_Pfotzer_Peak_Estimation`.
2. In VS Code, open **Terminal > New Terminal**. Use the PowerShell terminal.
3. Run the following commands in order. Run the extraction only once.

```powershell
Set-Location 'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'
Expand-Archive -LiteralPath '.\HERA_Real_Data_Addon.zip' -DestinationPath '.'
.\.venv\Scripts\python.exe .\scripts\check_real_data.py
```

If extraction reports existing files, do not add `-Force`: share the message
so we can distinguish a completed earlier extraction from other files.
If your terminal is already in the project folder, the Set-Location command
simply confirms the intended location.

An alternative is **Extract All** in Windows File Explorer: set the destination
to your existing project folder itself. Avoid creating an extra nested add-on
or project folder. The `data`, `config`, `docs` and `scripts` directories in
the archive belong directly inside your existing project.

## Expected check result

The checker prints six `OK` lines and then:

```text
PASS: 6/6 raw files match the recorded sources.
Raw-data transfer is verified. Scientific preprocessing is still pending.
This check did not modify the recordings or train a model.
```

This verifies that the exact files inspected here reached your PC. It does not
verify their scientific suitability or fix measurement problems. The checker
uses only Python's standard library; it needs no additional packages.

## Files supplied

| Flight folder | Detector events | Environmental logger |
| --- | --- | --- |
| `data/raw/2025-06-01/` | `Bravo_CW_Top.txt` | `Bravo_CW_Datalogger_Coincident.csv` |
| `data/raw/2025-10-04_Claire/` | `AxLab_M_011.txt` | `AHD009.csv` |
| `data/raw/2026-03-28/` | `TopScint.txt` | `DataLogger.csv` |

The June filenames use underscores for convenient local paths; their contents
are unchanged. `data/raw/HERA_sources.json` records the original Drive names,
source links, sizes, and SHA-256 fingerprints for all six files. Their source
folders identify candidate pairings; exact identity with researchers' earlier
local copies has not been established.

The add-on also contains:

- `scripts/check_real_data.py`: read-only file check.
- `config/flights.real_unverified.json`: a copy of the original raw-data
  template with paths updated to these actual files. Every verification flag
  remains false, and unknown conventions remain unspecified.
- `docs/HERA_Data_Check_Report.md`: the existing preliminary data report.
- This guide, `docs/REAL_DATA_START_HERE.md`.

## What comes after the file check

Keep `RUN_TRAINING = False` and `REFIT_ALL_DATA = False` in the original
notebook for now. Changing `MODE` to `real_hera` alone cannot prepare raw files.
There is no validated `data/processed/run01/combined.csv` yet.

Before real training, the next work is to:

1. Establish the elapsed-timer rollover rule and a reliable time reference,
   then align the detector and logger clocks, including any drift.
2. Confirm the detector's cumulative deadtime meaning, active acquisition
   intervals, detector identity/settings and selected event definition.
3. Flag startup dates, invalid GPS values, gaps and flight phases, while
   preserving the original recordings.
4. Calculate verified count rates and build measured profiles with supported
   altitude coverage.
5. Produce the processed table and matching metadata; only then run the real
   XGBoost experiment with flight-held-out evaluation and inner tuning.

The currently recorded timer and clock issues cannot be resolved by marking
the manifest verified. The template may also need adapter changes once those
conventions are established. Real-data scores and peak interpretations remain
pending. File presence alone does not establish three usable training flights.

When additional flights arrive, preserve their original files separately and
record their sources and detector configuration. Extra detectors from the same
flight do not count as additional independent flights.

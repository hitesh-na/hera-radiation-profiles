# HERA time-matching review add-on

This is an UPDATE for the existing project, not a replacement project or a new
raw-data download. Keep the original project, the real-data add-on, and the
diagnostics add-on installed. Archive ZIP files may be kept as backups; Python
uses their extracted contents.

## What has been done

- Read Annafy's local and Colab notebooks and Emma's CW-analysis code.
- Reproduced an offset-sign error using known-delay tests. In a +60-second
  fixture, the original mapping adds +60 instead of -60 seconds.
- Added a local diagnostic helper with the corrected sign, real overlapping
  intervals, and strict timer-wrap checks. No remote code was edited.
- Compared count-curve offsets and independent pressure candidates on all
  three flights; retained GPS gaps and unreliable estimates in the audit.
- Updated real-data feature sets to use inside-box temperature by its own
  name. Magnetic field, acceleration, and atmospheric density inferred from
  inside-box temperature are excluded. Synthetic demo behavior is unchanged.
- Kept all verification flags false. No real model or verified combined.csv
  has been produced. Calibration is deferred.

## Install in Windows VS Code

1. Save your notebook and leave RUN_TRAINING=False.
2. Put HERA_Time_Matching_Review_Addon.zip directly inside:
   V:\Radiation_Profiles_Pfotzer_Peak_Estimation
3. Open the VS Code PowerShell terminal in that folder and run:

```powershell
Set-Location 'V:\Radiation_Profiles_Pfotzer_Peak_Estimation'
Expand-Archive -LiteralPath '.\HERA_Time_Matching_Review_Addon.zip' -DestinationPath '.'
.\.venv\Scripts\python.exe .\scripts\install_time_matching_review.py
```

The installer checks the supplied code/config versions before changing them,
backs them up, and leaves the raw files and notebooks alone. If it says a file
differs or an earlier add-on is missing, stop and share that message. Do not
add -Force or delete your project to get around the check.

If you already extracted this ZIP using File Explorer, skip Expand-Archive.
Make sure scripts/install_time_matching_review.py and updates/ are directly
inside the main project, not nested inside a second project folder.

4. Reproduce the timing review on your real files:

```powershell
.\.venv\Scripts\python.exe .\scripts\review_time_matching.py
```

No new packages are required. The review uses the libraries already installed
for your project. It creates a NEW results/timing_review_<timestamp> directory
and prints the exact report.html path. Open that report in your browser.

The supplied reference report can be opened immediately after installation:

```powershell
Start-Process '.\results\timing_review_20261014\report.html'
```

5. Restart the Jupyter kernel before later using updated project modules.
   Do not enable training yet. The new files do not change RUN_TRAINING in your
   notebook or turn unverified metadata into verified metadata.

## What to send the project team

Send HERA_Time_Matching_Review.pdf. Ask them to review the lag-sign test and
the disagreements between timing estimates across flight sections. The key
remaining need is an independent timing reference or a justified mapping with
uncertainty for each exact CW/logger pair. Also confirm acquisition continuity
and detector identity/mode; these are separate from amplitude calibration.

Annafy's notebook assumes AHD011 has already been timer-corrected and filtered.
Ask for the preparation code or records if those contain the missing reference.
Do not reuse that dataset's fixed windows or offsets for the three project
flights. The report documents which files were actually checked.

## What the CSV files mean

alignment_candidates.csv contains diagnostic offset estimates, not values to
paste into a verified configuration. Same-clock control shifts reflect shape
differences and counting noise, not separate logger clocks.

*_review_bins_NOT_FOR_TRAINING.csv contains conditional 30-second rates and a
candidate pressure-based time map. Every row is explicitly marked not ready
for training. A nearby GPS reading does not validate the candidate clock map.

gps_supported_intervals.csv shows basic GPS coverage only. October's long
onboard-GPS interval is preserved. An APRS reception gap is a separate issue.

## Later training

Once timing, exposure and detector configuration are reviewed, prepare the
verified table and use config/research.json. Its six variants and three seeds
remain, with hyperparameter search inside held-out-flight validation. The
real-data feature profile is hera_uncalibrated_v1. Future flights should first
test the frozen model, then be included in a separately versioned retraining.

Do not compare the updated real-data feature set with the synthetic demo as
if they were the same scientific experiment. The demo only checks the software.

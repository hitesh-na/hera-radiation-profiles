# HERA data access and preliminary checks

Updated: 11 October 2026  
Project: Radiation Profiles & Pfotzer Peak Estimation

**The six raw files for the three original flight dates have now been downloaded and checked for basic structure. Real-data training has not started.** The remaining work is preprocessing and scientific validation.

## What changed after reading the README

The attached __README points to the broader [Devil Dragon HERA archive](https://drive.google.com/drive/folders/1NdylENhvHx30TF-EzT_Pdtux5pqrAkcu). Its [Data and Launches folder](https://drive.google.com/drive/folders/1p27O2V24nVIN2lZECW2c9IBxb4bYsV50) contains the original flight-date folders. The first link supplied earlier led to the coordinated-launch subfolder, which covered only part of the archive.

**The earlier request for separate March 28 and October 4 folder links is resolved.** The files are available through this archive.

**June 1 and June 8 are separate recordings.** Each now has its own detector and logger files, and the logger dates distinguish them. Their similar filenames initially prompted a date/identity question. The newly located June 1 data matches the date in the handoff. Do not combine or relabel the two flights. The [HAB Launch History](https://docs.google.com/spreadsheets/d/1w2uBBkvhDeYlfad36KOPgdf3UHzYCKIE0MindIPzr8U/edit?usp=drivesdk) also lists both launch dates separately.

The README was read from the supplied attachment. It was not modified. Its broader background statements and suggested clock-alignment methods are not substitutes for validation of these individual recordings.

## Original flight files

| Flight folder | Detector file | Environmental logger file |
| --- | --- | --- |
| 2025-06-01 | [Bravo CW Top.txt](https://drive.google.com/file/d/1BFyiSKFUi_vjM9d-6jTEj7g0MiOMM2De/view?usp=drivesdk) | [Bravo CW Datalogger Coincident.csv](https://drive.google.com/file/d/1iWsvagtcdMcUEoSBZeWTwnVp7bcW9Ibh/view?usp=drivesdk) |
| 2025-10-04_Claire | [AxLab_M_011.txt](https://drive.google.com/file/d/16hghPchV17eFsSZjJHaK-V0cA6Y5vwxI/view?usp=drivesdk) | [AHD009.csv](https://drive.google.com/file/d/1XBVeQL53nEtG-aP0c57mR5UM5p-jE9hN/view?usp=drivesdk) |
| 2026-03-28 | [TopScint.txt](https://drive.google.com/file/d/1FcZgDN-qDehrJph5aFkaXhLGDth2yuvw/view?usp=drivesdk) | [DataLogger.csv](https://drive.google.com/file/d/1zORjJ2qWglHbgkOLsGUWW2ePywYg-Ww2/view?usp=drivesdk) |

The March file is named DataLogger.csv in Drive; the old notebook references a locally renamed Mar28-26-DataLogger.csv. Date, column layout, and folder context support this candidate pairing. The exact local copies previously used by the researcher have not been compared byte-for-byte.

All six downloaded sizes matched Drive metadata. Files remain unchanged. All six parsed without missing fields in the inspected tables. This does not establish valid timing, complete physical coverage, or calibrated detector response.

## Preliminary checks

| Flight folder | Detector event rows | Logger rows | Maximum recorded altitude | Backward elapsed-timer jumps |
| --- | ---: | ---: | ---: | ---: |
| 2025-06-01 | 232,931 | 259,413 | 31,854.1 m | 4 |
| 2025-10-04_Claire | 244,351 | 131,929 | 31,454.5 m | 3 |
| 2026-03-28 | 120,808 | 43,713 | 17,530.9 m | 4 |

Altitude maxima above were retained after a basic screen requiring at least four reported satellites, nonzero and in-range coordinates, a valid modern date, and a valid clock format. These are descriptive coverage checks, not a full GPS quality assessment. **Maximum flight altitude is not an estimated Pfotzer peak altitude.**

### Dates and clock issues

- June 1 logger: 248,005 rows dated June 1, 2025; 11,398 dated June 2; ten startup rows dated 1970 or 1999. The date/time convention and day rollover need checking.
- October 4 logger: 131,927 rows dated October 4, 2025, plus two invalid startup dates.
- March 28 logger: all 43,713 rows dated March 28, 2026.
- All three elapsed timers jump backward. These may be counter rollovers, but correction must be documented and checked against an independent time reference.
- The recorded real-time clock also sometimes moves backward: 12,552 adjacent valid-clock pairs in June, 8,525 in October, and 1,731 in March. These counts flag nonmonotonic recorded times, not that many independently proven device resets. Magnitudes, ordering, and clock construction still need investigation.
- Do not sort or merge these recordings blindly by the displayed clock, use row number as time, or assume detector/logger clock offset is zero.

### Detector interpretation

All three CosmicWatch headers label elapsed time and deadtime in seconds. Detector event numbers increase by one throughout each file; detector elapsed time and deadtime do not decrease. The increasing deadtime appears cumulative, but its exact firmware meaning and continuous exposure require verification.

The June top-detector file has 54,390 coincidence-flagged events and 178,541 unflagged events. The October and March selected detector files have only zero coincidence flags. Zero flags do not mean zero detected particles. The March logger also uses Events CW1&2 and Events CW1&2&3 columns, unlike the separate CW1/CW2 columns in June and October. Keep detector configuration and event-selection rules explicit; do not assume these channels are interchangeable.

Counts per second must come from a verified count/exposure calculation. A logger event counter or a column labelled Counts Per Second must not be accepted without checking its definition.

## Additional recordings already inspected

| Candidate | Files inspected | Preliminary finding |
| --- | --- | --- |
| June 8, 2025 | [Bravo CW Top.txt](https://drive.google.com/file/d/19qHEG9DYVb4yIMN0eruACV1WoQs2AjYe/view) and [Bravo CW Datalogger Coincident.csv](https://drive.google.com/file/d/1fkhVhKG5-vIIH9fbd4BupkZJd87rdKAe/view) | 220,385 detector rows; 223,584 logger rows; 27,426.4 m maximum recorded altitude; five backward timer jumps; six invalid startup dates; one incomplete final row. Distinct from June 1. |
| May 31, 2026 folder | [AxLab_M_029.txt](https://drive.google.com/file/d/1CkpcyDuFRTtEZhG_JgmCIaoA1H6k1wD7/view) and [Annafy_DF_1PA025](https://drive.google.com/file/d/1mbPnLqzQcyl2RTDfWuaHSrmI1eqXTfKG/view) | 336,050 detector rows; 56,013 logger rows; 14,906.7 m maximum recorded altitude. The README names May 31, while the logger maximum is recorded on May 30. Date/time convention remains unresolved. |

The incomplete June 8 row ends in a partial pressure value of 99 with subsequent fields absent. Exclude the incomplete record from derived analysis rather than treating 99 as a valid high-altitude pressure.

There are now ten raw files inspected across five candidate recording pairs. Additional folders exist in the broader archive, but not every file or launch has been read or validated. The May 3 and November 15 shortcut destinations were not resolved in the earlier inspection. Ground tests, alternate detectors, native spreadsheet copies, and multiple payload channels must not automatically become additional independent flight groups.

## Work required before training

1. Verify timer rollover rules, date/time conventions, and detector/logger alignment, including any clock drift.
2. Confirm cumulative-deadtime semantics, continuous acquisition intervals, detector identity/settings, and the chosen event definition.
3. Create derived quality flags for startup dates, GPS validity, incomplete rows, gaps, and ground/ascent/descent segments. Preserve raw data.
4. Construct measured CPS profiles and assess whether each flight supports an interior maximum on both sides.
5. Run the existing XGBoost baseline, feature variants, and hyperparameter search with outer flight-held-out evaluation, inner grouped tuning, and the predefined seeds.

Three candidate original flights are available, which could support the planned nested evaluation once validated. Three inspected pairs are not yet three validated training inputs. The project has no real-data accuracy score, selected real-data hyperparameters, or confirmed peak estimate at this stage. The synthetic demonstration in the original ZIP remains labelled synthetic.

## Adding future data

Treat the dataset as growing. Keep a unique recording/flight identifier, source link, file fingerprint, detector metadata, and preprocessing version. Do not count extra sensors from one balloon as independent flights; the final grouping rule should also consider shared launch conditions. Evaluate an existing saved model on genuinely new flights before using them for an updated training run. Preserve prior models and results, then retrain and retune using grouped splits. Additional data does not guarantee better performance.

## Your next step

You do not need to send the three original flight folders again. Data preparation can now continue from the files already obtained. Keep the existing project ZIP; it has not yet been repackaged with real data or a trained real-data model. We will give the beginner VS Code/Jupyter training steps after the input table and configuration are validated.

## Fingerprints for the six newly checked files

SHA-256 values identify the exact source bytes inspected.
- 2025-06-01 / Bravo CW Top.txt: `47fa9d985061388c4f2c22f3de21f31539886e975d44a53cb2fd2c2e2cbab1f1` (11,745,447 bytes)
- 2025-06-01 / Bravo CW Datalogger Coincident.csv: `5988894780a52adbada6c4f74da33c975bf6e338f79257212be42bd58aa07aa0` (46,239,704 bytes)
- 2025-10-04_Claire / AxLab_M_011.txt: `a835363819a16c5b915a7fd910f382285fbfc0b7bce91c9b4a4bf10196ac9497` (12,282,828 bytes)
- 2025-10-04_Claire / AHD009.csv: `325cf07444a6ef8179aeb31463366d9dc532abcbc4152c985ecd034874aa1b6d` (23,091,633 bytes)
- 2026-03-28 / TopScint.txt: `136674a7af9e92541d8a0d025e3468f0a98af4e94cb07a02711f59afd1583972` (6,095,537 bytes)
- 2026-03-28 / DataLogger.csv: `d416bc4330d8709cbbe0f783384e019d10e5dc5fcf01c7c66611abf0c3cef6ed` (7,660,004 bytes)

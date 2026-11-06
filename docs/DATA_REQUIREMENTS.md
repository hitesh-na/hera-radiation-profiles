# Data needed for actual HERA training

The supplied notebooks contain code, plots and truncated table previews. They do not contain complete auditable flight tables. Reconstructing training data from plot pixels or a handful of preview rows would not meet the assignment.

## Paired files referenced in Florini-STAR.ipynb

| Flight | CosmicWatch event file | CLUE / environmental file |
|---|---|---|
| 2026-03-28 | `STAR/Data/Mar28/TopScint.txt` | `STAR/Data/Pre-Everything CSV/Mar28-26-DataLogger.csv` |
| 2025-10-04 | `STAR/Data/Oct4/Claire/AxLab_M_011.txt` | `STAR/Data/Oct4/Claire/AHD009.csv` |
| 2025-06-01 | `STAR/Data/Jun1/Bravo CW Top(1).txt` | `STAR/Data/Jun1/Bravo CW Datalogger Coincident.csv` |

The notebook paths identify the intended sources; they are not accessible in this package. Equivalent original files under different names are fine. Keep raw files unchanged. Additional flights can be appended to the manifest.

## What to include with them

1. Original headers and any recording-start/end or shutdown notes.
2. Meaning and units of event time and cumulative deadtime; confirm whether counters reset and whether the event counter is complete or filtered.
3. Instrument clock origin, any offsets or drift, logger date/timezone and the correct handling of midnight or restarts. The `RTC[H:M:S EST]` label does not alone establish daylight-saving behavior.
4. Which detector/scintillator and coincidence configuration produced the authoritative target; gain, threshold, geometry and changes across flights.
5. Altitude source and units; pressure, temperature, acceleration and magnetic units/calibration.
6. Independently measured exposure intervals if available. They are preferable to interpolating cumulative deadtime at event timestamps.

## Accepted combined-table alternative

Provide one CSV with at least `flight_id`, `Timestamp_s`, `Counts_Per_Second`, and `Altitude[m]`, with all source/unit/synchronization conventions documented in the metadata template. The baseline is strongest when pressure, temperature, acceleration magnitude, magnetic magnitude, latitude and longitude are also present. Missing optional sensors are retained as missing and reported.

Prefer `Raw_Counts`, `Live_Time_s`, `Rate_Error`, `match_time_error_s`, `detector_id` and `phase`. If counts are supplied, verified exposure is mandatory and CPS must agree with counts/exposure. Supply a single authoritative detector stream per independent flight; do not split one flight into fake independent groups or duplicate the same exposure under different IDs.

| Column | Required condition |
|---|---|
| `flight_id` | Unique physical flight, not a row, phase or detector pseudo-group |
| `Timestamp_s` | Finite seconds from a documented per-flight reference; one row per detector interval |
| `Counts_Per_Second` | Nonnegative observed detector rate, never a fitted trend |
| `Altitude[m]` | Finite measured altitude with known coordinate/calibration |
| `Pressure[Pa]` | Pascals; convert hPa to Pa explicitly if needed |
| `Temperature[C]` | Celsius; no unknown Kelvin/Celsius mix |
| `Raw_Counts`, `Live_Time_s` | Integer counts and positive verified exposure, both from the same interval |

At least three usable independent flights are needed for the default nested evaluation. Training software cannot create missing altitude coverage or establish a Pfotzer maximum in a flight that did not bracket one.

## Raw adapter choices

`exposure_table` counts events in independently verified non-overlapping `[start,end)` intervals with live time in seconds. Known valid zero-count intervals remain valid zeros.

`cumulative_linear` interpolates a verified cumulative deadtime counter at full time-bin boundaries, then computes `wall time - change in cumulative deadtime`. It **requires** confirmed continuous acquisition and an explicit maximum counter-record gap. Long gaps are excluded because an absence of events cannot by itself distinguish quiet counting from shutdown. Interpolation is an approximation and must be justified against the actual logger semantics. The adapter does not automatically repair resets or missing event-counter values.

Before real training, the template's `null` units, unknown detector fields and verification flags must be filled from evidence. The template is a request for those facts, not a set of guessed measurement conventions.

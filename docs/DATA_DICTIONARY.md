# Data dictionary

## Canonical input and audit fields

| Field | Units / meaning | Use |
|---|---|---|
| `flight_id` | Identifier of an independent physical flight | Grouping only; never a predictor |
| `Timestamp_s` | Documented synchronized seconds | Sensor derivatives and audit; not direct predictor |
| `Counts_Per_Second` | CosmicWatch counts per live second | Regression target |
| `Altitude[m]` | Measured altitude in m | Predictor, profile coordinate |
| `Pressure[Pa]` | Pa | Predictor |
| `Temperature[C]` | Celsius | Predictor |
| `Accel` | Vector magnitude, original verified units | Predictor; no claim of completed calibration |
| `Mag` | Vector magnitude, original verified units | Predictor; no claim of completed calibration |
| `Latitude`, `Longitude` | Degrees | Predictors; missing GPS fixes masked |
| `Raw_Counts` | Nonnegative integer events in a bin | Rate/uncertainty calculation only |
| `Live_Time_s` | Positive verified exposure | Aggregation and counting uncertainty only |
| `Rate_Error` | Approximate sqrt(N)/live-time CPS | Counting diagnostic; zero-count intervals need asymmetric bounds |
| `match_time_error_s` | Absolute sensor/detector midpoint separation | Synchronization audit |
| `sensor_timestamp_s` | Timestamp of the matched environmental record | Join provenance |
| `bin_start_s`, `bin_end_s` | Detector interval boundaries | Exposure audit, left-closed/right-open |
| `phase` | ascent/descent/stationary/unknown | Separate profiles and optional predictor encoding |
| `source_row` | Original processed CSV row index | Trace exclusions and predictions |
| `source_kind` | synthetic or real_hera | Mandatory provenance distinction |
| `detector_id` | Instrument/configuration identifier | Audit only, not a predictor |

## Internal model features

| Feature | Formula / source | Justification and limitation |
|---|---|---|
| `altitude_m`, `pressure_pa`, `temperature_c` | Renamed canonical measurements | Avoid special characters in XGBoost feature names |
| `acceleration`, `magnetic_field`, `latitude`, `longitude` | Renamed canonical measurements | Sensor context; correlated associations are not causal explanations |
| `log_pressure` | ln(P/1 Pa), P positive | Atmospheric pressure varies over orders of magnitude; redundant information is tested by ablation |
| `density_proxy_kg_m3` | P / [287.05 (T + 273.15)] | Dry ideal-air proxy, only meaningful if T represents ambient air; enclosure temperature is a confounder |
| `vertical_speed_m_s` | Gradient of a centered 5-record median altitude with respect to time | Offline sensor derivative, reset at configured time gaps |
| `pressure_change_pa_s` | Corresponding pressure derivative | Describes trajectory/environment, not target-derived trend |
| `phase_code` | ascent=1, descent=-1, stationary=0, unknown=missing | Keeps phase association explicit |

Do not add observed peaks, CPS smoothing/residuals, count-derived features or target-based normalizations to the predictor matrix. Exposure and match quality are not default predictors. No scaler is fitted; trees consume the sensor values and native missing values.

## Profile fields

Bins use `floor(altitude / bin_m)` and fixed origin 0 m. `altitude_m` is the midpoint, not an exact peak coordinate. `observed_alt_min_m/max_m` preserve actual sampled range within each bin. `supported` requires the configured minimum sample count. Missing bins are explicit rows. `rate_cps` is the observed aggregate, `predicted_cps` is the corresponding predicted aggregate. `counting_95_low_cps/high_cps` are conditional equal-tail Poisson intervals when valid count/exposure information exists; they are not total physical uncertainty. A live-time correction dependent on the counting process can invalidate a simple Poisson approximation, so detector behavior still needs review.

## Peak fields

`identified` denotes an interior screened candidate. `edge_limited` denotes a boundary maximum or insufficient coverage on one side. `ambiguous` denotes competing maxima or an excessively broad top. `unresolved` denotes insufficient support, gaps, flat profiles, inadequate shoulder decline or a model candidate outside training altitude support.

`peak_altitude_m`, `peak_altitude_km` and `peak_cps` are null unless identified. `candidate_altitude_m` may retain a diagnostic candidate and must never be presented as an accepted physical peak. `peak_error_m` is predicted minus measured altitude only when both pass. `abs_peak_error_m` is absolute difference, not accuracy against independently known physical truth. Bin spacing and smoothing limit meaningful precision.

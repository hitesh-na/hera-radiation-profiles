# Research log

## Implementation and verification - 2026-10-11 UTC

- Read the complete handoff and inspected all four prior notebooks and the poster.
- Confirmed that no raw CosmicWatch/CLUE files or complete combined HERA tables were supplied. Source copies and hashes are under references/.
- Implemented the XGBoost-only workflow and conservative measurement/peak checks documented in the master guide.
- Generated 3,642 explicitly synthetic rows in three independent fixture groups. The third group deliberately lacks measurements above its generating maximum.
- Completed 54 outer fits (6 variants x 3 flights x 3 seeds), 126 inner candidate/fold fits, and a separate grouped-tuned all-fixture refit.
- Synthetic macro RMSE: baseline 1.793354 CPS; tuned-engineered 1.829127 CPS; difference +0.035773 CPS. This run does not show tuning improvement over baseline. No result was selected by best seed.
- Both incomplete-flight phase profiles were edge-limited, with no accepted measured peak altitude.
- All 26 unit tests passed, including a raw-format fixture with explicit millisecond conversion, offsets and pressure scaling, plus no-leakage and model serialization checks.
- Saved-run validation passed: disjoint outer/inner groups, one prediction per row/variant/seed, absent error values for invalid peak pairs, all figures decodable and a saved model reproducing stored predictions.
- All ten notebook code cells executed in order with zero errors using IPython in-process. A separate Jupyter kernel could not start because this workspace disallows the required local sockets. Notebook syntax, outputs and schema were validated.
- Final-model sensor-only prediction was exercised on an explicitly relabeled synthetic fixture. This is a software interface check, not independent generalization evidence.
- An initial figure write was incomplete; figure export now validates PNG bytes before writing. The final five figures were decoded and visually inspected.
- Real measurements, detector calibration, empirical HERA training and a physical peak-uncertainty analysis remain pending.

## Next empirical milestone

Obtain the paired original detector/environmental files for the three dates in docs/DATA_REQUIREMENTS.md. Verify units and clocks before freezing the real-data protocol. Do not present example metrics or plots as HERA results.

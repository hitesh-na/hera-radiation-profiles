# Diagnostic add-on validation

Completed 2026-10-12 on the six original HERA files listed in
`data/raw/HERA_sources.json`.

- The delivered script completed its full run on 598,090 detector events and
  the corresponding three logger recordings. Every original file fingerprint
  matched before and after the run.
- The candidate rollover arithmetic was checked at a wrap boundary. A normal
  backwards reset, outside the specified wrap region, was rejected.
- A constructed pressure/time example with a known clock slope, offset,
  pressure gain and pressure bias recovered its time mapping to better than
  one second. This checks the implementation's algebra, not the accuracy of
  the fitted clocks in the actual recordings.
- All real-data pressure fits completed numerically without reaching their
  specified parameter bounds. That does not verify clock calibration.
- Diagnostic plots were inspected. The report identifies the long October
  GPS interval, pressure anomalies, and limitations of pressure-derived
  timing. It does not bridge that GPS interval to create training examples.
- The real-data configuration remains unverified. The add-on creates no
  processed training table, verified metadata or real-data model.

The reference report's `summary.json` records the actual script fingerprint,
input fingerprints, run timestamp and installed package versions. Small
optimization differences between package versions are possible; this does not
change the outstanding scientific questions.

The code verification example above is separate from the report. The report
and its three flight plots use the real recordings, not generated flight data.

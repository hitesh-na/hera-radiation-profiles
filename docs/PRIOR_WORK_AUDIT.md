# Audit of supplied research files

Only the six supplied attachments were inspected. No connected account or other Library file was searched.

## Findings

- The handoff is the implementation specification and explicitly allows a synthetic demonstration while raw HERA data are absent.
- Florini-STAR.ipynb contains two definitions of getCosmicWatch_df: one converts time/deadtime from milliseconds, and the later definition leaves those values unscaled.
- The later parser reads time from numeric column 1 and cumulative deadtime from column 5; the six-column legacy HERA parser reads a different layout. Neither layout is assumed valid without raw headers.
- The STAR nearest-time merge has no tolerance and discards duplicate matched detector timestamps after joining. The new code joins environmental records onto unique detector bins with a bounded tolerance.
- The old event-live-time histogram can assign a whole interval crossing bin edges to the last event's bin. The new raw adapter explicitly apportions verified cumulative-counter changes at boundaries, with its interpolation limitation documented.
- The main notebook table displays contain ellipses and head/tail rows, not full source tables. Florini-STAR-Misc adds experiments and plots, not complete raw recordings.
- HERA-Analysis.ipynb contains legacy flight-computer/CosmicWatch analysis and polynomial profile fitting; this package does not use those polynomial fits as observed truth.
- rateGroundPressure.ipynb concerns ground pressure and geometry/acceptance-normalized count rates. Its normalization is not transferred automatically into this CPS project.
- The poster's previous within-flight R2 values are prior reported findings, not a reproduced or new XGBoost result.

## Local reference copies

- `Florini-STAR-Misc.ipynb`: SHA-256 `57564e1894d7320b910ddff0f94a77cfc43c2a37ea95281c75dd955ac5137f4a`
- `Florini-STAR.ipynb`: SHA-256 `f1ef36f46506735d130e1f30b0764fa793beb5c2b14413983d862f5d714e7c4c`
- `HERA-Analysis.ipynb`: SHA-256 `d932cf1c70239b45264b7132d99f6d069ef87e586f4cd117b063d8ee93071d9a`
- `Pfotzer_XGBoost_Handoff_For_Another_AI.md`: SHA-256 `43d9d29a052b622597d137ecd3cb27b39aac8e760aca112c0d3b7c0f56f30a6e`
- `STAR Poster Florini(1).pdf`: SHA-256 `9c683c9ebf2e1d830bfa29455896200710dcce30db573e689a8674adc8df2668`
- `rateGroundPressure.ipynb`: SHA-256 `3a20f248292219100c1aa90ea790219bf72027916c4cb8ef2e503e8cac5c13cf`

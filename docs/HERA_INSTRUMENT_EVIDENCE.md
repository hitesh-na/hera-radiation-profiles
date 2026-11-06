# Instrument evidence inspected for the real-data diagnostics

Reviewed 2026-10-12. This note records evidence and remaining limits; it is not
a verification certificate. Raw-file identities remain in
`data/raw/HERA_sources.json`.

## Evidence available in the project archive

1. **Original detector headers, all three flights.** They identify CosmicWatch
   v2.1 and label event time and deadtime in seconds. The eight columns match
   event number, event time, coincidence flag, ADC, SiPM amplitude, deadtime,
   temperature and pressure. Each file's detector name is the generic `AxLab`.
   A header is useful evidence for units, but does not identify a physical
   detector's serial number, threshold or exact flashed firmware.

2. **[CosmicWatch_v2.1-SpencerSlides.pdf](https://drive.google.com/file/d/1wgRe63oAdWDvv7O5TDoJFIhs6XNm77ks/view).**
   The slides by Spencer Axani describe the same eight quantities and call the
   deadtime cumulative (slide labeled 12). They describe a Raspberry Pi Pico,
   12-bit readout, and an adjustable trigger threshold (slides labeled 12–14).
   This is relevant v2.1 documentation, unlike older v2 firmware with different
   units and fields. It does not prove each flight's acquisition settings.

3. **[plot_cw.py](https://drive.google.com/file/d/1IZt7gO0GIPXhhasL4S71utYBFvXhYStB/view).**
   The archive's analysis script has an eight-column branch that treats the
   event clock as seconds and the sixth column as cumulative deadtime. It
   differences deadtime and subtracts it from elapsed observation time. This
   corroborates the intended column meanings. It is analysis code, not the
   exact acquisition firmware; it was inspected as text and not executed.
   Its binning implementation was not copied into this diagnostic add-on.

4. **[CW Firmware folder](https://drive.google.com/drive/folders/1vSfzS5XVsU1PWRkDYrXm6z34UEjJ7f-l).**
   The inspected listing contains a compiled `CosmicWatch.uf2`. A compiled
   image's presence does not establish that it was flashed for these flights.
   The listing does not supply the logger's timer/date source code or a
   per-flight firmware/configuration record.

5. **[CW Notes 8/13/25](https://docs.google.com/document/d/1UFLu3L4VYbn_voP4xbTJ9nwpZKAOMyGVb73KFjIeb10/edit).**
   These are talk notes about thresholds, saturation and variation among
   detectors. They support checking calibration and configuration, but are not
   a calibration record for the three files. Do not assign the example
   threshold in those notes to every flight.

6. **[June 1 launch sheet](https://docs.google.com/spreadsheets/d/1ZnYwlBS9UmTyBrG5Kedg_VxAksFENbpUJyaOxs9uSOA/edit).**
   It contains payload logistics but does not resolve the required timing,
   detector settings or exposure details. Its balloon labels alone should not
   be used to assign a calibrated detector identity to the `Bravo` filenames.

7. **[October Claire folder](https://drive.google.com/drive/folders/1iLEe4GuxJoedomwQ4wtWsiVgvL7OdWJp).**
   The inspected listing contains the AHD009 logger in CSV and Google Sheets
   forms and the AxLab detector event file. It does not list a separate tracker
   record to fill the identified GPS coverage gap. A record elsewhere may
   still exist; its payload identity and clock would need to be matched.

## Direct numerical observations

- Detector timestamps strictly increase and event IDs advance by one in all
  three recordings. Recorded deadtime never decreases, and no deadtime
  increment exceeds its corresponding elapsed interval. These facts are
  consistent with continuous cumulative counters; they do not prove that all
  acquisition interruptions would be represented by those counters.
- Logger timer drops occur four, three and four times, respectively. Adding
  `2**32 / 1_000_000 = 4294.967296` seconds at each qualifying drop produces
  increasing candidate clocks. This remains a hypothesis about the logger.
- Logger displayed date/time has 12,552, 8,525 and 1,731 backwards steps among
  adjacent rows with plausible dates. June's date advances near 20:00.
- The two pressure traces broadly agree after provisional time and pressure
  response adjustments. Fits made on separate flight portions differ.
  Shared pressure structure is useful evidence, but cannot independently
  distinguish timing error from sensor lag or calibration differences.
- After a basic GPS validity screen, October has a 2,771.69-second interval
  between accepted readings, from about 15,096.7 m to 31,424.1 m on ascent.
  This is about 46.2 minutes on the candidate logger clock. It must not be
  silently interpolated to create an apparent measured ascent profile.
- The coincidence flag is one for 54,390 June detector rows and zero for every
  October and March detector row. This does not mean October/March had zero
  physical coincidences; hardware mode and wiring must be confirmed.
- No detector ADC reading reaches the header's 4095 limit. Analog saturation,
  threshold changes and trigger losses cannot be ruled out from that alone.

The reproducible numerical record is in the included `summary.json`, timer
jump tables, pressure review flags and plotted report. None of these findings
is an XGBoost result or a measured Pfotzer peak.

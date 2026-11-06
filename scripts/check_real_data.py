"""Read-only verification of the six original HERA files in this add-on.

This checks file presence and exact bytes. It does not validate clocks,
exposure, detector calibration, flight coverage, or training readiness.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest = root / "data" / "raw" / "HERA_sources.json"
    if not (root / "pfotzer_xgb").is_dir():
        print("Project code not found beside this add-on.")
        print("Extract the add-on into Radiation_Profiles_Pfotzer_Peak_Estimation,")
        print("the existing folder containing pfotzer_xgb, notebooks and data.")
        return 1
    try:
        records = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    except (OSError, ValueError, KeyError) as error:
        print(f"Cannot read data/raw/HERA_sources.json: {error}")
        return 1

    failures = 0
    for record in records:
        relative = record["relative_path"]
        source = root / relative
        if not source.is_file():
            print(f"MISSING: {relative}")
            failures += 1
            continue
        try:
            digest = hashlib.sha256()
            with source.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            matches = (source.stat().st_size == record["bytes"]
                       and digest.hexdigest() == record["sha256"])
        except OSError as error:
            print(f"CANNOT READ: {relative}: {error}")
            failures += 1
            continue
        if matches:
            print(f"OK: {relative}")
        else:
            print(f"DIFFERENT CONTENT: {relative}")
            failures += 1

    if failures:
        print(f"CHECK FAILED: {failures} of {len(records)} files need attention.")
        print("Keep the files unchanged and share this output before continuing.")
        return 1
    print(f"PASS: {len(records)}/{len(records)} raw files match the recorded sources.")
    print("Raw-data transfer is verified. Scientific preprocessing is still pending.")
    print("This check did not modify the recordings or train a model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

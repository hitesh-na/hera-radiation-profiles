"""Predict HERA recorded count rates from a CSV with pressure in pascals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

BUNDLE = Path(__file__).resolve().parent
PRESSURE_HEADERS = ("Pressure_Pa", "Pressure[Pa]")
RATE_COLUMNS = {"CW12": "Predicted_CW12_cps", "CW123": "Predicted_CW123_cps"}
STATUS = "HERA_pressure_status"


def predict(data: pd.DataFrame, model_directory: Path) -> pd.DataFrame:
    headers = [name for name in PRESSURE_HEADERS if name in data.columns]
    if len(headers) != 1:
        raise ValueError("Provide exactly one pressure column named Pressure_Pa or Pressure[Pa]")
    if data.empty:
        raise ValueError("The input CSV has no data rows")
    if set(data.columns).intersection([STATUS, *RATE_COLUMNS.values()]):
        raise ValueError("The input already has HERA prediction columns")
    manifest = json.loads((model_directory / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest["feature"] != "Pressure_Pa" or manifest["feature_unit"] != "Pa":
        raise ValueError("The model manifest must specify pressure in Pa")
    pressure = pd.to_numeric(data[headers[0]], errors="coerce").to_numpy(float)
    valid = np.isfinite(pressure) & (pressure > 0)
    supported = valid & (pressure >= manifest["training_pressure_min_pa"]) & (
        pressure <= manifest["training_pressure_max_pa"]
    )
    result = data.copy()
    result[STATUS] = np.select(
        [~valid, supported], ["invalid_pressure", "within_training_pressure_span"],
        default="outside_training_pressure_span",
    )
    features = np.ascontiguousarray(pressure[supported].reshape(-1, 1))
    for channel, column in RATE_COLUMNS.items():
        info = manifest["models"][channel]
        model_path = model_directory / info["file"]
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Model fingerprint mismatch for {channel}")
        model = xgb.Booster(params={"nthread": 1}, model_file=model_path)
        if model.num_features() != 1 or model.attr("hera_channel") != channel:
            raise ValueError(f"Model metadata does not match {channel}")
        predictions = np.full(len(result), np.nan)
        if len(features):
            predictions[supported] = np.maximum(model.inplace_predict(features), 0)
        result[column] = predictions
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--models-dir", type=Path, default=BUNDLE / "results",
                        help="Directory containing model_manifest.json and models/")
    args = parser.parse_args()
    try:
        frame = pd.read_csv(args.input_csv)
        predictions = predict(frame, args.models_dir)
        with args.output_csv.open("x", encoding="utf-8", newline="") as destination:
            predictions.to_csv(destination, index=False)
    except (ValueError, OSError, xgb.core.XGBoostError) as exc:
        parser.error(str(exc))
    scored = int(predictions[STATUS].eq("within_training_pressure_span").sum())
    print(f"Wrote {len(predictions)} rows to {args.output_csv} ({scored} scored).")


if __name__ == "__main__":
    main()

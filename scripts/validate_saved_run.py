"""Verify saved scientific audit invariants; run from the project root."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
from xgboost import XGBRegressor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run_directory")
    args=parser.parse_args()
    root=Path(args.run_directory)
    metrics=pd.read_csv(root/"cv_metrics/per_flight_seed.csv")
    splits=pd.read_csv(root/"cv_metrics/split_audit.csv")
    search=pd.read_csv(root/"cv_metrics/hyperparameter_search.csv")
    preds=pd.read_csv(root/"cv_metrics/heldout_predictions.csv")
    peaks=pd.read_csv(root/"peak_tables/heldout_peaks_and_profile_metrics.csv")
    cfg=json.loads((root/"run_config.json").read_text())
    features=pd.read_csv(root/"quality_reports/validated_features.csv")
    expected=len(cfg["variants"])*len(cfg["seeds"])*features.flight_id.nunique()
    assert len(metrics)==expected, "Missing outer fits"
    for _,row in splits.iterrows():
        assert row.flight_id not in row.train_flights.split("|"), "Outer leakage"
    for _,row in search.iterrows():
        train=set(row.inner_train_flights.split("|"))
        assert row.outer_flight not in train
        assert row.inner_validation_flight not in train
        assert row.inner_validation_flight != row.outer_flight
    for _,g in preds.groupby(["variant","seed"]):
        assert len(g)==len(features)
        assert g.source_row.nunique()==len(features)
    assert peaks.loc[~peaks.both_identified,"abs_peak_error_m"].isna().all()
    for fig in (root/"figures").glob("*.png"):
        with Image.open(fig) as im: im.load()
    # Replay one outer model against its exact stored feature rows.
    model_meta=next((root/"model_artifacts").glob("*.metadata.json"))
    meta=json.loads(model_meta.read_text())
    model=XGBRegressor();model.load_model(str(model_meta).replace(".metadata.json",".json"))
    d=features[features.flight_id==meta["flight_id"]]
    replay=np.maximum(model.predict(d[meta["features"]]),0)
    saved=preds[(preds.flight_id==meta["flight_id"])&(preds.variant==meta["variant"])&(preds.seed==meta["seed"])]
    expected_pred=saved.set_index("source_row").loc[d.source_row,"prediction_cps"].to_numpy()
    np.testing.assert_allclose(replay,expected_pred,rtol=1e-6,atol=1e-6)
    print(f"PASS: {expected} complete outer fits; disjoint splits; exactly-once predictions; rejected peak errors absent; PNGs valid; saved model replays.")


if __name__=="__main__": main()

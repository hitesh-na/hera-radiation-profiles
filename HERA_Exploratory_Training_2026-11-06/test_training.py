import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from train_exploratory import (equal_flight_weights, feature_frame, fit_model,
                               contribution_values, verify_inputs, ROOT, VARIANTS)


class TrainingTests(unittest.TestCase):
    def test_equal_total_training_weight_for_unequal_flights(self):
        groups = np.array(['a'] * 10 + ['b'] * 3 + ['c'] * 27)
        weights = equal_flight_weights(groups)
        totals = pd.Series(weights).groupby(groups).sum().to_numpy()
        np.testing.assert_allclose(totals, totals[0])
        self.assertAlmostEqual(float(weights.mean()), 1)

    def test_sensor_features_ignore_radiation_and_never_bridge_gap(self):
        f = pd.DataFrame({'flight_id': ['a'] * 5, 'bin_index': [0, 1, 2, 8, 9],
                          'Timestamp_s': [15, 45, 75, 255, 285], 'Altitude[m]': [1000, 1100, 1200, 5000, 8000],
                          'Pressure[Pa]': [90000, 89000, 88000, 50000, 40000], 'Box_Temperature[C]': 20,
                          'Latitude': 40, 'Longitude': -75, 'Rate_CW12_cps': 1})
        a = feature_frame(f)
        f['Rate_CW12_cps'] = 100000
        pd.testing.assert_frame_equal(a, feature_frame(f))
        self.assertTrue(a.loc[[3, 4], 'Vertical_speed_m_s'].isna().all())

    def test_exact_restored_cohort_and_split_isolation(self):
        d, p, splits, _ = verify_inputs(ROOT / 'inputs')
        self.assertEqual(len(d), 918)
        self.assertEqual(len(splits), 20)
        self.assertFalse(d.training_verified.any())
        self.assertEqual(d.flight_id.nunique(), 5)

    def test_no_counter_time_or_flight_identifier_features(self):
        forbidden = ['Rate_', 'Event_', 'Timestamp', 'flight_id', 'Battery', 'Acceleration', 'Mag']
        for columns in VARIANTS.values():
            self.assertFalse(any(word in col for col in columns for word in forbidden))

    def test_native_shap_reconstructs_raw_predictions(self):
        x = pd.DataFrame({'altitude': np.arange(60), 'pressure': np.arange(60) ** 2})
        y = np.sin(np.arange(60) / 12) + 4
        model = fit_model(x, y, ['a'] * 30 + ['b'] * 30,
                          {'n_estimators': 12, 'max_depth': 2}, 42)
        shap, raw, error = contribution_values(model, x)
        self.assertEqual(shap.shape, (60, 3))
        self.assertLess(error, 1e-4)

    def test_native_saved_model_preserves_predictions(self):
        x = pd.DataFrame({'altitude': np.arange(30)})
        model = fit_model(x, np.arange(30) / 10, ['a'] * 15 + ['b'] * 15,
                          {'n_estimators': 10, 'max_depth': 2}, 42)
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'model.ubj'
            model.get_booster().save_model(filename)
            restored = xgb.Booster(params={'nthread': 1}, model_file=filename)
            prediction = restored.predict(xgb.DMatrix(x.to_numpy(), nthread=1))
        np.testing.assert_allclose(prediction, model.predict(x.to_numpy()), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

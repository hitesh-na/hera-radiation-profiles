import unittest
import tempfile
from pathlib import Path
import json
import numpy as np
import pandas as pd
from pfotzer_xgb.preprocessing import bin_cumulative_events, bin_with_exposure, synchronize, parse_clock, prepare
from pfotzer_xgb.quality import audit_table


class PreprocessingTests(unittest.TestCase):
    def test_cumulative_live_time_is_apportioned_at_boundaries(self):
        t=np.array([0,1,2,8,9,10.])
        d=bin_cumulative_events(t,.1*t,bin_seconds=5,max_counter_gap_s=10,continuous_acquisition_verified=True)
        np.testing.assert_allclose(d.Live_Time_s,[4.5,4.5])
        np.testing.assert_array_equal(d.Raw_Counts,[3,2])
        self.assertAlmostEqual(d.Live_Time_s.sum(),9)

    def test_unknown_acquisition_refused(self):
        with self.assertRaises(ValueError):
            bin_cumulative_events([0,10],[0,1],bin_seconds=5,max_counter_gap_s=20,continuous_acquisition_verified=False)

    def test_long_gap_is_not_a_zero_rate_measurement(self):
        d=bin_cumulative_events([0,1,9,10],[0,.1,.9,1],bin_seconds=5,max_counter_gap_s=2,continuous_acquisition_verified=True)
        self.assertFalse(d.preprocessing_valid.any())
        self.assertTrue(d.Counts_Per_Second.isna().all())

    def test_impossible_or_reset_counter_rejected(self):
        for dead in [[0,8,9],[0,1,.5]]:
            with self.assertRaises(ValueError):
                bin_cumulative_events([0,5,10],dead,bin_seconds=5,max_counter_gap_s=10,continuous_acquisition_verified=True)

    def test_explicit_exposure_keeps_valid_zero_counts(self):
        e=pd.DataFrame({"bin_start_s":[0,5],"bin_end_s":[5,10],"Live_Time_s":[5,4]})
        d=bin_with_exposure([1,2,3],e)
        self.assertEqual(d.Raw_Counts.tolist(),[3,0])
        self.assertEqual(d.Counts_Per_Second.iloc[1],0)

    def test_no_double_count_at_edges(self):
        e=pd.DataFrame({"bin_start_s":[0,5],"bin_end_s":[5,10],"Live_Time_s":[5,5]})
        d=bin_with_exposure([0,5,10],e)
        self.assertEqual(d.Raw_Counts.tolist(),[1,1])

    def test_time_tolerance_preserves_unmatched_rows(self):
        cw=pd.DataFrame({"Timestamp_s":[2.,20.],"Counts_Per_Second":[1.,2.]})
        clue=pd.DataFrame({"Timestamp_s":[1.,1.,3.],"Altitude[m]":[10.,12.,20.]})
        d=synchronize(cw,clue,1.5)
        self.assertEqual(len(d),2)
        self.assertTrue(np.isnan(d.sensor_timestamp_s.iloc[1]))
        self.assertEqual(d.match_time_error_s.iloc[0],1)

    def test_rtc_rollover_and_small_backward_reset(self):
        cfg={"mode":"rtc_only","column":"t","reference_origin_s":86399,"offset_s":0}
        np.testing.assert_allclose(parse_clock(pd.DataFrame({"t":["23:59:59","00:00:01"]}),cfg),[0,2])
        with self.assertRaises(ValueError):
            parse_clock(pd.DataFrame({"t":["10:00:02","10:00:01"]}),cfg)

    def test_quality_rejects_inconsistent_cps(self):
        d=pd.DataFrame({"flight_id":["A","A"],"Timestamp_s":[0,5],"Altitude[m]":[100,200],
                        "Counts_Per_Second":[2,100],"Raw_Counts":[10,10],"Live_Time_s":[5,5]})
        clean,excluded,_,_=audit_table(d)
        self.assertEqual(len(clean),1)
        self.assertIn("rate_count_exposure_mismatch",excluded.exclusion_reason.iloc[0])

    def test_raw_adapter_units_clocks_and_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            times=np.arange(0,10001,1000)
            np.savetxt(root/"cw.txt",np.c_[np.arange(len(times)),times,times*.1])
            pd.DataFrame({"time":[2500,7500],"height":[1000,1025],"pressure":[900,898]}).to_csv(root/"clue.csv",index=False)
            spec={"flight_id":"FIXTURE","verified":True,"detector_id":"SIMULATED",
                "cosmicwatch":{"path":"cw.txt","time_column":1,"time_unit":"ms","event_id_column":0,
                    "deadtime_column":2,"deadtime_unit":"ms","exposure_method":"cumulative_linear",
                    "continuous_acquisition_verified":True,"max_counter_gap_s":2,"reference_origin_s":0,"offset_s":10},
                "clue":{"path":"clue.csv","clock":{"mode":"numeric","column":"time","unit":"ms","reference_origin_s":0,"offset_s":10},
                    "columns":{"Altitude[m]":"height","Pressure[Pa]":"pressure"},"scales":{"Pressure[Pa]":100}}}
            cfg={"source_kind":"real_hera","verified":True,"time_bin_s":5,"match_tolerance_s":.1,"flights":[spec]}
            (root/"manifest.json").write_text(json.dumps(cfg))
            output=prepare(root/"manifest.json",root/"processed")
            d=pd.read_csv(output)
            np.testing.assert_allclose(d.Timestamp_s,[12.5,17.5])
            np.testing.assert_allclose(d.Live_Time_s,[4.5,4.5])
            np.testing.assert_allclose(d["Pressure[Pa]"],[90000,89800])
            self.assertEqual(d.match_time_error_s.sum(),0)
            self.assertTrue((root/"processed/provenance.json").exists())

    def test_unverified_raw_manifest_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"manifest.json";p.write_text(json.dumps({"source_kind":"real_hera","verified":False}))
            with self.assertRaises(ValueError): prepare(p,Path(folder)/"output")


if __name__=="__main__": unittest.main()

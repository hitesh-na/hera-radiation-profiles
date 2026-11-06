import unittest
import numpy as np
import pandas as pd
from pfotzer_xgb.profiles import altitude_profile,smooth_supported,overlap_table
from pfotzer_xgb.peaks import screen_peak,compare_peaks


def fixture(y):
    h=(np.arange(len(y))+.5)*250
    return pd.DataFrame({"altitude_m":h,"bin_m":250,"supported":True,"n_samples":8,
        "rate_cps":y,"counting_se_cps":.1,"observed_alt_min_m":h-100,"observed_alt_max_m":h+100})


class ProfileTests(unittest.TestCase):
    def test_aggregation_uses_total_exposure_not_mean_of_rates(self):
        d=pd.DataFrame({"flight_id":["A","A"],"phase":["ascent","ascent"],"Altitude[m]":[100,110],
            "Counts_Per_Second":[10,1],"Raw_Counts":[10,9],"Live_Time_s":[1,9]})
        p=altitude_profile(d,min_samples=1)
        self.assertAlmostEqual(p.rate_cps.iloc[0],1.9)

    def test_zero_counts_have_nonzero_upper_interval(self):
        d=pd.DataFrame({"flight_id":["A"],"phase":["ascent"],"Altitude[m]":[100],
            "Counts_Per_Second":[0],"Raw_Counts":[0],"Live_Time_s":[5]})
        p=altitude_profile(d,min_samples=1)
        self.assertEqual(p.counting_95_low_cps.iloc[0],0)
        self.assertGreater(p.counting_95_high_cps.iloc[0],0)

    def test_gaps_are_not_filled_or_bridged(self):
        p=fixture([1,2,3,4,3,2,1,0,0,0])
        p.loc[3,"supported"]=False
        self.assertTrue(np.isnan(smooth_supported(p,"rate_cps",3).iloc[3]))
        result=screen_peak(p)
        self.assertEqual(result["status"],"unresolved")
        self.assertIn("gap",result["reason"])

    def test_bracketed_peak_identified(self):
        h=(np.arange(100)+.5)*250
        p=fixture(2+25*np.exp(-.5*((h-14000)/3000)**2))
        result=screen_peak(p)
        self.assertEqual(result["status"],"identified")
        self.assertLessEqual(abs(result["peak_altitude_m"]-14000),250)

    def test_monotone_is_edge_limited(self):
        result=screen_peak(fixture(np.arange(50)))
        self.assertEqual(result["status"],"edge_limited")
        self.assertIsNone(result["peak_altitude_m"])

    def test_flat_is_unresolved(self):
        self.assertEqual(screen_peak(fixture(np.ones(50)))["status"],"unresolved")

    def test_two_prominent_peaks_are_ambiguous(self):
        h=(np.arange(100)+.5)*250
        y=2+25*np.exp(-.5*((h-7000)/1000)**2)+25*np.exp(-.5*((h-18000)/1000)**2)
        self.assertEqual(screen_peak(fixture(y))["status"],"ambiguous")

    def test_no_peak_error_when_either_unresolved(self):
        result=compare_peaks({"status":"edge_limited","peak_altitude_m":None},{"status":"identified","peak_altitude_m":14000})
        self.assertTrue(np.isnan(result["abs_peak_error_m"]))


if __name__=="__main__": unittest.main()

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from pfotzer_xgb.time_alignment import curve_offset, unwrap_logger_timer, ROLLOVER_S, rate_curve
from pfotzer_xgb.features import engineer, matrix, HERA_FEATURE_SETS
from pfotzer_xgb.quality import audit_table
from pfotzer_xgb.preprocessing import parse_clock, prepare
from pfotzer_xgb.experiments import read_and_check, tune, make_model


class TimingReviewTests(unittest.TestCase):
    def test_shift_sign_with_known_positive_negative_and_zero_delays(self):
        t=np.arange(0.,1500.,5.)
        def shape(x):return np.exp(-((x-350)/40)**2)+.6*np.exp(-((x-1000)/80)**2)
        for delay in [60.,-45.,0.]:
            result=curve_offset(shape(t),shape(t-delay),5.,max_lag_s=120)
            self.assertAlmostEqual(result['offset_s'],-delay,delta=.1)
            self.assertFalse(result['at_search_boundary'])

    def test_nonperiodic_edge_is_not_wrapped(self):
        # The distinctive tail moves beyond the observation window and is lost.
        rng=np.random.default_rng(8);a=rng.normal(size=300);b=np.r_[np.full(12,np.nan),a[:-12]]
        result=curve_offset(a,b,1.,max_lag_s=30)
        self.assertAlmostEqual(result['offset_s'],-12.,delta=.1)
        self.assertEqual(result['overlap_bins'],288)

    def test_constant_series_cannot_claim_a_timing_estimate(self):
        r=curve_offset(np.ones(100),np.arange(100.),5.)
        self.assertEqual(r['status'],'unavailable');self.assertIsNone(r['offset_s'])

    def test_wrap_preserves_elapsed_intervals_and_rejects_reset(self):
        elapsed=np.array([ROLLOVER_S-1,ROLLOVER_S+.25,2*ROLLOVER_S-.2,2*ROLLOVER_S+.1])
        raw=np.mod(elapsed,ROLLOVER_S);t,jumps=unwrap_logger_timer(raw)
        np.testing.assert_allclose(t,elapsed);self.assertEqual(jumps.tolist(),[1,3])
        with self.assertRaisesRegex(ValueError,'high-to-low'):unwrap_logger_timer([100.,101.,0.,1.])
        with self.assertRaises(ValueError):unwrap_logger_timer([1.,1.,2.])
        cfg={'mode':'numeric_wrapped','column':'clock','unit':'s','rollover_s':ROLLOVER_S,
             'reference_origin_s':0.,'offset_s':0.}
        np.testing.assert_allclose(parse_clock(pd.DataFrame({'clock':raw}),cfg),elapsed)

    def test_rate_boundaries_and_deadtime_reset(self):
        t=np.array([0.,5.,10.]);edges=np.array([0.,5.,10.])
        rate,counts,live,good=rate_curve(t,np.ones(3),edges,(0,10),.1*t)
        np.testing.assert_array_equal(counts,[1,1]);np.testing.assert_allclose(live,[4.5,4.5])
        with self.assertRaises(ValueError):rate_curve(t,np.ones(3),edges,(0,10),[0,1,.1])


def example():
    frames=[]
    for f in ['A','B','C']:
        n=40;t=np.arange(n)*5.;h=1000+5*t
        frames.append(pd.DataFrame({'flight_id':f,'Timestamp_s':t,'Altitude[m]':h,
            'Pressure[Pa]':101325*np.exp(-h/7600),'Temperature[C]':20+t/100,
            'Accel':9.81,'Mag':123.,'Latitude':40.,'Longitude':-75.,
            'Counts_Per_Second':2+h/500,'phase':'ascent'}))
    return pd.concat(frames,ignore_index=True)


class FeaturePolicyTests(unittest.TestCase):
    def test_inside_box_temperature_does_not_generate_atmospheric_density(self):
        a=example();clean,*_=audit_table(a)
        d=engineer(clean,temperature_context='inside_box')
        np.testing.assert_allclose(d.box_temperature_c,a['Temperature[C]'])
        self.assertTrue(d.temperature_c.isna().all())
        self.assertTrue(d.density_proxy_kg_m3.isna().all())
        self.assertTrue(engineer(a).density_proxy_kg_m3.notna().all())

    def test_uncalibrated_inputs_and_labels_cannot_change_main_model_matrix(self):
        a=example();b=a.copy()
        b['Accel']=-90000.;b['Mag']=1e12;b['Counts_Per_Second']=99999.;b['Raw_Counts']=999999.
        aa=engineer(a,temperature_context='inside_box');bb=engineer(b,temperature_context='inside_box')
        for variant in HERA_FEATURE_SETS:
            xa=matrix(aa,variant,'hera_uncalibrated_v1');xb=matrix(bb,variant,'hera_uncalibrated_v1')
            pd.testing.assert_frame_equal(xa,xb)
            self.assertFalse(set(xa)&{'magnetic_field','acceleration','density_proxy_kg_m3','temperature_c'})

    def test_tuning_uses_the_reviewed_feature_profile(self):
        frame=engineer(example(),temperature_context='inside_box');frame=frame[frame.flight_id!='C']
        seen=[]
        def factory(params,seed,jobs):
            model=make_model({**params,'n_estimators':4},seed,jobs)
            original=model.fit
            def fit(x,y,**kwargs):
                seen.append(list(x.columns));return original(x,y,**kwargs)
            model.fit=fit;return model
        cfg={'feature_profile':'hera_uncalibrated_v1','search_candidates':1,'search_seed':2026,'n_jobs':1,
             'search_space':{'n_estimators':[4],'max_depth':[2],'learning_rate':[.1]}}
        with patch('pfotzer_xgb.experiments.make_model',side_effect=factory):
            tune(frame,'tuned_engineered',11,cfg,'C')
        self.assertTrue(seen)
        self.assertTrue(all(cols==HERA_FEATURE_SETS['tuned_engineered'] for cols in seen))

    def test_review_table_and_unverified_manifest_do_not_start_training(self):
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'config/research.json').read_text())
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'review.csv';example().assign(source_kind='real_hera_review_only').to_csv(p,index=False)
            meta=Path(temp)/'meta.json';meta.write_text(json.dumps({'source_kind':'real_hera','verified':{}}))
            with self.assertRaisesRegex(ValueError,'verification'):read_and_check(p,cfg,meta)
            with self.assertRaisesRegex(ValueError,'verified=true'):
                prepare(root/'config/flights.real_unverified.json',Path(temp)/'prepared')


if __name__=='__main__':unittest.main()

import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pfotzer_xgb.features import engineer,matrix,FEATURE_SETS
from pfotzer_xgb.experiments import grouped_splits,tune,make_model,flight_weights,BASE,read_and_check


def data():
    frames=[]
    for j in range(3):
        t=np.arange(40)*5.
        h=1000+t*5
        frames.append(pd.DataFrame({"flight_id":f"F{j}","Timestamp_s":t,"Altitude[m]":h,
            "Pressure[Pa]":101325*np.exp(-h/7600),"Temperature[C]":15-h*.0065,
            "Counts_Per_Second":2+h/500,"phase":"ascent"}))
    return pd.concat(frames,ignore_index=True)


class ValidationTests(unittest.TestCase):
    def test_feature_allowlist_has_no_target_or_exposure(self):
        a=data();b=a.copy();b["Counts_Per_Second"]=9999
        b["Raw_Counts"]=333;b["Live_Time_s"]=0.001;b["Rate_Error"]=888
        for variant in FEATURE_SETS:
            pd.testing.assert_frame_equal(matrix(engineer(a),variant),matrix(engineer(b),variant))

    def test_other_flight_cannot_change_derivatives(self):
        a=data();b=a.copy();b.loc[b.flight_id=="F1","Altitude[m]"]*=1000
        ea,eb=engineer(a),engineer(b)
        pd.testing.assert_series_equal(ea.loc[ea.flight_id=="F0","vertical_speed_m_s"],eb.loc[eb.flight_id=="F0","vertical_speed_m_s"])

    def test_inner_outer_groups_disjoint_and_all_tested_once(self):
        d=engineer(data());tested=[]
        for tr,te in grouped_splits(d):
            train,test=d.iloc[tr],d.iloc[te]
            self.assertFalse(set(train.flight_id)&set(test.flight_id))
            tested.extend(te.tolist())
            for it,iv in grouped_splits(train):
                a,b=train.iloc[it],train.iloc[iv]
                self.assertFalse(set(a.flight_id)&set(b.flight_id))
                self.assertFalse(set(test.flight_id)&(set(a.flight_id)|set(b.flight_id)))
        self.assertEqual(sorted(tested),list(range(len(d))))

    def test_tuning_independent_of_outer_target(self):
        d=engineer(data());train=d[d.flight_id!="F2"]
        cfg={"search_candidates":1,"search_seed":2026,"n_jobs":1,
             "search_space":{"n_estimators":[8],"max_depth":[2],"learning_rate":[.1]}}
        before,_=tune(train,"tuned_engineered",11,cfg,"F2")
        d.loc[d.flight_id=="F2","Counts_Per_Second"]=-100000
        after,hist=tune(d[d.flight_id!="F2"],"tuned_engineered",11,cfg,"F2")
        self.assertEqual(before,after)
        self.assertTrue(all("F2" not in row["inner_train_flights"] for row in hist))

    def test_model_roundtrip(self):
        d=engineer(data());x=matrix(d,"baseline")
        model=make_model({**BASE,"n_estimators":10},11,1)
        model.fit(x,d.Counts_Per_Second,sample_weight=flight_weights(d))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"model.json";model.save_model(path)
            loaded=XGBRegressor();loaded.load_model(path)
            np.testing.assert_allclose(model.predict(x),loaded.predict(x),rtol=0,atol=0)

    def test_real_run_requires_verified_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"data.csv";data().to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"audited metadata"):
                read_and_check(p,{"data_mode":"real_hera"})

    def test_synthetic_mode_rejects_unlabeled_real_table(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"data.csv";data().to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"labeled synthetic"):
                read_and_check(p,{"data_mode":"synthetic"})


if __name__=="__main__": unittest.main()

"""Review HERA clock alignment on local copies. Never train or modify raw files."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfotzer_xgb.time_alignment import unwrap_logger_timer, rate_curve, curve_offset
from diagnose_real_data import inspect_flight, map_to_logger, sha256

STATUS = 'TIMING_REVIEW_COMPLETE_TRAINING_PENDING'
VERSION = '1.0'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def known_shift_check():
    """Reproduce the reviewed np.roll/+(-lag) sign convention on known delays.

    Synthetic fixtures here test code only. They are never training flights.
    This reproduces the algebra in both reviewed notebooks; their pure xcorr
    function was also extracted and checked separately during source review.
    """
    t = np.arange(5., 1400., 10.)
    def pulse(x):
        return np.exp(-((x-350)/35)**2) + .7*np.exp(-((x-800)/60)**2) + .3*np.exp(-((x-1050)/20)**2)
    rows=[]
    for delay in [60., -40., 0.]:
        a, b = pulse(t), pulse(t-delay)
        shifts=np.arange(-12, 13)
        scores=np.array([np.corrcoef(a, np.roll(b, int(k)))[0,1] for k in shifts])
        lag=float(shifts[int(np.argmax(scores))]*10)
        actual=curve_offset(a,b,10,max_lag_s=120)
        row={'planted_source_delay_s':delay, 'required_offset_s':-delay,
             'upstream_roll_shift_s':lag, 'upstream_applied_offset_s':-lag,
             'upstream_residual_s':delay-lag,
             'corrected_offset_s':actual['offset_s'],
             'corrected_residual_s':delay+actual['offset_s']}
        assert abs(row['corrected_residual_s']) < .1
        rows.append(row)
    return rows


def source_arrays(events, logger, logger_time):
    result=[{'name':'CW singles', 't':events[:,1], 'weights':np.ones(len(events)),
             'dead':events[:,5], 'coverage':(float(events[0,1]),float(events[-1,1]))}]
    for col in logger:
        if not col.startswith('Events CW'):
            continue
        values=pd.to_numeric(logger[col],errors='raise').to_numpy(float)
        delta=np.diff(values)
        if (delta<0).any() or not np.isfinite(delta).all() or not np.allclose(delta,np.round(delta)):
            raise ValueError(f'Invalid logger cumulative counter: {col}')
        result.append({'name':col, 't':logger_time[1:], 'weights':delta, 'dead':None,
                       'coverage':(float(logger_time[0]),float(logger_time[-1]))})
    if len(result)<2:
        raise ValueError('No logger CW counters available')
    return result


def pressure_windows(logger, tl, detector_t):
    # Choose diagnostic windows from SENSOR data, never radiation peaks.
    z=pd.DataFrame({'t':tl,'p':pd.to_numeric(logger['Pressure[Pa]'],errors='coerce')})
    z=z[np.isfinite(z.p)&z.p.between(100,120000)].copy()
    z['bin']=np.floor(z.t/10)
    z=z.groupby('bin')[['t','p']].median()
    low=z[z.p<85000]
    if len(low)<50:
        raise ValueError('Insufficient pressure structure for flight-window comparison')
    start=max(float(low.t.iloc[0]),float(detector_t[0])+200,float(tl[0])+200)
    stop=min(float(low.t.iloc[-1]),float(detector_t[-1])-200,float(tl[-1])-200)
    turn=float(z.loc[z.p.idxmin(),'t'])
    if not start<turn<stop:
        raise ValueError('Pressure minimum is not inside the diagnostic window')
    return {'whole_flight':(start,stop),'before_pressure_minimum':(start,turn),
            'after_pressure_minimum':(turn,stop)}, turn


def curves_for(sources, start, stop, step):
    start=np.ceil(start/step)*step
    stop=np.floor(stop/step)*step
    edges=np.arange(start,stop+step*.1,step)
    if len(edges)<21:
        raise ValueError('Diagnostic interval too short')
    curves={}
    for s in sources:
        y,counts,_,_=rate_curve(s['t'],s['weights'],edges,s['coverage'],s['dead'],
                                smooth_bins=max(1,round(60/step)))
        curves[s['name']]={'curve':y,'counts':float(counts.sum())}
    return (edges[:-1]+edges[1:])/2,curves,edges


def gps_screen(logger):
    a,lat,lon,sats=[pd.to_numeric(logger[k],errors='coerce').to_numpy(float)
                    for k in ['Altitude[m]','Latitude','Longitude','Sats']]
    return (np.isfinite(a)&np.isfinite(lat)&np.isfinite(lon)&(sats>=4)&(abs(lat)<=90)&(abs(lon)<=180)
            &~((lat==0)&(lon==0))&(a>=-500)&(a<=60000))


def gps_intervals(fid, logger, tl, gps):
    idx=np.flatnonzero(gps)
    groups=np.split(idx,np.flatnonzero(np.diff(tl[idx])>30)+1)
    rows=[]
    for group in groups:
        if len(group)<2:
            continue
        rows.append({'flight_id':fid,'logger_start_s':float(tl[group[0]]),
                     'logger_end_s':float(tl[group[-1]]),'accepted_rows':len(group),
                     'min_altitude_m':float(logger['Altitude[m]'].iloc[group].min()),
                     'max_altitude_m':float(logger['Altitude[m]'].iloc[group].max()),
                     'rule':'basic_GPS_screen_and_no_accepted_reading_gap_above_30s',
                     'training_ready':False})
    return rows


def review_bins(fid, events, logger, tl, gps, clock, out):
    # Full 30-second bins; these remain review products with no training label.
    td=events[:,1]
    edges=np.arange(np.ceil(td[0]/30)*30,np.floor(td[-1]/30)*30+1,30)
    rate,counts,live,good=rate_curve(td,np.ones(len(td)),edges,(td[0],td[-1]),events[:,5])
    centers=(edges[:-1]+edges[1:])/2
    mapped=map_to_logger(centers,clock['fits']['joint']['parameters'],clock['pivot_detector_s'])
    frame=pd.DataFrame({'detector_bin_center_s':centers,'candidate_logger_time_s':mapped,
                        'raw_event_count':counts.astype(int),'conditional_live_seconds':live,
                        'conditional_rate_cps':rate,'counter_interpolation_valid':good})
    max_event_gap=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        a=max(0,np.searchsorted(td,lo)-1);b=min(len(td),np.searchsorted(td,hi)+1)
        max_event_gap.append(float(np.diff(td[a:b]).max()) if b-a>=2 else None)
    frame['max_interevent_gap_s']=max_event_gap
    # NO GPS interpolation. Use only an accepted fix within a finite tolerance.
    sensors=pd.DataFrame({'sensor_logger_time_s':tl[gps],
                           'reported_altitude_m':logger.loc[gps,'Altitude[m]'].to_numpy(float),
                           'logger_pressure_pa':logger.loc[gps,'Pressure[Pa]'].to_numpy(float),
                           'box_temperature_c':logger.loc[gps,'Temperature[C]'].to_numpy(float)})
    frame=pd.merge_asof(frame,sensors,left_on='candidate_logger_time_s',right_on='sensor_logger_time_s',
                         direction='nearest',tolerance=2.5)
    frame['candidate_nearest_fix_distance_s']=(frame.candidate_logger_time_s-frame.sensor_logger_time_s).abs()
    frame['flight_id']=fid
    frame['source_kind']='real_hera_review_only'
    frame['training_ready']=False
    frame['review_status']='clock_alignment_and_exposure_not_verified'
    frame.to_csv(out/f'{fid}_review_bins_NOT_FOR_TRAINING.csv',index=False)
    return {'review_bins':len(frame),'bins_with_candidate_nearby_accepted_gps':int(frame.reported_altitude_m.notna().sum()),
            'training_ready_bins':0}


def inspect_alignment(fid,pair,out,checks):
    numerical=inspect_flight(fid,pair,checks)
    events=np.loadtxt(pair['detector'],comments='#',ndmin=2)
    logger=pd.read_csv(pair['logger'])
    tl,jumps=unwrap_logger_timer(logger['Timer[S]'].to_numpy(float))
    sources=source_arrays(events,logger,tl)
    windows,turn=pressure_windows(logger,tl,events[:,1])
    rows=[]
    for window,(start,stop) in windows.items():
        for step in [5.,10.,20.]:
            centers,curves,edges=curves_for(sources,start,stop,step)
            for reference,source,kind in [(s['name'],'CW singles','CW_vs_logger') for s in sources[1:]] + [
                    (sources[1]['name'],s['name'],'same_logger_clock_control') for s in sources[2:]]:
                a,b=curves[reference],curves[source]
                if min(a['counts'],b['counts'])<100:
                    fit={'status':'unavailable','reason':'fewer_than_100_counts_in_window','offset_s':None,
                         'correlation':None,'at_search_boundary':False,'overlap_bins':0}
                else:
                    fit=curve_offset(a['curve'],b['curve'],step,max_lag_s=180)
                    if fit['correlation'] is not None and fit['correlation'] < .8:
                        fit['status']='rejected_shape_match'
                        fit['reason']='correlation_below_0.8_diagnostic_screen'
                rows.append({'flight_id':fid,'window':window,'window_start_s':float(edges[0]),
                             'window_end_s':float(edges[-1]),'bin_seconds':step,'smoothing_bins':round(60/step),
                             'reference':reference,'source':source,'kind':kind,
                             'reference_count_in_window':a['counts'],'source_count_in_window':b['counts'],**fit})
    clock=numerical['candidate_clock_alignment']
    gps=gps_screen(logger)
    bins=review_bins(fid,events,logger,tl,gps,clock,out)
    intervals=gps_intervals(fid,logger,tl,gps)
    candidate_rows=[r for r in rows if r['kind']=='CW_vs_logger' and r['window']=='whole_flight' and
                    r['status']=='candidate_only' and r['offset_s'] is not None and not r['at_search_boundary']]
    full_range=[float(min(r['offset_s'] for r in candidate_rows)),float(max(r['offset_s'] for r in candidate_rows))] if candidate_rows else None
    summary={'flight_id':fid,'timer_wraps':len(jumps),'windows_s':{k:list(v) for k,v in windows.items()},
             'whole_flight_curve_offset_range_s':full_range,
             'largest_accepted_GPS_gap_minutes':numerical['largest_gap_between_accepted_gps_readings_s']/60,
             'pressure_clock_candidate':clock,'training_ready':False,**bins}
    make_figure(fid,events,logger,tl,gps,sources,windows,rows,clock,out)
    return summary,rows,intervals,numerical


def make_figure(fid,events,logger,tl,gps,sources,windows,rows,clock,out):
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    fig.suptitle(f'{fid} | Timing comparison — diagnostic candidates only',fontsize=15,fontweight='bold')
    a=axes[0,0]
    use=np.arange(0,len(events),max(1,len(events)//4000))
    p=clock['fits']['joint']['parameters'];mapped=map_to_logger(events[use,1],p,clock['pivot_detector_s'])
    a.plot(tl[::max(1,len(tl)//4000)]/60,logger['Pressure[Pa]'].iloc[::max(1,len(tl)//4000)]/1000,label='Logger',lw=1.5,c='black')
    a.plot(events[use,1]/60,events[use,7]/1000,label='CW raw time',alpha=.55,lw=.8,c='#969ca4')
    a.plot(mapped/60,(p[2]*events[use,7]+p[3])/1000,label='CW pressure-fit candidate',lw=.8,c='#b44c22')
    a.set(title='Independent sensor comparison',xlabel='Logger/candidate time (min)',ylabel='Pressure (kPa)');a.legend(fontsize=8)
    a=axes[0,1]
    t,curves,_=curves_for(sources,*windows['whole_flight'],10)
    for name,rec in curves.items():
        y=rec['curve'];scale=np.nanmean(y)
        if rec['counts']>=100 and scale>0:
            a.plot(t/60,y/scale,label=name,lw=1.1)
    a.set_yscale('symlog',linthresh=1)
    a.set(title='Rate shapes on their existing timer axes',xlabel='Time (min)',ylabel='Rate / window mean (symlog scale)');a.legend(fontsize=8)
    a=axes[1,0]
    names=list(windows);colors=['#2469a0','#a65724']
    for j,s in enumerate(sources[1:]):
        xs=[];ys=[];low=[];high=[]
        for i,name in enumerate(names):
            v=[r['offset_s'] for r in rows if r['kind']=='CW_vs_logger' and r['reference']==s['name'] and r['window']==name and r['status']=='candidate_only' and r['offset_s'] is not None]
            if v:
                m=float(np.median(v));xs.append(i+(j-.5)*.12);ys.append(m);low.append(m-min(v));high.append(max(v)-m)
        if ys:a.errorbar(xs,ys,yerr=[low,high],fmt='o',capsize=4,label=s['name'],color=colors[j%len(colors)])
    for i,name in enumerate(names):
        mid=np.mean(windows[name]);shift=float(map_to_logger(mid,p,clock['pivot_detector_s'])-mid)
        a.scatter(i,shift,marker='x',s=65,c='black',label='Pressure fit at window midpoint' if i==0 else None)
    a.axhline(0,c='#777777',lw=.7);a.set_xticks(range(3),['Whole flight','Before P minimum','After P minimum'])
    a.set(title='Offset to ADD to CW time\nBars: 5/10/20 s bin range, not confidence intervals',ylabel='Candidate offset (seconds)');a.legend(fontsize=7)
    if any(r['at_search_boundary'] for r in rows):
        a.text(.02,.02,'Some optima reach the +/-180 s search limit',transform=a.transAxes,fontsize=8,color='#a14b18')
    if any(r['status']=='rejected_shape_match' for r in rows):
        a.text(.02,.02,'Poor shape matches omitted here; retained in CSV',transform=a.transAxes,fontsize=8,color='#a14b18')
    a=axes[1,1]
    step=max(1,len(tl)//5000)
    a.plot(tl[::step]/60,logger['Altitude[m]'].iloc[::step]/1000,c='#cccccc',lw=.7,label='All reported altitude')
    a.plot(tl[::step]/60,np.where(gps,logger['Altitude[m]']/1000,np.nan)[::step],c='#2b7b58',lw=1.1,label='Passes basic GPS screen')
    idx=np.flatnonzero(gps)
    for j in np.flatnonzero(np.diff(tl[idx])>30):a.axvspan(tl[idx[j]]/60,tl[idx[j+1]]/60,alpha=.15,color='#b85d20')
    a.set(title='GPS coverage: shaded gaps are preserved',xlabel='Unwrapped logger time (min)',ylabel='Reported altitude (km)');a.legend(fontsize=8)
    for a in axes.flat:a.grid(alpha=.2)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/f'{fid}_timing_comparison.png',dpi=150);plt.close(fig)


def write_report(out,summaries,rows,sign,provenance):
    table=['| Flight | Timer wraps | Whole-flight count-shape offsets* | Longest accepted GPS gap | Training ready |',
           '| --- | ---: | --- | ---: | --- |']
    for s in summaries:
        r=s['whole_flight_curve_offset_range_s'];v=f'{r[0]:.1f} to {r[1]:.1f} s' if r else 'unavailable'
        table.append(f"| {s['flight_id']} | {s['timer_wraps']} | {v} | {s['largest_accepted_GPS_gap_minutes']:.2f} min | No |")
    lines=['# HERA time-matching review — 14 October 2026','',f'**Status: {STATUS}**','',
      'Six original real-data files were checked before and after the run. No original recordings were changed. '
      'No real-data model was trained. Calibration remains deferred.','',
      '## What changed after the professor’s reply','',
      '- The logger temperature is inside the payload. It is now named box_temperature_c in the real-data feature matrix. '
      'It is not used as outside-air temperature and is not used to calculate atmospheric density.',
      '- Uncalibrated magnetic readings and acceleration with unconfirmed units are excluded from the main real-data feature sets. '
      'The original sensor columns remain in the raw files.',
      '- METEOR APRS temperature is reported to be outside the box. No APRS series has been merged without a confirmed payload/time match.',
      '- Detector amplitude calibration is deferred. Emma’s ground-calibration code was reviewed, not applied.',
      '- The synthetic demonstration remains a software exercise; its feature settings are unchanged.','',
      '## Source review','',
      'Annafy’s local and Colab notebooks expect an already timer-corrected, battery-filtered AHD011 file. '
      'Their fixed search windows refer to that example dataset, not automatically to these three flights. '
      'June and October have Events CW1 / Events CW2 columns; March has Events CW1&2 / Events CW1&2&3.',
      'Emma’s calibration_methods.py explicitly uses 4294.967296 seconds per logger wrap. '
      'This corroborates the earlier numerical hypothesis, but is analysis code rather than proof of the exact flashed firmware. '
      'The local helper rejects backward jumps that do not satisfy the high-to-low wrap rule.',
      'Emma’s calibration workflow fits per-scintillator amplitude peaks in ground/coincidence data and can reuse values from '
      'the same detector. Its example MPVs and gain-temperature fits were not transferred to our recordings.','',
      '## Reproducible sign error in the supplied alignment calculation','',
      'Both reviewed notebooks correlate a reference with np.roll(source, k), then apply the negative of the resulting shift to source timestamps. '
      'For this convention the returned shift itself is the offset to add. In the planted +60 s test, the supplied mapping adds +60 s '
      'instead of -60 s, leaving a +120 s mismatch. The -40 s and zero-delay controls are also recorded. '
      'The original notebook function was extracted and checked without running its download, upload, plotting or file-writing cells.',
      'The local diagnostic implementation uses the corrected sign and real overlapping intervals. It does not wrap the end of '
      'a flight onto its beginning. No GitHub or Drive source was edited.','',
      '## Results on the three real flights','',*table,'',
      '*These are offsets estimated from count-curve shape, across logger channels and 5/10/20 s bins, excluding poor shape matches and boundary optima. '
      'They are not final clock corrections or confidence intervals. The full per-window results are in alignment_candidates.csv.',
      'The estimates change with flight portion, bin size and radiation channel. In some windows the optimum reaches the search boundary, '
      'or a logger counter has too few events to estimate a shift. October CW2 has a different late-burst shape and fails the correlation screen. '
      'Its estimates remain in the CSV but are excluded from the summary range. Even a high correlation is not evidence of timing accuracy.',
      'Two logger counters share one timestamp column. Their apparent relative shift is a useful control for shape and counting effects, '
      'not a real offset between two clocks. We keep that logger clock fixed.',
      'Pressure-based estimates use no radiation counts or peak positions and provide an independent comparison. '
      'They also depend on pressure sensor response and flight portion; pressure residuals are not timing error bars. '
      'The present checks do not establish the project’s 2.5-second synchronization requirement.',
      'October’s approximately 46.2-minute interval is between accepted ONBOARD logger GPS readings. '
      'The professor’s explanation about intermittent APRS reception does not, by itself, resolve that onboard gap. '
      'No altitude is interpolated across it.','',
      '## Files produced and what they can be used for','',
      '- alignment_candidates.csv: corrected-sign count-shape comparisons and same-clock controls. Diagnostic use only.',
      '- gps_supported_intervals.csv: intervals passing the basic GPS screen, split at gaps above 30 seconds. These are not approved training intervals.',
      '- *_review_bins_NOT_FOR_TRAINING.csv: conditional rates in full 30-second bins, with a pressure-fit candidate time map. '
      'Every row has training_ready=False and source_kind=real_hera_review_only. These files are not the project’s combined.csv.',
      '- checks/: fresh raw-data checks, pressure-fit details, timing jumps and GPS coverage evidence.',
      '- source_review.json and validation records: exact source references/hashes and the known-delay test.','',
      'The review-bin rates subtract linearly interpolated cumulative deadtime from the 30-second interval. '
      'That assumes continuous acquisition and the documented cumulative-seconds interpretation. '
      'No verification flag is enabled merely because the counters look consistent. '
      'A GPS value is joined only if an accepted logger fix is within 2.5 seconds of the candidate mapped bin centre; '
      'this nearby-sample distance is NOT proof of clock accuracy.','',
      '## What is still needed before real training','',
      'A defensible timing map for each exact file pair, including its uncertainty and usable intervals, is still needed. '
      'Ask the team to review the sign-test result and the per-flight comparisons. A recorded common event, sync pulse, '
      'known start/end reference, or other independent timing evidence would help distinguish clock offset/drift from different rate shapes. '
      'The preparation that produced AHD011_timer_corrected_battery_filtered.csv is also relevant; it is not supplied by the notebook reader.',
      'Recording continuity/exposure and each file’s detector mode/identity still need review. This is separate from the deferred amplitude calibration. '
      'Keep RUN_TRAINING=False and keep the manifest unverified. Do not fill in the fitted candidate offsets as verified values.',
      'After those items are resolved, create the reviewed training table. Hold out whole flights, tune hyperparameters only on the other flights, '
      'and retain future flights as a new test set. Three flights remain a small exploratory study.','',
      '## Reproducible diagnostic settings','',
      'Windows come from ten-second logger pressure medians: the first/last values below 85,000 Pa, split at the pressure minimum, '
      'bounded by a 200-second recording-edge margin. These are diagnostic choices, not validated operating limits. '
      'Radiation curves use 5, 10 and 20-second bins, roughly 60 seconds of smoothing, ±180-second offset searches, '
      'at least 75% overlapping bins, and at least 100 counts per compared series/window. '
      'Correlations below 0.8 are marked rejected_shape_match; this is a diagnostic screen, not an accuracy guarantee. '
      'No peak location or model score selects the windows or fitted pressure map. '
      'The basic GPS screen and pressure fit follow diagnose_real_data.py; exact inputs, code hashes and versions are recorded.','']
    text='\n'.join(lines)
    for s in summaries:text+=f"\n## {s['flight_id']}\n\n![Timing comparison]({s['flight_id']}_timing_comparison.png)\n"
    (out/'report.md').write_text(text,encoding='utf-8')
    body='<h1>HERA time-matching review</h1><p class="status">Review complete. Real training remains pending.</p>'
    body+='<pre>'+html.escape('\n'.join(lines[4:]))+'</pre>'
    for s in summaries:
        p=out/f"{s['flight_id']}_timing_comparison.png"
        body+=f'<h2>{s["flight_id"]}</h2><img alt="Timing comparison" src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}">'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><title>HERA timing review</title><style>body{max-width:1150px;margin:32px auto;padding:0 20px;font:16px/1.5 system-ui;color:#243746}.status{color:#a14b18;font-weight:700}pre{white-space:pre-wrap;font:15px/1.6 system-ui;background:#f4f6f8;padding:20px}img{width:100%;height:auto}</style><body>'+body+'</body></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    write_json(out/'summary.json',{'status':STATUS,'training_ready':False,'provenance':provenance,
                                 'known_shift_tests':sign,'flights':summaries})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.project_root.resolve()
    manifest_path=root/'data/raw/HERA_sources.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));pairs={};inputs=[]
    for record in manifest['files']:
        p=(root/record['relative_path']).resolve()
        if not p.is_relative_to(root/'data/raw') or not p.is_file() or p.stat().st_size!=record['bytes'] or sha256(p)!=record['sha256']:
            raise ValueError(f"Missing or changed original file: {record['relative_path']}")
        pair=pairs.setdefault(record['flight_id'],{})
        if record['kind'] in pair:raise ValueError('Duplicate flight/kind in source manifest')
        pair[record['kind']]=p
        inputs.append({k:record[k] for k in ['flight_id','kind','relative_path','bytes','sha256']})
    if not pairs or any(set(p)!={'detector','logger'} for p in pairs.values()):raise ValueError('Need detector/logger pairs')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=(args.output or root/'results'/f'timing_review_{stamp}').resolve()
    if not out.is_relative_to(root/'results') or out==root/'results':raise ValueError('Choose a NEW directory under project/results')
    out.mkdir(parents=True,exist_ok=False);checks=out/'checks';checks.mkdir()
    sign=known_shift_check();write_json(out/'known_shift_validation.json',sign)
    provenance={'generated_utc':datetime.now(timezone.utc).isoformat(),'review_version':VERSION,
                'script_sha256':sha256(__file__),'alignment_module_sha256':sha256(root/'pfotzer_xgb/time_alignment.py'),
                'raw_diagnostics_script_sha256':sha256(root/'scripts/diagnose_real_data.py'),
                'manifest_sha256':sha256(manifest_path),'inputs':inputs,
                'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':matplotlib.__version__}}
    print(f'Verified {len(inputs)} original file fingerprints.',flush=True)
    summaries=[];rows=[];intervals=[];numerical=[]
    for fid,pair in sorted(pairs.items()):
        print(f'Reviewing {fid} ...',flush=True)
        summary,records,gps,checks_record=inspect_alignment(fid,pair,out,checks)
        summaries.append(summary);rows.extend(records);intervals.extend(gps);numerical.append(checks_record)
        print(f"  Whole-flight curve offsets: {summary['whole_flight_curve_offset_range_s']}; training ready: False",flush=True)
    if any(sha256(root/r['relative_path'])!=r['sha256'] for r in inputs):raise ValueError('An original file changed during review')
    pd.DataFrame(rows).to_csv(out/'alignment_candidates.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'gps_supported_intervals.csv',index=False)
    write_json(checks/'raw_diagnostics_summary.json',numerical)
    source_review=root/'docs/TIMING_SOURCE_REVIEW.json'
    if source_review.is_file():
        provenance['source_review_sha256']=sha256(source_review)
        (out/'source_review.json').write_bytes(source_review.read_bytes())
    write_report(out,summaries,rows,sign,provenance)
    print(f'\n{STATUS}\nReport: {out / "report.html"}')
    print('Original recordings unchanged. No verified training table or real model created. Keep RUN_TRAINING=False.')
    return 0


if __name__=='__main__':
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as error:
        print(f'TIMING REVIEW STOPPED: {error}',file=sys.stderr)
        sys.exit(1)

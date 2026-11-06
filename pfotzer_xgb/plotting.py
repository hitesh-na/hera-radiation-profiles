from pathlib import Path
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .profiles import smooth_supported

COLORS = ["#147D92", "#E58B30", "#6D62A7"]


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#25384A", "axes.titleweight": "bold",
                         "axes.grid": True, "grid.alpha": .17, "figure.dpi": 120,
                         "savefig.dpi": 180})


def save(fig, path):
    from PIL import Image
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    payload = buffer.getvalue()
    # Decode before writing so a partial PNG cannot become a deliverable.
    with Image.open(io.BytesIO(payload)) as check:
        check.load()
    Path(path).write_bytes(payload)
    plt.close(fig)


def make_figures(data, measured, peaks, out, cfg):
    style()
    mode = "SYNTHETIC SOFTWARE DEMO - NOT HERA RESULTS" if cfg["data_mode"] == "synthetic" else "HERA flight measurements / held-out XGBoost"
    flights = sorted(data.flight_id.unique())
    # The quality plot is a scientific diagnostic, not a performance claim.
    fig, axes = plt.subplots(len(flights), 4, figsize=(17, 3.2*len(flights)), squeeze=False)
    for i, flight in enumerate(flights):
        d = data[data.flight_id == flight]
        axes[i,0].scatter(d.Timestamp_s/60, d.Counts_Per_Second, s=2, alpha=.35, color=COLORS[i%3])
        axes[i,1].plot(d.Timestamp_s/60, d["Altitude[m]"]/1000, lw=1.3, color=COLORS[i%3])
        axes[i,2].scatter(d["Altitude[m]"]/1000, d["Pressure[Pa]"]/1000, s=2, color=COLORS[i%3])
        axes[i,3].scatter(d["Altitude[m]"]/1000, d.Counts_Per_Second, s=2, alpha=.35, color=COLORS[i%3])
        for ax, xlabel, ylabel in zip(axes[i], ["Time (min)", "Time (min)", "Altitude (km)", "Altitude (km)"],
                                      ["Count rate (CPS)", "Altitude (km)", "Pressure (kPa)", "Count rate (CPS)"]):
            ax.set(xlabel=xlabel, ylabel=ylabel, title=flight)
    fig.suptitle(mode + "\nData quality and observed coverage", fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94])
    save(fig, out/"figures/data_quality.png")

    held = pd.read_csv(out/"profile_tables/heldout_profiles.csv")
    selected = held[held.variant == "tuned_engineered"]
    keylist = list(measured.groupby(["flight_id", "phase"]).groups)
    fig, axes = plt.subplots(len(flights), 2, figsize=(14,3.5*len(flights)), squeeze=False)
    for row, flight in enumerate(flights):
        for col, phase in enumerate(["ascent", "descent"]):
            ax = axes[row,col]
            p = measured[(measured.flight_id == flight)&(measured.phase == phase)].reset_index(drop=True)
            if p.empty:
                ax.text(.5,.5,"No supported phase",ha="center",transform=ax.transAxes)
                continue
            x = p.altitude_m/1000
            y = p.rate_cps.where(p.supported)
            ax.plot(x, y, ".", ms=3.5, alpha=.45, color="#25384A", label="Observed bin rate")
            ax.plot(x, smooth_supported(p,"rate_cps",cfg["peak_rules"]["smoothing_bins"]), color="#25384A", lw=1.3, label="Observed, documented smoothing")
            if p.counting_95_low_cps.notna().any():
                ax.fill_between(x, p.counting_95_low_cps.where(p.supported),p.counting_95_high_cps.where(p.supported), color="#25384A",alpha=.09,label="Counting-only 95% interval")
            q = selected[(selected.flight_id == flight)&(selected.phase == phase)]
            if len(q):
                pivot = q.pivot(index="bin_index",columns="seed",values="predicted_cps").reindex(p.bin_index)
                mean = pivot.mean(axis=1).to_numpy()
                mask = p.supported.to_numpy(bool)
                ax.plot(x, np.where(mask,mean,np.nan), color="#E17C29",lw=1.6,label="Held-out XGBoost: seed mean")
                ax.fill_between(x, np.where(mask,pivot.min(axis=1),np.nan),np.where(mask,pivot.max(axis=1),np.nan),color="#E17C29",alpha=.23,label="Seed range (not a confidence interval)")
            peak = peaks[(peaks.flight_id == flight)&(peaks.phase == phase)].iloc[0]
            status = peak.status
            if status == "identified":
                ax.axvline(peak.peak_altitude_m/1000,ls=":",color="#147D92",lw=1.5)
            ax.set(title=f"{flight} | {phase} | observed: {status}", xlabel="Altitude (km)", ylabel="Detector count rate (CPS)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(mode+"\nMeasured profiles and predictions for entirely withheld flights",fontsize=15)
    fig.tight_layout(rect=[0,0,1,.94])
    save(fig,out/"figures/heldout_profiles.png")

    common = pd.read_csv(out/"profile_tables/common_supported_overlap.csv")
    fig, axes = plt.subplots(1,2,figsize=(13,4.3))
    for phase, ax in zip(["ascent","descent"],axes):
        for i,flight in enumerate(flights):
            p=measured[(measured.flight_id==flight)&(measured.phase==phase)]
            q=common[(common.flight_id==flight)&(common.phase==phase)]
            # Reindex to original grid to break lines across absent common bins.
            if not p.empty:
                y=p.bin_index.map(q.set_index("bin_index").rate_cps if not q.empty else pd.Series(dtype=float))
                ax.plot(p.altitude_m/1000,y,color=COLORS[i%3],label=flight,lw=1.3)
        ax.set(title=phase, xlabel="Altitude (km)",ylabel="Observed CPS (uncalibrated)")
        ax.legend(fontsize=8)
    fig.suptitle(mode+"\nAbsolute profiles in altitude bins supported by every flight",fontsize=13)
    fig.tight_layout(rect=[0,0,1,.88])
    save(fig,out/"figures/common_coverage_overlay.png")

    normrows=[]
    fig, axes=plt.subplots(1,2,figsize=(13,4.3))
    for phase,ax in zip(["ascent","descent"],axes):
        valid_peaks=peaks[(peaks.phase==phase)&(peaks.status=="identified")]
        eligible=measured[(measured.phase==phase)&measured.flight_id.isin(valid_peaks.flight_id)]
        if len(valid_peaks)<2:
            ax.text(.5,.5,"Fewer than two identified peaks\nShape comparison unresolved",ha="center",transform=ax.transAxes)
        else:
            wide=eligible[eligible.supported].pivot(index="bin_index",columns="flight_id",values="rate_cps")
            common_idx=wide.dropna(how="any").index
            for i,row in valid_peaks.reset_index(drop=True).iterrows():
                p=eligible[eligible.flight_id==row.flight_id].copy()
                p["normalized_cps"]=p.rate_cps.where(p.bin_index.isin(common_idx))/row.peak_cps
                normrows.append(p[["flight_id","phase","altitude_m","normalized_cps"]])
                ax.plot(p.altitude_m/1000,p.normalized_cps,label=row.flight_id,color=COLORS[i%3])
            ax.legend(fontsize=8)
        ax.set(title=phase,xlabel="Altitude (km)",ylabel="CPS / identified observed peak CPS")
    fig.suptitle(mode+"\nShape comparison: valid peaks and overlapping observed coverage only",fontsize=13)
    fig.tight_layout(rect=[0,0,1,.88])
    save(fig,out/"figures/normalized_shapes.png")
    if normrows:
        pd.concat(normrows).to_csv(out/"profile_tables/normalized_shapes.csv",index=False)

    m=pd.read_csv(out/"cv_metrics/per_flight_seed.csv")
    variants=cfg["variants"]
    fig,ax=plt.subplots(figsize=(11,5))
    for i,flight in enumerate(flights):
        means=m[m.flight_id==flight].groupby("variant").rmse_cps.agg(["mean","min","max"]).reindex(variants)
        x=np.arange(len(variants))+(i-(len(flights)-1)/2)*.16
        ax.errorbar(x,means["mean"],yerr=[means["mean"]-means["min"],means["max"]-means["mean"]],fmt="o",capsize=4,color=COLORS[i%3],label=flight)
    ax.set(xticks=np.arange(len(variants)),xticklabels=[v.replace("_","\n") for v in variants],ylabel="Held-out flight RMSE (CPS)",title=mode+"\nPoints: mean across seeds; whiskers: seed range")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save(fig,out/"figures/xgboost_variants.png")


def write_run_report(out,cfg,summary,peaks,data):
    demo=cfg["data_mode"]=="synthetic"
    lines=["# Radiation Profiles & Pfotzer Peak Estimation", "",
           "> Map continuous atmospheric radiation curves to precisely locate the Pfotzer-Regener maximum and generate standardized profiles for cross-flight comparison.", "",
           "## Status", "",
           "**SYNTHETIC SOFTWARE DEMONSTRATION ONLY. No real HERA training or empirical peak estimates.**" if demo else "Real-data run using the verified source metadata saved in provenance.json.", "",
           f"Processed {len(data):,} rows from {data.flight_id.nunique()} independent flight groups. Used seeds {cfg['seeds']}; six XGBoost feature/tuning variants when running the default protocol. Each test flight was excluded from all inner tuning.", "",
           "## Held-out row prediction", "",
           "Each flight is averaged over seeds first. The table averages those flight means equally. The spread is between flights, not a confidence interval. Negative changes versus baseline indicate improvement.", "",
           "| XGBoost variant | Mean flight RMSE (CPS) | Between-flight SD (CPS) | Mean flight MAE (CPS) | Mean flight R2 | RMSE change vs baseline |",
           "|---|---:|---:|---:|---:|---:|"]
    baseline=float(summary.loc[summary.variant=="baseline","rmse_cps_mean"].iloc[0])
    for _,row in summary.iterrows():
        lines.append(f"| {row.variant} | {row.rmse_cps_mean:.3f} | {row.rmse_cps_std:.3f} | {row.mae_cps_mean:.3f} | {row.r2_mean:.3f} | {row.rmse_cps_mean-baseline:+.3f} |")
    lines += ["", "These differences describe this fixed run. They are not a guarantee of improvement on future flights.", "",
              "## Profile and peak evaluation", "", "| Variant | Mean profile RMSE (CPS) | Both peaks identified / evaluated flight-phase-seed pairs | Mean absolute peak difference (m), identified pairs only |", "|---|---:|---:|---:|"]
    for variant,g in peaks.groupby("variant"):
        err=g.abs_peak_error_m.dropna()
        value=f"{err.mean():.0f}" if len(err) else "not estimable"
        lines.append(f"| {variant} | {g.profile_rmse_cps.mean():.3f} | {int(g.both_identified.sum())}/{len(g)} | {value} |")
    lines += ["", "These peak differences compare two empirical reconstructions, not predictions against an independently known physical peak. Identification coverage is reported to expose selection bias in the error average.", "",
        "![Held-out profiles](figures/heldout_profiles.png)", "",
        "![XGBoost experiments](figures/xgboost_variants.png)", "",
        "## Interpretation limits", "",
        "- Three independent flights give only two training groups per outer fold; inner validation trains on one and validates on one. Seeds do not increase the number of independent flights.",
        "- The model uses environmental measurements from the completed held-out flight, including centered trajectory derivatives. It is an offline reconstruction, not a real-time forecast.",
        "- All observed/predicted profiles use the same bins and observed trajectories. Empty or sparse bins remain unsupported. No curve extends beyond observed altitude coverage.",
        "- The reported identified status means a candidate passes explicit screening rules. Instrument response, calibration and atmospheric interpretation still require review.",
        "- Conditional Poisson counting intervals exclude detector calibration, atmospheric variation and model error. Seed ranges and sensitivity ranges are not scientific confidence intervals.",
        "- Bin widths 100, 250 and 500 m and smoothing widths 1 and 3 bins are saved as sensitivity analyses; none are selected to minimize outer-test error.",
        "- Gain and grouped circular perturbation importances describe associations. Correlated predictors, disrupted sensor relationships and detector differences limit interpretation.",
        "- Absolute count rates are detector CPS, not radiation dose or calibrated flux. Hardware consistency must be reviewed before comparing absolute rates.", "",
        "## Reproducibility", "", "The run includes the exact configuration, package versions, input hash, exclusions, split audit, complete search history, held-out predictions and all fitted outer-fold models. The separate final refit is for future data and has no independent test score.", ""]
    (out/"REPORT.md").write_text("\n".join(lines))

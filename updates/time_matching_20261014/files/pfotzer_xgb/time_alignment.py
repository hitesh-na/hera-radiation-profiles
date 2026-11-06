"""Diagnostic timing helpers. A correlation maximum is not verified clock accuracy."""
from __future__ import annotations

import numpy as np

ROLLOVER_S = 2**32 / 1_000_000


def unwrap_logger_timer(raw, rollover_s=ROLLOVER_S):
    """Apply an explicit period to high-to-low wraps; reject other clock reversals.

    The period is corroborated by Emma's analysis code. A hardware restart can
    imitate a wrap, so successful unwrapping alone does not verify acquisition.
    """
    raw = np.asarray(raw, float)
    if not np.isfinite(rollover_s) or rollover_s <= 0:
        raise ValueError('A positive finite rollover period is required')
    if len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError('Need at least two finite timer readings')
    if ((raw < 0) | (raw >= rollover_s)).any():
        raise ValueError('Raw timer is outside the specified rollover range')
    jumps = np.flatnonzero(np.diff(raw) < 0) + 1
    if len(jumps) and not ((raw[jumps-1] > .7*rollover_s) & (raw[jumps] < .3*rollover_s)).all():
        raise ValueError('Backward step does not satisfy the high-to-low wrap rule')
    t = raw + np.r_[0, np.cumsum(np.diff(raw) < 0)] * rollover_s
    if (np.diff(t) <= 0).any():
        raise ValueError('Unwrapped timer is not strictly increasing')
    return t, jumps


def smooth_curve(values, bins):
    y = np.asarray(values, float)
    if not isinstance(bins, (int, np.integer)) or bins < 1 or bins > len(y):
        raise ValueError('Smoothing width must fit the curve')
    good = np.isfinite(y)
    kernel = np.ones(bins)
    numerator = np.convolve(np.where(good, y, 0), kernel, 'same')
    denominator = np.convolve(good.astype(float), kernel, 'same')
    result = np.divide(numerator, denominator, out=np.full_like(y, np.nan), where=denominator > 0)
    result[~good] = np.nan
    return result


def rate_curve(times, weights, edges, coverage, cumulative_deadtime=None, smooth_bins=1):
    """Full-bin diagnostic rate, with exact [left,right) counting.

    Logger counter increments are attributed to the later sample for comparison
    only. Deadtime interpolation assumes cumulative seconds and continuous
    acquisition; this function does not assert those assumptions are verified.
    """
    t, w, edges = np.asarray(times, float), np.asarray(weights, float), np.asarray(edges, float)
    if len(t) != len(w) or len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(w).all():
        raise ValueError('Invalid diagnostic event/counter arrays')
    if (np.diff(t) <= 0).any() or (w < 0).any() or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
        raise ValueError('Invalid ordering, counts or bin edges')
    positions = np.searchsorted(t, edges, side='left')
    cumulative = np.r_[0., np.cumsum(w)]
    counts = np.diff(cumulative[positions])
    live = np.diff(edges).copy()
    good = (edges[:-1] >= coverage[0]) & (edges[1:] <= coverage[1])
    if cumulative_deadtime is not None:
        dead = np.asarray(cumulative_deadtime, float)
        if len(dead) != len(t) or not np.isfinite(dead).all() or (dead < 0).any():
            raise ValueError('Invalid cumulative deadtime')
        dd, dt = np.diff(dead), np.diff(t)
        if (dd < -1e-9).any() or (dd > dt + 1e-8).any():
            raise ValueError('Deadtime reset or impossible increment; do not clip it')
        good &= (edges[:-1] >= t[0]) & (edges[1:] <= t[-1])
        live -= np.diff(np.interp(edges, t, dead))
    good &= live > 0
    rate = np.divide(counts, live, out=np.full_like(live, np.nan), where=good)
    return smooth_curve(rate, smooth_bins), counts, live, good


def curve_offset(reference, source, step_s, max_lag_s=180., min_overlap_fraction=.75):
    """Estimate the offset to ADD to source timestamps: t_ref = t_source + offset.

    Positive k compares reference[k:] to source[:-k]. Only their real overlap
    is compared; data never wrap around the ends of a recording. Sub-bin
    parabolic interpolation is a numerical estimate, not uncertainty evidence.
    """
    a, b = np.asarray(reference, float), np.asarray(source, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 20:
        raise ValueError('Need two equal curves with at least 20 bins')
    if not np.isfinite([step_s, max_lag_s, min_overlap_fraction]).all() or step_s <= 0 or max_lag_s <= 0:
        raise ValueError('Invalid lag search limits')
    if not 0 < min_overlap_fraction <= 1:
        raise ValueError('Invalid minimum overlap fraction')
    limit = min(int(max_lag_s / step_s), len(a)-1)
    shifts = np.arange(-limit, limit+1)
    scores, overlaps = [], []
    for k in shifts:
        aa, bb = (a[k:], b[:-k]) if k > 0 else ((a[:k], b[-k:]) if k < 0 else (a, b))
        mask = np.isfinite(aa) & np.isfinite(bb)
        overlaps.append(int(mask.sum()))
        aa, bb = aa[mask], bb[mask]
        enough = len(aa) >= max(20, int(np.ceil(min_overlap_fraction * len(a))))
        scores.append(float(np.corrcoef(aa, bb)[0,1]) if enough and np.std(aa)>1e-12 and np.std(bb)>1e-12 else np.nan)
    scores = np.asarray(scores)
    if not np.isfinite(scores).any():
        return {'status':'unavailable', 'reason':'constant_curve_or_insufficient_overlap', 'offset_s':None,
                'correlation':None, 'at_search_boundary':False, 'overlap_bins':0}
    idx = int(np.nanargmax(scores))
    sub = 0.
    if 0 < idx < len(scores)-1 and np.isfinite(scores[idx-1:idx+2]).all():
        y0,y1,y2 = scores[idx-1:idx+2]
        denominator = y0 - 2*y1 + y2
        if denominator < 0:
            sub = float(np.clip(.5*(y0-y2)/denominator, -.5, .5))
    boundary = idx in (0, len(scores)-1)
    return {'status':'candidate_only', 'reason':'search_boundary' if boundary else 'shape_alignment_not_clock_verification',
            'offset_s':float((shifts[idx]+sub)*step_s), 'correlation':float(scores[idx]),
            'at_search_boundary':bool(boundary), 'overlap_bins':overlaps[idx]}

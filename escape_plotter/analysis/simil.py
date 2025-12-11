import numpy as np
from shapely.geometry import Polygon as SPolygon
from matplotlib.patches import Polygon as MplPolygon

from escape_plotter.processing import calc

def compute_path_retracing_distance(before_path, escape_path, conversion_factor=0.072, x_tolerance=30):

    before_points = before_path.copy()
    min_dists = []

    for e_point in escape_path:
        
        if x_tolerance is not None:
            candidates = [
                (i, b_point)
                for i, b_point in enumerate(before_points)
                if abs(b_point[0] - e_point[0]) <= x_tolerance]
            if len(candidates) == 0:
                candidates = [
                (i, b_point)
                for i, b_point in enumerate(before_points)]#
        else:
            candidates = [
                (i, b_point)
                for i, b_point in enumerate(before_points)]
        
        dists = [calc.calc_dist_between_points(e_point, b_point)
            for _, b_point in candidates]

        best_candidate_idx = int(np.argmin(dists))
        min_dist = dists[best_candidate_idx] * conversion_factor

        min_dists.append(min_dist)

    return np.mean(min_dists) if len(min_dists) > 0 else np.nan

def get_hist_percentage(data_list, q, tail="below"):
    
    flat = np.array([v for grp in data_list for v in grp if np.isfinite(v)])
    thr = np.nanpercentile(flat, q if tail == "below" else (100 - q))

    all_counts, sel_counts = [], []
    for g in data_list:
        g = np.asarray(g)
        m = np.isfinite(g)
        vals = g[m]
        all_counts.append(m.sum())
        sel_counts.append(np.sum(vals <= thr) if tail == "below" else np.sum(vals >= thr))

    all_counts = np.asarray(all_counts, dtype=float)
    sel_counts = np.asarray(sel_counts, dtype=float)

    # Convert to composition percentages
    y0 = 100.0 * all_counts / (all_counts.sum() or 1.0)
    y1 = 100.0 * sel_counts / (sel_counts.sum() or 1.0)
    
    return y0, y1

def sliding_window_metrics(trials, half=10, area_conversion=1.0, conv_factor=0.072,
                           min_after_len=5, min_before_len=None):
    """
    Compute per-trial means of:
      distance, area, distance_norm, area_norm,
      after_disp, after_pathlen

    Only includes a window if its 'after' (and optionally 'before') segment
    travels at least the given minimum *displacement* (first→last) in the same
    units as conv_factor outputs (e.g., cm).

    Returns
    -------
    out_dist, out_area, out_dist_norm, out_area_norm, out_after_disp, out_after_pathlen
      each is a list with length == len(trials), containing the per-trial mean over windows.
    """
    out_dist, out_area, out_dist_norm, out_area_norm = [], [], [], []
    out_after_disp, out_after_pathlen = [], []

    for trial in trials:
        coords = [p for p in trial if not np.isnan(p).any()]
        if len(coords) < 2 * half:
            out_dist.append(np.nan); out_area.append(np.nan)
            out_dist_norm.append(np.nan); out_area_norm.append(np.nan)
            out_after_disp.append(np.nan); out_after_pathlen.append(np.nan)
            continue

        dvals = []; avals = []; dnorm = []; anorm = []
        after_disps = []; after_pathlens = []

        for t in range(half, len(coords) - half):
            before = np.asarray(coords[t - half:t], float)
            after  = np.asarray(coords[t:t + half], float)
            if len(before) < 2 or len(after) < 2:
                continue

            # window "after" displacement (first→last) and true path length (sum of steps)
            disp_after  = np.linalg.norm(after[-1]  - after[0]) * conv_factor
            path_after  = (np.sum(np.linalg.norm(after[1:] - after[:-1], axis=1)) * conv_factor)

            # same displacement for "before" (used for optional threshold)
            disp_before = np.linalg.norm(before[-1] - before[0]) * conv_factor

            # apply thresholds (using displacement, to match your original intent)
            if (min_after_len  is not None) and (not (np.isfinite(disp_after)  and disp_after  >= min_after_len)):
                continue
            if (min_before_len is not None) and (not (np.isfinite(disp_before) and disp_before >= min_before_len)):
                continue

            d = compute_path_retracing_distance(before, after)  # your bidirectional average
            a = polygon_area_between_paths(before, after) * area_conversion

            dvals.append(d); avals.append(a)
            dnorm.append(d / disp_after if (disp_after and np.isfinite(disp_after)) else np.nan)
            anorm.append(a / disp_after if (disp_after and np.isfinite(disp_after)) else np.nan)
            after_disps.append(disp_after)
            after_pathlens.append(path_after)

        # per-trial means over windows (NaN if none)
        out_dist.append(np.nanmean(dvals) if dvals else np.nan)
        out_area.append(np.nanmean(avals) if avals else np.nan)
        out_dist_norm.append(np.nanmean(dnorm) if dnorm else np.nan)
        out_area_norm.append(np.nanmean(anorm) if anorm else np.nan)
        out_after_disp.append(np.nanmean(after_disps) if after_disps else np.nan)
        out_after_pathlen.append(np.nanmean(after_pathlens) if after_pathlens else np.nan)

    return out_dist, out_area, out_dist_norm, out_area_norm, out_after_disp, out_after_pathlen


def polygon_area_between_paths(before, stim):
    b = np.asarray(before, float)
    s = np.asarray(stim,   float)
    if b.shape != s.shape:
        raise ValueError("Paths must be the same shape")
    return SPolygon(np.vstack([b, s])).area

def area_drawer(before, stim, ax, area, score):  # used by plot_trial_grid_paths
    b = np.asarray(before, float); s = np.asarray(stim, float)
    ax.plot(b[:, 0], b[:, 1], color="grey", label="Before")
    ax.plot(s[:, 0], s[:, 1], color="blue",  label="Stim")
    ax.add_patch(MplPolygon(np.vstack([b, s]), closed=True, facecolor="red", alpha=0.3))
    if area is not None:
        ax.text(0.05, 0.95, f"Area: {area:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="red",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.3"))
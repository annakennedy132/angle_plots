import numpy as np
from escape_plotter.processing import calc

def calc_tortuosity(locs, min_path_length):
    """Mirror the original inline logic: per-mouse-type tort and total distance."""
    tort, dists = [], []

    for points in locs:
        # keep only valid (x,y) points with no NaNs
        try:
            pts = [p for p in points if hasattr(p, "__len__") and len(p) >= 2 and not np.isnan(p).any()]
        except TypeError:
            continue  # skip non-iterable/invalid trials

        if len(pts) < 2:
            continue

        total = sum(calc.calc_dist_between_points(pts[i], pts[i - 1]) for i in range(1, len(pts)))
        straight = calc.calc_dist_between_points(pts[0], pts[-1])

        if straight >= min_path_length:
            tort.append(total / straight)
            dists.append(total)

    return tort, dists
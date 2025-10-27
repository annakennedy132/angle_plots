import numpy as np
from shapely.geometry import Polygon as SPolygon
from matplotlib.patches import Polygon as MplPolygon

def compute_path_retracing_distance(before_path, escape_path):
    if len(before_path) == 0 or len(escape_path) == 0:
        return np.nan

    min_dists = []
    
    before_path = before_path[::-1]

    for e_point in escape_path:
        min_dist = float('inf')
        for b_point in before_path:
            # Euclidean distance between escape and before point
            dist = np.sqrt((e_point[0] - b_point[0])**2 + (e_point[1] - b_point[1])**2)
            dist = dist * 46.5 / 645
            if dist < min_dist:
                min_dist = dist
        min_dists.append(min_dist)

    mean_min_dist = np.mean(min_dists)  # scale to real-world units
    return mean_min_dist

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
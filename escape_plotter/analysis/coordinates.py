import numpy as np
from scipy.interpolate import interp1d

def get_baseline_stats(coords, max_frames=3600, fps=30):

    times_success = []   # arrays of times (s) for successful trials only
    fail_flags    = []   # arrays of 0/100 per trial (for % failure bar w/ points)

    for trials in coords:
        ts = []
        ff = []
        for trial in trials:
            L = len(trial)
            if L >= max_frames:       # did not find nest (censored)
                ff.append(100.0)
            else:                      # success
                ts.append(L / fps)
                ff.append(0.0)
        times_success.append(np.asarray(ts, dtype=float))
        fail_flags.append(np.asarray(ff, dtype=float))
    
    return times_success, fail_flags


def stretch_trials(trials, target_length=60):
    stretched_trials = []
    for trial in trials:
        # Remove NaN-containing points from trial
        trial = [point for point in trial if not (np.isnan(point).any() if isinstance(np.isnan(point), np.ndarray) else np.isnan(point))] if len(trial) else trial
        trial = np.array(trial, dtype=np.float64)
        orig_len = len(trial)

        if orig_len == 0:
            stretched_trials.append([np.nan] * target_length)
            continue

        if orig_len == 1:
            stretched = np.array([trial[0]] * target_length, dtype=np.float64)
        elif orig_len == target_length:
            stretched = trial
        else:
            x_orig = np.linspace(0, 1, orig_len)
            x_target = np.linspace(0, 1, target_length)

            if trial.ndim == 1:
                f = interp1d(x_orig, trial, kind='linear', fill_value='extrapolate', bounds_error=False)
                stretched = f(x_target)
            else:
                stretched = np.vstack([
                    interp1d(x_orig, trial[:, dim], kind='linear', fill_value='extrapolate', bounds_error=False)(x_target)
                    for dim in range(trial.shape[1])
                ]).T

        # Forward fill NaNs in stretched result
        if stretched.ndim == 1:
            if np.isnan(stretched[0]):
                first_valid = np.nanmin(stretched)
                stretched[0] = first_valid
            for i in range(1, len(stretched)):
                if np.isnan(stretched[i]):
                    stretched[i] = stretched[i - 1]
            stretched_trials.append(stretched.tolist())

        elif stretched.ndim == 2:
            for row in stretched:
                if np.isnan(row[0]):
                    first_valid = np.nanmin(row)
                    row[0] = first_valid
                for i in range(1, len(row)):
                    if np.isnan(row[i]):
                        row[i] = row[i - 1]
            # Append as list of tuples
            stretched_trials.append([tuple(pt) for pt in stretched])


    return stretched_trials

def calculate_arena_coverage(locations, grid_size=20, arena_bounds=(90, 790, 80, 670)):
        xmin, xmax, ymin, ymax = arena_bounds

        x_bins = np.arange(xmin, xmax + grid_size, grid_size)
        y_bins = np.arange(ymin, ymax + grid_size, grid_size)

        total_cells = (len(x_bins) - 1) * (len(y_bins) - 1)  # Total number of grid cells in the arena

        coverage_percentages = []

        for mouse_locs in locations:
            filtered_locs = [loc for loc in mouse_locs if isinstance(loc, (list, tuple)) and len(loc) == 2 and not (np.isnan(loc[0]) or np.isnan(loc[1]))]
            x_coords = [loc[0] for loc in filtered_locs]
            y_coords = [loc[1] for loc in filtered_locs]
            x_grid = np.digitize(x_coords, x_bins)
            y_grid = np.digitize(y_coords, y_bins)
            visited_cells = set(zip(x_grid, y_grid))
            percentage_covered = (len(visited_cells) / total_cells) * 100
            coverage_percentages.append(percentage_covered)

        return coverage_percentages
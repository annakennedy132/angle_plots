import numpy as np
from scipy.interpolate import interp1d

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

def compute_group_msd(trials, fps, max_lag_s=5):
    """
    Compute MSD for a group of trials.

    trials: list of trials, each trial is a sequence of positions [x,y,...]
    fps: frames per second
    max_lag_s: maximum lag in seconds for MSD
    """
    max_lag = int(max_lag_s * fps)
    msds = []

    for trial in trials:
        coords = np.asarray(trial)

        # Assume coords[...,0] = x, coords[...,1] = y
        coords = coords[:, :2]

        # Drop frames with NaNs in x or y
        valid = ~np.isnan(coords).any(axis=1)
        coords = coords[valid]

        T = len(coords)
        if T < 3:
            continue

        L = min(max_lag, T - 1)  # max lag in frames for this trial
        trial_msd = np.empty(L)

        for lag in range(1, L + 1):
            disp = coords[lag:] - coords[:-lag]
            sq = np.sum(disp ** 2, axis=1)
            trial_msd[lag - 1] = np.mean(sq)

        msds.append(trial_msd)

    if not msds:
        return None
    
    # Use the maximum available lag length
    L_max = max(len(m) for m in msds)
    n_trials = len(msds)

    # Pad with NaNs so we can use nanmean/nanstd
    msd_mat = np.full((n_trials, L_max), np.nan)
    for i, m in enumerate(msds):
        msd_mat[i, :len(m)] = m

    # Mean MSD at each lag (over available trials)
    mean_msd = np.nanmean(msd_mat, axis=0)

    # Effective number of trials contributing at each lag
    n_eff = np.sum(~np.isnan(msd_mat), axis=0)

    # SEM at each lag (NaN where fewer than 2 trials contribute)
    sem_msd = np.nanstd(msd_mat, axis=0, ddof=1)
    sem_msd = sem_msd / np.sqrt(n_eff)
    sem_msd[n_eff <= 1] = np.nan

    lags_frames = np.arange(1, L_max + 1)
    lags_sec = lags_frames / fps

    return lags_sec, mean_msd, sem_msd

def fit_msd_alpha(lags_sec, mean_msd, min_lag=0.05, max_lag=1.0):
    """
    Fit MSD exponent alpha from log-log MSD curve.
    Only uses lags between min_lag and max_lag (seconds)
    to avoid noise and plateau.
    """
    lags_sec = np.asarray(lags_sec)
    mean_msd = np.asarray(mean_msd)

    valid = (lags_sec >= min_lag) & (lags_sec <= max_lag)
    if valid.sum() < 3:
        return np.nan

    x = np.log10(lags_sec[valid])
    y = np.log10(mean_msd[valid])

    alpha, _ = np.polyfit(x, y, 1)   # slope = α
    return alpha

def compute_alpha(trials, fps, maxlag, minlag, maxfit):
    res = compute_group_msd(trials, fps, max_lag_s=maxlag)
    if res is None:
        return np.nan
    lags, mean, _ = res
    return fit_msd_alpha(lags, mean, min_lag=minlag, max_lag=maxfit)

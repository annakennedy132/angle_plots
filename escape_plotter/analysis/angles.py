import numpy as np
from scipy.interpolate import interp1d

def stretch_angle_trials(trials, target_length=60):
    stretched_trials = []

    for trial in trials:
        trial = np.array(trial, dtype=np.float64)

        # Remove NaNs
        trial = trial[~np.isnan(trial)]
        orig_len = len(trial)

        if orig_len == 0:
            stretched_trials.append([np.nan] * target_length)
            continue

        if orig_len == 1:
            stretched_trials.append([trial[0]] * target_length)
            continue

        # Interpolation coordinates
        x_orig = np.linspace(0, 1, orig_len)
        x_target = np.linspace(0, 1, target_length)

        # Linear interpolation
        f = interp1d(x_orig, trial, kind='linear', bounds_error=False, fill_value="extrapolate")
        stretched = f(x_target)

        stretched_trials.append(stretched.tolist())

    return stretched_trials

def _to_rad(arr_deg):
    a = np.asarray(arr_deg, dtype=float)
    return np.deg2rad(a[~np.isnan(a)])

def compute_mvl(angle_trials_deg):
    mvl = []
    for trial in angle_trials_deg:
        th = _to_rad(trial)
        if th.size == 0:
            mvl.append(np.nan)
            continue
        C = np.cos(th).sum()
        S = np.sin(th).sum()
        mvl.append(np.hypot(C, S) / th.size)
    return mvl

def heading_autocorr(angle_trials_deg, max_lag):
    Cs = []
    for lag in range(1, max_lag + 1):
        trial_means = []
        for trial in angle_trials_deg:
            a = np.asarray(trial, dtype=float)
            n = a.size - lag
            if n <= 0:
                continue
            t0 = a[:n]
            t1 = a[lag:]
            mask = ~np.isnan(t0) & ~np.isnan(t1)
            if mask.any():
                # cos of difference avoids unwrap issues
                trial_means.append(np.cos(np.deg2rad(t1[mask] - t0[mask])).mean())
        Cs.append(np.nanmean(trial_means) if len(trial_means) else np.nan)
    lags = np.arange(1, max_lag + 1)
    return lags, np.array(Cs)
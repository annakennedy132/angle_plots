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
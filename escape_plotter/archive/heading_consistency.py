"""def plot_heading_consistency(self, max_lag_seconds=5, min_speed=10.0):

        colors = self.colors
        fps = getattr(self, "fps", 30)
        max_lag = int(max_lag_seconds * fps)

        # --- Extract angle trials per mouse type (all frames) ---
        angles_by_type = [
            extract.extract_data_col(
                self.global_angles_file, nested=True,
                data_start=4, data_end=None, mouse_type=mt
            )
            for mt in self.mouse_types
        ]

        # --- Extract speed trials per mouse type ---
        speeds_by_type = [
            extract.extract_data_col(
                self.global_speeds_file, nested=True,
                data_start=4, data_end=None, mouse_type=mt
            )
            for mt in self.mouse_types
        ]

        # --- Build speed-controlled angle trials (moving only) ---
        angles_moving_by_type = _angles.mask_angles_by_speed(
            angles_by_type, speeds_by_type, min_speed=min_speed
        )

        # ======================================================================
        # 1) MVL (all frames)
        # ======================================================================
        mvl_by_type_all = [_angles.compute_mvl(trials) for trials in angles_by_type]

        fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1.5, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax,
            [np.array(v, dtype=float) for v in mvl_by_type_all],
            x_label="Mouse Type",
            y_label="Mean Vector Length (MVL)",
            bar_labels=self.mouse_labels,
            colors=colors,
            ylim=(0, 0.5),
            bar_width=0.1,
            log_y=False,
            error_bars=False
        )

        # ======================================================================
        # 2) MVL (moving only, speed-controlled)
        # ======================================================================
        mvl_by_type_moving = [_angles.compute_mvl(trials) for trials in angles_moving_by_type]

        fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1.5, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax,
            [np.array(v, dtype=float) for v in mvl_by_type_moving],
            x_label="Mouse Type",
            y_label="Mean Vector Length (MVL)",
            bar_labels=self.mouse_labels,
            colors=colors,
            ylim=(0, 0.5),
            bar_width=0.1,
            log_y=False,
            error_bars=False
        )

        # ======================================================================
        # 3) Heading autocorrelation & half-life (all frames)
        # ======================================================================
        lags = None
        C_by_type_all, half_lives_all = [], []
        for trials in angles_by_type:
            lags, C = _angles.heading_autocorr(trials, max_lag=max_lag)
            C_by_type_all.append(C)
            hl = _angles.corr_half_life(lags, C, level=0.5)
            half_lives_all.append(hl)

        # Plot C(tau) – all frames
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        for i, C in enumerate(C_by_type_all):
            ax.plot(lags / fps, C, label=self.mouse_labels[i], color=colors[i])
        ax.axhline(0.5, ls='--', lw=1)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Heading autocorrelation C(τ)')
        ax.set_title('Heading Autocorrelation (all frames)')
        ax.legend()
        plt.close()

        # ======================================================================
        # 4) Heading autocorrelation & half-life (moving only)
        # ======================================================================
        lags_m = None
        C_by_type_moving, half_lives_moving = [], []
        for trials in angles_moving_by_type:
            if len(trials) == 0:
                # no moving frames for this mouse type
                C_by_type_moving.append(np.full_like(lags, np.nan) if lags is not None else np.array([]))
                half_lives_moving.append(np.nan)
                continue

            lags_m, C = _angles.heading_autocorr(trials, max_lag=max_lag)
            C_by_type_moving.append(C)
            hl_m = _angles.corr_half_life(lags_m, C, level=0.5)
            half_lives_moving.append(hl_m)

        # Plot C(tau) – moving only
        if lags_m is not None:
            fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
            self.imgs.append(fig)
            for i, C in enumerate(C_by_type_moving):
                if C is None or len(C) == 0:
                    continue
                ax.plot(lags_m / fps, C, label=self.mouse_labels[i], color=colors[i])
            ax.axhline(0.5, ls='--', lw=1)
            ax.set_xlabel('Lag (s)')
            ax.set_ylabel('Heading autocorrelation C(τ)')
            ax.set_title(f'Heading Autocorrelation (moving only, speed ≥ {min_speed})')
            ax.legend()
            plt.close()

        # --- Extract angle trials per mouse type (all frames) ---
        angles_by_type = [
            extract.extract_data_col(
                self.event_angles_file, nested=True,
                data_start=155, data_end=None,
                escape=True, get_escape_index=True,
                escape_col=4, mouse_type=mt
            )[0]   # index 0 = the angle trials
            for mt in self.mouse_types
        ]
        # --- Extract speed trials per mouse type ---
        speeds_by_type = [
            extract.extract_data_col(
                self.event_speeds_file, nested=True,
                data_start=155, data_end=None,
                escape=True, get_escape_index=True,
                escape_col=4, mouse_type=mt
            )[0]   # index 0 = the angle trials
            for mt in self.mouse_types
        ]

        # --- Build speed-controlled angle trials (moving only) ---
        angles_moving_by_type = _angles.mask_angles_by_speed(
            angles_by_type, speeds_by_type, min_speed=min_speed
        )

        # ======================================================================
        # 1) MVL (all frames)
        # ======================================================================
        mvl_by_type_all = [_angles.compute_mvl(trials) for trials in angles_by_type]

        fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1.5, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax,
            [np.array(v, dtype=float) for v in mvl_by_type_all],
            x_label="Mouse Type",
            y_label="Mean Vector Length (MVL)",
            bar_labels=self.mouse_labels,
            colors=colors,
            ylim=(0, 1),
            bar_width=0.1,
            log_y=False,
            error_bars=False
        )

        # ======================================================================
        # 2) MVL (moving only, speed-controlled)
        # ======================================================================
        mvl_by_type_moving = [_angles.compute_mvl(trials) for trials in angles_moving_by_type]

        fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1.5, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax,
            [np.array(v, dtype=float) for v in mvl_by_type_moving],
            x_label="Mouse Type",
            y_label="Mean Vector Length (MVL)",
            bar_labels=self.mouse_labels,
            colors=colors,
            ylim=(0, 1),
            bar_width=0.1,
            log_y=False,
            error_bars=False
        )

        # ======================================================================
        # 3) Heading autocorrelation & half-life (all frames)
        # ======================================================================
        lags = None
        C_by_type_all, half_lives_all = [], []
        for trials in angles_by_type:
            lags, C = _angles.heading_autocorr(trials, max_lag=max_lag)
            C_by_type_all.append(C)
            hl = _angles.corr_half_life(lags, C, level=0.5)
            half_lives_all.append(hl)

        # Plot C(tau) – all frames
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        for i, C in enumerate(C_by_type_all):
            ax.plot(lags / fps, C, label=self.mouse_labels[i], color=colors[i])
        ax.axhline(0.5, ls='--', lw=1)
        ax.set_xlabel('Lag (s)')
        ax.set_ylabel('Heading autocorrelation C(τ)')
        ax.legend()
        plt.close()"""
        

"""# ---------- helpers (put once in a utils file or near your class) ----------
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

def corr_half_life(lags, C, level=0.5):
    C = np.asarray(C, dtype=float)
    finite = np.isfinite(C)
    if finite.sum() < 2:
        return np.nan
    l = lags[finite]
    c = C[finite]
    below = np.where(c <= level)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return l[0]
    l0, c0 = l[i-1], c[i-1]
    l1, c1 = l[i], c[i]
    if np.isclose(c1, c0):
        return l1
    frac = (level - c0) / (c1 - c0)
    return l0 + frac * (l1 - l0)

import numpy as np
from scipy import stats

def framewise_turn_angles(trials_deg):
    dtheta_all = []
    for trial in trials_deg:
        a = np.asarray(trial, float)
        if a.size < 2:
            continue
        # convert to radians
        a_rad = np.deg2rad(a)
        # circular difference using complex representation
        dtheta = np.angle(np.exp(1j * np.diff(a_rad)))
        dtheta_all.append(dtheta)

    if len(dtheta_all) == 0:
        return np.array([], dtype=float)

    return np.concatenate(dtheta_all)


def microturn_metrics(trials_deg):
    dtheta = framewise_turn_angles(trials_deg)  # radians
    if dtheta.size == 0:
        return {
            "mean_abs": np.nan,
            "median_abs": np.nan,
            "std_abs": np.nan,
            "kurtosis": np.nan
        }, np.array([], dtype=float)

    abs_dtheta = np.abs(dtheta)          # radians
    abs_dtheta_deg = np.rad2deg(abs_dtheta)

    metrics = {
        "mean_abs": np.nanmean(abs_dtheta_deg),
        "median_abs": np.nanmedian(abs_dtheta_deg),
        "std_abs": np.nanstd(abs_dtheta_deg),
        "kurtosis": stats.kurtosis(abs_dtheta_deg, fisher=True, bias=False)
    }
    return metrics, abs_dtheta_deg


def mask_angles_by_speed(angles_by_type, speeds_by_type, min_speed):

    angles_moving_by_type = []

    for angle_trials, speed_trials in zip(angles_by_type, speeds_by_type):
        moving_trials = []

        for ang, spd in zip(angle_trials, speed_trials):
            a = np.asarray(ang, float)
            s = np.asarray(spd, float)

            # align lengths safely
            n = min(a.size, s.size)
            if n == 0:
                continue
            a = a[:n]
            s = s[:n]

            # keep only frames with finite values and speed above threshold
            mask = np.isfinite(a) & np.isfinite(s) & (s >= min_speed)
            if np.any(mask):
                moving_trials.append(a[mask])

        angles_moving_by_type.append(moving_trials)

    return angles_moving_by_type

import numpy as np

def _curvature_for_trial(locs, eps=1e-9):

    locs = np.asarray(locs, float)
    if locs.ndim != 2 or locs.shape[0] < 3:
        return np.array([], dtype=float)

    # Use first two columns as x,y if more dims
    if locs.shape[1] > 2:
        locs = locs[:, :2]

    # Vectors AB and BC for points A,B,C
    v1 = locs[1:-1] - locs[:-2]    # shape (T-2, 2)
    v2 = locs[2:]   - locs[1:-1]   # shape (T-2, 2)

    # Norms
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)

    valid = (n1 > eps) & (n2 > eps)
    if not np.any(valid):
        return np.array([], dtype=float)

    v1 = v1[valid]
    v2 = v2[valid]
    n2 = n2[valid]

    # Angle between v1 and v2 (0..pi), via cross and dot
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot   = (v1 * v2).sum(axis=1)
    theta = np.arctan2(np.abs(cross), dot)  # radians, >= 0

    # Discrete curvature ~ angle per step length
    kappa = theta / (n2 + eps)             # units: rad / distance-unit

    return kappa


def curvature_and_rolling(locs_trials, rolling_window=5):

    all_kappa = []
    all_roll  = []

    for locs in locs_trials:
        kappa = _curvature_for_trial(locs)
        if kappa.size == 0:
            continue

        all_kappa.append(kappa)

        # rolling mean of |kappa|
        if rolling_window is not None and rolling_window > 1 and kappa.size >= rolling_window:
            abs_k = np.abs(kappa)
            kernel = np.ones(rolling_window, dtype=float) / rolling_window
            roll = np.convolve(abs_k, kernel, mode='valid')
            all_roll.append(roll)

    if len(all_kappa) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    kappa_all = np.concatenate(all_kappa)
    if len(all_roll) == 0:
        rolling_all = np.array([], dtype=float)
    else:
        rolling_all = np.concatenate(all_roll)

    return kappa_all, rolling_all

import numpy as np

def angular_speed_for_trials(trials_deg, fps, rolling_window=5):

    all_omega = []
    all_roll  = []

    dt = 1.0 / float(fps)

    for trial in trials_deg:
        a = np.asarray(trial, float)
        if a.size < 2:
            continue

        # radians
        a_rad = np.deg2rad(a)

        # circular difference between frames (radians)
        dtheta = np.angle(np.exp(1j * np.diff(a_rad)))

        # |dθ/dt| in deg/s
        omega = np.rad2deg(np.abs(dtheta)) / dt   # = |dθ| * fps, converted to deg
        omega = omega[np.isfinite(omega)]
        if omega.size == 0:
            continue

        all_omega.append(omega)

        # rolling mean of |angular speed|
        if rolling_window is not None and rolling_window > 1 and omega.size >= rolling_window:
            kernel = np.ones(rolling_window, dtype=float) / rolling_window
            roll = np.convolve(omega, kernel, mode='valid')
            roll = roll[np.isfinite(roll)]
            if roll.size > 0:
                all_roll.append(roll)

    if len(all_omega) == 0:
        angspeed_all = np.array([], dtype=float)
    else:
        angspeed_all = np.concatenate(all_omega)

    if len(all_roll) == 0:
        rolling_angspeed = np.array([], dtype=float)
    else:
        rolling_angspeed = np.concatenate(all_roll)

    return angspeed_all, rolling_angspeed
"""
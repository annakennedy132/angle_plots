from pathlib import Path
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from escape_plotter.analysis import (behaviour, coordinates, simil, angles as _angles, tort as _tort, speed as _speed,)
from escape_plotter.processing import extract, plot, calc, escape
from escape_plotter.utils import files

class FinalPlots:
    def __init__(self, folder, settings, mouse_types, save_pdf=True, save_imgs=True, stats=False):

        for k, v in settings["dimensions"].items():
            setattr(self, k, v)
            
        for k, v in settings["video"].items():
            setattr(self, k, v)
            
        self.mouse_types = mouse_types
        self.folder = folder
        image_file = "images/arena.tif"
        self.background_image = mpimg.imread(image_file)
        
        self.colors = ['mediumslateblue', 'navy', 'silver', 'slategray']
        #self.colors = ['mediumslateblue', 'silver', 'cadetblue']
        #self.colors = ['mediumslateblue', 'silver']
        #self.colors = ['navy', 'slategray']
        
        self.mouse_labels=["WT-phot.","WT-scot.","rd1-phot.","rd1-scot."]
        #self.mouse_labels=mouse_types
        
        self.imgs = []
        self.save_pdf = save_pdf
        self.save_imgs = save_imgs
        self.stats=stats

        self.parent_folder = os.path.dirname(self.folder)
        if not os.path.exists(os.path.join(self.parent_folder, "final_plots_output")):
            self.output_folder = files.create_folder(self.parent_folder, "final_plots_output", append_date=False)
        else:
            self.output_folder = os.path.join(self.parent_folder, "final_plots_output")

        patterns = {
            "global_angles_file":    "*global_angles.csv",
            "global_locs_file":      "*global_locs.csv",
            "global_speeds_file":    "*global_speeds.csv",
            "global_distances_file": "*global_distances.csv",
            "after_angles_file":     "*after_angles.csv",
            "during_angles_file":    "*during_angles.csv",
            "prev_esc_locs_file":    "*collated_prev_esc_locs.csv",
            "event_locs_file":       "*event_locs.csv",
            "event_distances_file":  "*event_distances.csv",
            "event_angles_file":     "*event_angles.csv",
            "event_speeds_file":     "*event_speeds.csv",
            "stim_file":             "*stim.csv",
            "escape_stats_file":     "*escape-stats.csv",
            "escape_success_file":   "*collated_escape_success.csv",}

        for attr, pat in patterns.items():
            m = next(Path(self.folder).glob(pat), None)
            setattr(self, attr, str(m) if m else None)

    def plot_coord_data(self):
        colors = self.colors
        
        #--- Extract data ---
        baseline_locs = [extract.extract_data_col(self.global_locs_file, nested=True, data_start=4, data_end=None, process_coords=True, escape_col=None, mouse_type=mt) for mt in self.mouse_types]
        event_locs = [extract.extract_data_col(self.event_locs_file, nested=True, data_start=155, data_end=None, process_coords=True, escape=True, get_escape_index=True, escape_col=4, mouse_type=mt)[0] for mt in self.mouse_types]
        dist_ev = [extract.extract_data_col(self.event_distances_file, data_start=155, escape=True,
                                                get_escape_index=True, escape_col=4, mouse_type=mt)for mt in self.mouse_types]
        dist_ev = [coordinates.stretch_trials(x[0], target_length=450) for x in dist_ev]
        baseline_id = [extract.extract_data_rows(self.global_locs_file, data_row=0,             mouse_type=mt)
                    for mt in self.mouse_types]
        dist_ev_norm = []
        for group in dist_ev:               # group = list of mice within one mouse_type
            group_norm = []
            for trial in group:            # trial = 1D array of timecourse values
                max_val = np.nanmax(trial) if np.nanmax(trial) != 0 else 1
                group_norm.append(trial / max_val)
            dist_ev_norm.append(group_norm)
        
        #flatten for coord heatmaps
        flat_baseline_locs = [sum(mt_list, []) for mt_list in baseline_locs]
        
        self.nest_pct = []
        self.arena_pct = []

        for trials in baseline_locs:  # baseline_locs: list[list[trial_coords]]
            group_nest = []
            group_arena = []

            for trial in trials:
                # trial is a list/array of (x, y) with NaNs when in nest / missing
                trial_arr = np.asarray(trial, dtype=float)

                mask = escape.nest_mask_from_coords(trial_arr, self.exit_roi)

                nest_frames = mask.sum()
                total_frames = max(1, len(trial_arr))

                nest_percent = 100.0 * nest_frames / total_frames
                arena_percent = 100.0 - nest_percent

                group_nest.append(nest_percent)
                group_arena.append(arena_percent)

            self.nest_pct.append(group_nest)
            self.arena_pct.append(group_arena)
        
        zero_pct = []
        for i, j in zip(baseline_id, self.nest_pct):
            zero_nest_count = sum(1 for idx, val in zip(i, j) if val == 0.0)
            zero_nest_pct = (zero_nest_count / len(i)) * 100
            zero_pct.append([zero_nest_pct])
                
        files.create_csv(self.nest_pct, os.path.join(self.output_folder, 'time_in_nest.csv'), columns=self.mouse_types)
        files.create_csv(zero_pct, os.path.join(self.output_folder, 'failure.csv'), columns=self.mouse_types)
        
        # --- MSD ---
        cfg = {"baseline": dict(
                name="Baseline",
                trials=baseline_locs,
                maxlag=10,
                minlag=0.05,
                maxfit=1.0),
            "event": dict(
                name="Event",
                trials=event_locs,
                maxlag=2,
                minlag=0.02,
                maxfit=0.5), }

        msd_store  = {"baseline": [], "event": []}
        alpha_store = {"baseline": {}, "event": {}}
        
        alpha_trials_store = {
            "baseline": {mt: [] for mt in self.mouse_types},
            "event":    {mt: [] for mt in self.mouse_types},}
        
        #--- Plotting ---
        coords_args = dict(
            xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=10, 
            xmin=80, xmax=790, ymin=700, ymax=70, smooth=1, show=False, close=True)

        # Plot baseline and event heatmaps
        fig, axes = plt.subplots(len(self.mouse_types), 1, figsize=(7, 5 * len(self.mouse_types)))
        self.imgs.append(fig)
        for i in range(len(self.mouse_types)):
            plot.plot_coords(fig, axes[i], flat_baseline_locs[i], **coords_args)
            axes[i].set_title(f"{self.mouse_labels[i]}")
            x0 = coords_args["xmin"] + 20     # left padding
            y0 = coords_args["ymax"] + 25     # slightly below top (remember: y increases downward)

            # Choose length
            bar_cm = 10                        # length in cm (example)
            bar_px = bar_cm / self.conversion_factor   # convert to px if needed

            # Draw the line
            axes[i].plot([x0, x0 + bar_px], [y0, y0],
                    color="white", lw=3, solid_capstyle="butt", clip_on=False)

        #--- Plot time in nest vs arena bar graphs ---
        grouped_data = list(zip(self.nest_pct, self.arena_pct))
        fig, ax = plt.subplots(figsize=(1.5*len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)

        plot.plot_grouped_bar(fig, ax,
            grouped_data,
            xticks=self.mouse_labels,
            labels=("Nest", "Arena"),
            colors=colors,
            error_bars=True,
            ylim=(0,None),
            y_label="Time spent in nest (%)",
            bar_width=0.1,
            legend_loc='upper left')
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, self.nest_pct, "Mouse Type", "Time spent in nest (%)", self.mouse_labels, colors,
                    ylim=(0, None), bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        # --- Plot % of mice with 0 nest pct ---
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 4))
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax,
            zero_pct,    # one row of values
            "",
            "% mice with zero nest time",
            self.mouse_labels,
            self.colors,
            bar_width=0.2,
            ylim=(0, 100), stats=False)
        
        #--- Plot distance from wall timecourses ---
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_timecourse(
                fig, ax,
                speeds_by_type=dist_ev_norm,
                mouse_types=self.mouse_labels,
                color="tab:green",
                frame_time=30,
                title="",
                y_label="Distance (cm)",
                x_label="Normalised Time",
                shade=False,
                ylim=(0,1),
                show=False, close=True)
        
        #--- MSD Plotting ---
        for key, pars in cfg.items():
            per_type_res = []
            for mt, trials in zip(self.mouse_types, pars["trials"]):
                # --- group MSD (what you already had) ---
                res = coordinates.compute_group_msd(trials, self.fps, max_lag_s=pars["maxlag"])
                per_type_res.append(res)

                if res is None:
                    alpha_store[key][mt] = np.nan
                    alpha_trials_store[key][mt] = []
                else:
                    lags, mean, _ = res
                    # group-level alpha (single value per mouse_type)
                    alpha_store[key][mt] = coordinates.fit_msd_alpha(
                        lags, mean,
                        min_lag=pars["minlag"],
                        max_lag=pars["maxfit"]
                    )

                    # --- per-trial alphas ---
                    trial_alphas = []
                    for trial in trials:
                        # skip empty / all-NaN trials
                        trial = np.asarray(trial)
                        if trial.size == 0 or np.all(np.isnan(trial)):
                            continue

                        # compute MSD for this single trial
                        trial_res = coordinates.compute_group_msd(
                            [trial],           # list with one trial
                            self.fps,
                            max_lag_s=pars["maxlag"]
                        )
                        if trial_res is None:
                            continue

                        l_t, msd_t, _ = trial_res
                        # fit alpha on this trial
                        alpha_t = coordinates.fit_msd_alpha(
                            l_t, msd_t,
                            min_lag=pars["minlag"],
                            max_lag=pars["maxfit"]
                        )
                        trial_alphas.append(alpha_t)

                    alpha_trials_store[key][mt] = trial_alphas

            msd_store[key] = per_type_res

            # timecourse plot
            fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
            self.imgs.append(fig)
            for res, color, label in zip(per_type_res, colors, self.mouse_labels):
                if res is None:
                    continue
                lags, mean_msd, sem_msd = res
                ax.plot(lags, mean_msd, color=color, label=label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Lag (s)")
            ax.set_ylabel("MSD (pixels²)")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            alpha_ref = 1.0   # Brownian

            # pick a reference point to anchor the line (no reason it has to go through 1,1)
            # e.g. the first valid lag where msd is defined
            valid = lags > 0
            lags_valid = lags[valid]
            mean_msd_valid = mean_msd[valid]

            lag0 = lags_valid[0]
            msd0 = mean_msd_valid[0]

            # MSD_ref(lag0) = msd0  =>  A_ref = msd0 / lag0^alpha_ref
            A_ref = msd0 / (lag0**alpha_ref)

            # build the reference line over the full lag range
            x_ref = lags_valid
            y_ref = A_ref * (x_ref**alpha_ref)

            ax.plot(x_ref, y_ref, '--', color='red', linewidth=2, label=r'$\alpha = 1$ ref')

            ax.legend(fontsize='x-small',frameon=False)

        # --- α baseline vs event (per-trial input, barplot shows mean ± error) ---
        self.alpha_baseline = alpha_store["baseline"]
        self.alpha_event    = alpha_store["event"]

        self.alpha_trials_baseline = alpha_trials_store["baseline"]
        self.alpha_trials_event    = alpha_trials_store["event"]

        # For each mouse type, give grouped_bar the list of per-trial alphas
        grouped_alpha = [
            (
                self.alpha_trials_baseline[mt],  # all baseline α for this mouse type
                self.alpha_trials_event[mt],     # all event α for this mouse type
            )
            for mt in self.mouse_types
        ]

        fig, ax = plt.subplots(figsize=(1.5*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_grouped_bar(
            fig, ax,
            grouped_data=grouped_alpha,
            xticks=self.mouse_labels,
            labels=("Baseline", "Event"),
            colors=colors,
            bar_width=0.1,
            y_label="MSD exponent α",
            ylim=(0, None),
            error_bars=True,   # now uses trial-wise spread
            show=False,
            close=True,
        )
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Brownian (α=1)")
        ax.legend(fontsize='x-small', frameon=False)

    def plot_angle_data(self):
        colors = self.colors
        
        #---- Plot Polar plots for angles before, during, and before escape ----
        fig, axes = plt.subplots(len(self.mouse_types), 3, figsize=(20, 5 * len(self.mouse_types)),subplot_kw=dict(projection='polar'),squeeze=False)
        self.imgs.append(fig)

        column_titles = ['5 Seconds Before Stimulus', 'From Stimulus Onset', '2 Seconds Before Escape']

        for i, mouse_type in enumerate(self.mouse_types):
            # Extract all three segments
            data_before, data_middle, data_after = extract.extract_angle_segments(
                file=self.event_angles_file,
                data_start=5,
                data_end=None,
                escape_col=4,
                mouse_type=mouse_type)

            angle_segments = [data_before, data_middle, data_after]

            list_lens =[]
            for list in data_middle:
                len_list = len(list)
                list_lens.append(len_list)
            avg_len = np.nanmean(list_lens)
            
            for j, angles in enumerate(angle_segments):
                ax = axes[i][j]
                
                # Add column titles only to top row
                if i == 0:
                    ax.set_title(column_titles[j], fontsize=14, pad=20)
                # Add row labels on the left-most column
                if j == 0:
                    ax.annotate(mouse_type, xy=(-0.3, 0.5), 
                        xycoords='axes fraction', fontsize=16, 
                        ha='right', va='center')

                # Flatten angle data for histogram
                flat_angles = [angle for trial in angles for angle in trial if not np.isnan(angle)]

                kwargs = {'bins': 36, 'show': False, 'close': True, 'direction': 1, 'zero': "E"}
                plot.plot_polar_chart(fig, ax, flat_angles, **kwargs)
                if j == 1:
                    ax.text(0.4, 1.07, f"Average length {avg_len/30:.2f}s", color='black', ha='right', va='bottom', transform=ax.transAxes, fontsize=10)
        
        #---- Plot Average Timecourses -----     
        stretched_ba = []
        for mt in self.mouse_types:
            ba = extract.extract_data_col(self.global_angles_file, data_start=4, data_end=None, mouse_type=mt)
            ba = [[(0-(abs(val))) for val in trial] for trial in ba]
            ba = _angles.stretch_angle_trials(ba, target_length=3605)
            stretched_ba.append(ba)
               
        stretched_ta = []
        for mt in self.mouse_types:
            ta, _ = extract.extract_data_col(self.event_angles_file, data_start=155, escape=True,
                                            get_escape_index=True, escape_col=4, mouse_type=mt)
            ta = [[(0-(abs(val))) for val in trial] for trial in ta]
            ta = _angles.stretch_angle_trials(ta, target_length=450)
            stretched_ta.append(ta)
            
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_timecourse(
            fig, ax,
            speeds_by_type=stretched_ba,
            mouse_types=self.mouse_labels,
            color="blue",
            frame_time=30,
            title="Facing angles (mean) - baseline",
            y_label="Facing angle (degrees)",
            x_label="Normalised Time",
            ylim=(-180,0),
            smooth=20,
            shade=False,
            show=False, close=True)
                    
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_timecourse(
            fig, ax,
            speeds_by_type=stretched_ta,
            mouse_types=self.mouse_labels,
            color="blue",
            frame_time=450,
            title="",
            y_label="Facing angle (degrees)",
            x_label="Normalised Time",
            ylim=(-180,0),
            show=False, close=True)
        
        if len(self.mouse_types) > 3:
            tc_specs = [
                (stretched_ta[:2], self.mouse_labels[:2]),
                (stretched_ta[2:], self.mouse_labels[2:]),
                ([stretched_ta[i] for i in (0, 2)], [self.mouse_labels[i] for i in (0, 2)]),
                ([stretched_ta[i] for i in (1, 3)], [self.mouse_labels[i] for i in (1, 3)]),]
            
            for speeds_subset, labels_subset in tc_specs:
                fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
                self.imgs.append(fig)
                plot.plot_timecourse(
                    fig, ax,
                    speeds_by_type=speeds_subset,
                    mouse_types=labels_subset,
                    color="blue",
                    frame_time=450,
                    title="",
                    y_label="Facing angle (degrees)",
                    x_label="Normalised Time",
                    ylim=(-180, 0))

        
        #----Plot Histograms, Bar plots, Violin plots ----
        baseline_angles = [extract.extract_data_col(
                        self.global_angles_file, nested=False,
                        data_start=4, data_end=None,
                        escape=False, get_escape_index=False,
                        escape_col=None, mouse_type=mt)
                    for mt in self.mouse_types]
        event_angles = [extract.extract_data_col(
                        self.event_angles_file, nested=False,
                        data_start=155, data_end=None,
                        escape=True, get_escape_index=True,
                        escape_col=4, mouse_type=mt)[0]
                    for mt in self.mouse_types]
        event_angles_end = [extract.extract_data_col(
                        self.event_angles_file, nested=True,
                        data_start=155, data_end=None,
                        escape=True, get_escape_index=True,
                        escape_col=4, mouse_type=mt)[0]
                    for mt in self.mouse_types]
        event_angles_end = [[trial[-60:] for trial in mt_list] for mt_list in event_angles_end]
        event_angles_end = [sum(mt_list, []) for mt_list in event_angles_end]
        
        angle_groups = [event_angles, event_angles_end, baseline_angles]
        group_names = ['Event', 'Last 2s', 'Baseline']
        bins = 15
        range_degrees = (-180, 180)

        for group, name in zip(angle_groups, group_names):
            
            # Plot histogram
            group_percents = []
            for angles in group:
                counts, bin_edges = np.histogram(angles, bins=bins, range=range_degrees)
                group_percents.append(counts / counts.sum() * 100)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
            self.imgs.append(fig)
            for i, label in enumerate(self.mouse_labels):
                ax.plot(bin_centers, group_percents[i], label=label, color=colors[i])
            ax.set_xlabel("Angular deviation (degrees)")
            ax.set_ylabel("Percentage of samples (%)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.close()

            # Plot Bar plot
            for i in range(len(group)):
                group[i] = [(np.abs(trial)) for trial in group[i]]
            fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1.5, 4), layout='tight')
            self.imgs.append(fig)
            plot.plot_bar(fig, ax, group,
                x_label="Mouse Type", y_label="Mean angular deviation (degrees)",
                bar_labels=self.mouse_labels, colors=colors,
                ylim=None, bar_width=0.1, points=False,
                log_y=False, error_bars=True, show=False, close=True)
            
            # Plot violinplot
            flat_data = []
            for i, mouse_label in enumerate(self.mouse_labels):
                for angle in group[i]:
                    flat_data.append({'Mouse Type': mouse_label, 'Angle': angle})
                    
            fig, ax = plt.subplots(figsize=(len(self.mouse_types)*1, 4), layout='tight')
            self.imgs.append(fig)

            sns.violinplot(data=(pd.DataFrame(flat_data)), x='Mouse Type', y='Angle',
                palette=colors, hue='Mouse Type',
                ax=ax, cut=0, inner='box', legend=False)

            ax.set_xlabel("Mouse Type")
            ax.set_ylabel("Angular deviation (degrees)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.close()

    def plot_heading_consistency(self, max_lag_seconds=5):

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

            # MVL (all frames)
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
                ylim=(0, None),
                bar_width=0.1,
                log_y=False,
                error_bars=True
            )

            # Heading autocorrelation & half-life (all frames)
            lags = None
            C_by_type_all, half_lives_all = [], []
            for trials in angles_by_type:
                lags, C = _angles.heading_autocorr(trials, max_lag=max_lag)
                C_by_type_all.append(C)

            # Plot C(tau) – all frames
            fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
            self.imgs.append(fig)
            for i, C in enumerate(C_by_type_all):
                ax.plot(lags / fps, C, label=self.mouse_labels[i], color=colors[i])
            ax.set_xlabel('Lag (s)')
            ax.set_ylabel('Heading autocorrelation C(τ)')
            ax.set_title('Heading Autocorrelation (all frames)')
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.legend()
            plt.close()

    def plot_stats_data(self):
        colors = self.colors
        
        # --- Data extraction ---
        params = {
            'time': (self.escape_stats_file, 7, False),
            'distance': (self.escape_stats_file, 9, True),
            'escape_avg': (self.escape_success_file, 4, False),
            'prev_nest': (self.escape_stats_file, 8, True),
            'age': (self.escape_success_file, 3, False),
            'time_angle': (self.escape_stats_file, 10, False),
            'true_false_locs': (self.escape_stats_file, 5, True)}
        
        data = {k: [] for k in params}
        for mouse in self.mouse_types:
            for key, (file, row, esc) in params.items():
                data[key].append(extract.extract_data_rows(file, data_row=row, escape=esc, mouse_type=mouse))
        data['time'] = [[val for val in mouse_times if val < 15] for mouse_times in data['time']]

        #--- Plotting ---
        # --- Bar plots ---
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 4),  layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, data['escape_avg'], "", "% Escape Success",
                    self.mouse_labels, colors,
                    ylim=None, bar_width=0.2, points=True, error_bars=False, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 4)   , layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, data['time'], "", "Time (s)",
                    self.mouse_labels, colors,
                    ylim=None, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        # --- Age vs Escape (regression) ---
        fig, axes = plt.subplots(1, len(self.mouse_types), figsize=(5 * len(self.mouse_types), 5))
        self.imgs.append(fig)
        for ax, age, esc, color, label in zip(axes, data['age'], data['escape_avg'], colors, self.mouse_labels):
            sns.regplot(x=age, y=esc, ax=ax, scatter_kws={'color': color, 'alpha': 0.7, 's': 10},
                        line_kws={'color': color}, ci=None)
            ax.set(xlabel="Mouse Age", ylabel="Escape Success (%)", title=str(label))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.close()

        # --- Distance vs Time regression ---
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        for i, (dist, time, color, label) in enumerate(zip(data['distance'], data['time'], colors, self.mouse_labels)):
            plot.regression_plot(
                fig, ax, dist[0], time, label, color,
                x_label="Distance from Exit (cm)", y_label="Time to Escape (s)", title="",
                text_index=i, stats=self.stats, scatter=True, close=(i == len(self.mouse_labels) - 1))
        plt.tight_layout()
        plt.close()

        # --- True/False location scatter + heatmap ---
        true_locs, false_locs = zip(*[(loc[0], loc[1]) for loc in data['true_false_locs']])
        
        fig, axes = plt.subplots(1, len(self.mouse_types), figsize=(7 * len(self.mouse_types), 6), layout='tight')
        self.imgs.append(fig)
        for (i, label), ax in zip(enumerate(self.mouse_labels), axes):
            ax.set_title(f'{label}')
            ax.imshow(self.background_image, cmap="gray", extent=[0, 850, 755, 0], aspect='auto', zorder=0)
            plot.scatter_plot_with_stats(fig, ax, true_locs[i], point_color='tab:blue', mean_marker='o',
                                x_limits=(0, 850), y_limits=(755, 0), show=False, close=False)
            plot.scatter_plot_with_stats(fig, ax, false_locs[i], point_color='tab:red', mean_marker='o',
                                x_limits=(0, 850), y_limits=(755, 0), show=False, close=True)
            from matplotlib.lines import Line2D
            ax.legend(
                handles=[Line2D([0],[0], marker='o', color=c, linestyle='') for c in ['tab:blue','tab:red']],
                labels=['Escape','Non-Escape'],
                loc='upper right',
                frameon=False,
                fontsize='x-small')
            x_limits = (0, 850)
            y_limits = (755, 0)
            x0 = x_limits[0] + 20
            y0 = y_limits[1] + 25
            bar_cm = 10
            bar_px = bar_cm / self.conversion_factor
            axes[i].plot([x0, x0 + bar_px], [y0, y0],
                    color="white", lw=3, solid_capstyle="butt", clip_on=False)
        plt.tight_layout()
        plt.close()
        
        files.create_csv(data['escape_avg'], os.path.join(self.output_folder, 'esc_avg.csv'), columns=self.mouse_types)
        files.create_csv(data['time'], os.path.join(self.output_folder, 'time_to_esc.csv'), columns=self.mouse_types)
        files.create_csv(true_locs, os.path.join(self.output_folder, 'loc_at_stim.csv'), columns=self.mouse_types)
        
    def plot_traj_data(self):
        
        #--- Extraction ---
        true_locs = []
        for mouse_type in self.mouse_types:
            true_loc, _ = extract.extract_data_col(self.event_locs_file,
                data_start=155, escape=True, process_coords=True,
                get_escape_index=True, escape_col=4, mouse_type=mouse_type)
            true_locs.append(coordinates.stretch_trials(true_loc, target_length=450))

        #--- Plot trajectories ---
        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, axes = plt.subplots(len(self.mouse_types), 1, figsize=(13, 10 * len(self.mouse_types)), layout='tight')
        self.imgs.append(fig)
        for i, mouse_type in enumerate(self.mouse_labels):
            ax_escape = axes[i]
            ax_escape.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            ax_escape.set_title(f"{mouse_type}")
            plot.time_plot(fig, ax_escape, true_locs[i], fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
            
            x0 = x_limits[0] + 20
            y0 = y_limits[1] + 25
            bar_cm = 10
            bar_px = bar_cm / self.conversion_factor
            axes[i].plot([x0, x0 + bar_px], [y0, y0],
                    color="white", lw=3, solid_capstyle="butt", clip_on=False)

    def plot_tort_data(self):
        colors = self.colors
        all_tort = []
        all_dists = []

        # --- extract data ---
        for mtype in self.mouse_types:
            true, _ = extract.extract_data_col(self.event_locs_file, data_start=155,
                                            escape=True, process_coords=True, get_escape_index=True,
                                            escape_col=4, mouse_type=mtype)

            tort, dists = _tort.calc_tortuosity(true, self.min_path_length)
            all_tort.append(tort)
            all_dists.append(dists)
            
            self.true_total_dists = dists
        
        files.create_csv(all_tort, os.path.join(self.output_folder, 'tort.csv'), columns=self.mouse_types)

        # --- plot ---
        fig, ax = plt.subplots(figsize=(1.5 * len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plt.tight_layout()
        plot.plot_bar(fig, ax, all_tort,
            "Mouse type", "Tortuosity",
            bar_labels=self.mouse_labels, colors=colors,
            bar_width=0.1, ylim=(1,None), error_bars=True, log_y=False)

        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_histogram(fig, ax, data_list=all_tort, bins=80,
            labels=self.mouse_labels,colors=self.colors,xlabel="Tortuosity")
        ax.set_xlim(None)

    def plot_behaviour(self):
        
        colors = ["mediumblue", "cornflowerblue", "silver"]

        # --- extract data ---
        baseline_angles = [extract.extract_data_col(self.global_angles_file, data_start=4, data_end=3604, mouse_type=mt)
                        for mt in self.mouse_types]
        baseline_locs   = [extract.extract_data_col(self.global_locs_file, data_start=4, data_end=3604, process_coords=True, mouse_type=mt)
                        for mt in self.mouse_types]
        event_angles    = [extract.extract_data_col(self.event_angles_file, data_start=155, mouse_type=mt)
                        for mt in self.mouse_types]
        event_locs      = [extract.extract_data_col(self.event_locs_file, data_start=155, process_coords=True, mouse_type=mt)
                        for mt in self.mouse_types]

        true_angles, false_angles, true_locs, false_locs = [], [], [], []
        for mt in self.mouse_types:
            ta, fa = extract.extract_data_col(self.event_angles_file, data_start=155, escape=True, escape_col=4, mouse_type=mt)
            tl, fl = extract.extract_data_col(self.event_locs_file, data_start=155, escape=True, escape_col=4, process_coords=True, mouse_type=mt)
            true_angles.append(ta); false_angles.append(fa)
            true_locs.append(tl);   false_locs.append(fl)

        # --- compute means ---
        categories = {
            "Baseline (Global)": behaviour.compute_mean_behaviour(baseline_angles, baseline_locs),
            "Event (All)"      : behaviour.compute_mean_behaviour(event_angles,   event_locs),
            "Escape"           : behaviour.compute_mean_behaviour(true_angles,    true_locs),
            "No Escape"        : behaviour.compute_mean_behaviour(false_angles,   false_locs),}

        # --- plot ---
        for title, per_mouse in categories.items():
            fig, axes = plt.subplots(len(per_mouse), 1, figsize=(6, 4 * len(per_mouse)))
            self.imgs.append(fig)
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

            for i, ax in enumerate(axes):
                plot.plot_pie_chart(fig, ax,
                    data=(per_mouse[i] or {}),
                    title=f"{self.mouse_labels[i]}",
                    colors="darkslateblue",
                    autopct='%1.1f%%')

    def plot_speed_data(self):
        colors = self.colors

        # --- Extract baseline and event speed data ---
        baseline_speeds = [extract.extract_data_col(self.global_speeds_file, data_start=4, data_end=None,  mouse_type=mt)
                   for mt in self.mouse_types]
        event_speeds   = [extract.extract_data_col(self.event_speeds_file,  data_start=155, escape=True,
                                                    get_escape_index=True, escape_col=4, mouse_type=mt)[0]
                            for mt in self.mouse_types]
        baseline_speeds = [coordinates.stretch_trials(trials, target_length=25000) for trials in baseline_speeds]
        
        global_avg_speeds = _speed.get_avg_speeds(baseline_speeds)
        event_avg_speeds  = _speed.get_avg_speeds(event_speeds)
        global_max_speeds = _speed.get_max_speeds(baseline_speeds)
        event_max_speeds  = _speed.get_max_speeds(event_speeds)

        # --- True / false speeds ---
        tf_speeds = [extract.extract_data_col(self.event_speeds_file, data_start=5, data_end=605,
                                            escape=True, escape_col=4, mouse_type=mt)
                    for mt in self.mouse_types]
        true_speeds  = [x[0] for x in tf_speeds]

        tf_esc_time = [extract.extract_data_rows(self.escape_stats_file, data_row=7, escape=True, mouse_type=mt)
                for mt in self.mouse_types]
        true_esc_time  = [x[0] for x in tf_esc_time]

        # --- Stretch trials for timecourse plots ---
        tf_stretched = [extract.extract_data_col(self.event_speeds_file, data_start=155, escape=True,
                                                get_escape_index=True, escape_col=4, mouse_type=mt)
                        for mt in self.mouse_types]
        stretched_ts = [coordinates.stretch_trials(x[0], target_length=450) for x in tf_stretched]

        # --- Match baseline vs event by ID for regression ---
        event_all   = [extract.extract_data_col(self.event_speeds_file, data_start=155, escape=False,
                                                get_escape_index=True, escape_col=4, mouse_type=mt)
                    for mt in self.mouse_types]
        baseline_id = [extract.extract_data_rows(self.global_speeds_file, data_row=1,             mouse_type=mt)
                    for mt in self.mouse_types]
        event_id    = [extract.extract_data_rows(self.event_speeds_file,  data_row=1, escape=False, mouse_type=mt)
                    for mt in self.mouse_types]

        baseline_means = [_speed.nanmean_per_trial(trials) for trials in baseline_speeds]
        event_means    = [_speed.nanmean_per_trial(trials) for trials in event_all]

        baseline_maps = [_speed.to_id_map(ids, means, cast_int=True) for ids, means in zip(baseline_id, baseline_means)]
        event_maps    = [_speed.to_id_map(ids, means, cast_int=False) for ids, means in zip(event_id,    event_means)]

        matched_event_means = [_speed.match_means(bm, em) for bm, em in zip(baseline_maps, event_maps)]

        cleaned_baseline, cleaned_event = [], []
        for b_means, e_means in zip(baseline_means, matched_event_means):
            b, e = _speed.clean_pairs(b_means, e_means)
            cleaned_baseline.append(b)
            cleaned_event.append(e)
            
        files.create_csv(global_avg_speeds, os.path.join(self.output_folder, 'baseline_mean_speeds.csv'), self.mouse_types)
        files.create_csv(event_avg_speeds, os.path.join(self.output_folder, 'event_mean_speeds.csv'), self.mouse_types)
        files.create_csv(global_max_speeds, os.path.join(self.output_folder, 'baseline_max_speeds.csv'), self.mouse_types)
        files.create_csv(event_max_speeds, os.path.join(self.output_folder, 'event_max_speeds.csv'), self.mouse_types)
        files.create_trial_csv(event_speeds, event_id, self.mouse_types, os.path.join(self.output_folder, "speeds.csv"),)

        # --- Plot bar charts ---
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, global_avg_speeds, "", "Baseline Average Speed (cm/s)", self.mouse_labels,
                        colors, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, event_avg_speeds, "", "Events Average Speed (cm/s)", self.mouse_labels,
                        colors, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
            
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, global_max_speeds, "", "Baseline Max Speed (cm/s)", self.mouse_labels,
                        colors, bar_width=0.2, ylim=(0,None), points=False, error_bars=True, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, event_max_speeds, "", "Event Max Speed (cm/s)", self.mouse_labels,
                        colors, bar_width=0.2, ylim=(0,None), points=False, error_bars=True, show=False, close=True)

        # --- Plot raw heatmaps ---
        fig, axs = plt.subplots(len(self.mouse_types), 1, figsize=(6, 4 * len(self.mouse_types)), layout='tight')
        self.imgs.append(fig)
        for ax, data, sort, title, vmax in zip(axs, true_speeds, true_esc_time, self.mouse_labels, [50, 50, 30, 30]):
            plot.cmap_plot(fig, ax, data, sort_data=sort,
                    title=title, ylabel="Trial", xlabel="Time from stimulus onset (s)",
                    cbar_label="speed (cm/s)", vmin=0, vmax=vmax, show=False, close=True)

        #--- Plot speed timecourses ---
        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_timecourse(
            fig, ax,
            speeds_by_type=baseline_speeds,
            mouse_types=self.mouse_labels,
            color=self.colors,
            frame_time=25000,
            title="",
            y_label="Speed (cm/s)",
            x_label="Normalised Time",
            ylim=(0,40), legend_loc='upper right', smooth=100, shade=False,
            show=False, close=True)
        ax.set_xlim(0.02,None)
        
        if len(self.mouse_types) > 3:
        
            tc_sets = [
                (stretched_ts[:2],  self.mouse_labels[:2]),
                (stretched_ts[2:],  self.mouse_labels[2:]),
                ([stretched_ts[i] for i in (0, 2)], [self.mouse_labels[i] for i in (0, 2)]),
                ([stretched_ts[i] for i in (1, 3)], [self.mouse_labels[i] for i in (1, 3)]),]

            for speeds_subset, labels_subset in tc_sets:
                fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
                self.imgs.append(fig)
                plot.plot_timecourse(
                    fig, ax,
                    speeds_by_type=speeds_subset,
                    mouse_types=labels_subset,
                    color="red",
                    frame_time=450,
                    title="",
                    y_label="Speed (cm/s)",
                    ylim=(0, 50),
                    x_label="Normalised Time",
                    show=False,
                    close=True)

    def plot_arena_coverage_data(self):
        colors = self.colors
        
        baseline_locs = [extract.extract_data_col(self.global_locs_file, data_start=4, data_end=None, process_coords=True, mouse_type=mt)for mt in self.mouse_types]
        event_locs = [extract.extract_data_col(self.event_locs_file, data_start=155, escape=True, process_coords=True,
                                                                    get_escape_index=True, escape_col=4, mouse_type=mt)[0]for mt in self.mouse_types]

        baseline_coverage = [coordinates.calculate_arena_coverage(locs) for locs in baseline_locs]
        event_coverage = [coordinates.calculate_arena_coverage(locs) for locs in event_locs]
        
        # --- Baseline coverage, nest, escape correlations ---
        escape_success = [extract.extract_data_rows(self.escape_success_file, data_row=4, mouse_type=mt)for mt in self.mouse_types]
        baseline_ids = [extract.extract_data_rows(self.global_locs_file, data_row=0, mouse_type=mt)for mt in self.mouse_types]

        coverage_dicts = [extract.build_dicts(i, c) for i, c in zip(baseline_ids, baseline_coverage)]
        escape_dicts   = [extract.build_dicts(i, e) for i, e in zip(baseline_ids, escape_success)]
        nest_dicts     = [extract.build_dicts(i, n) for i, n in zip(baseline_ids, self.nest_pct)]
        
        files.create_csv(baseline_coverage, os.path.join(self.output_folder, 'baseline_coverage.csv'), columns=self.mouse_types)
        files.create_csv(event_coverage, os.path.join(self.output_folder, 'event_coverage.csv'), columns=self.mouse_types)

        # --- Arena Coverage bar Plots ---
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, baseline_coverage,
            "", "Arena Coverage (%)",
            bar_labels=self.mouse_labels, colors=colors, bar_width=0.2,
            error_bars=True, ylim=(0,100))
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax, event_coverage,
            "", "Arena Coverage (%)",
            bar_labels=self.mouse_labels,
            colors=colors, bar_width=0.2,
            error_bars=True)
        
        #--- Regression plots ---
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(self.mouse_labels):
            nest_vals = list(nest_dicts[i].values())
            escape_vals = list(escape_dicts[i].values())
            plot.regression_plot(fig, ax, nest_vals, escape_vals,label, colors[i],
                x_label="Time spent in nest at baseline (%)", y_label="Escape Success (%)", title="",
                text_index=i, stats=self.stats, scatter=True, close=(i == len(self.mouse_labels) - 1), legend_loc='lower right')
            ax.set_ylim(0, 105)
        self.imgs.append(fig)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(self.mouse_labels):
            coverage_vals = list(coverage_dicts[i].values())
            escape_vals = list(escape_dicts[i].values())
            plot.regression_plot(fig, ax, coverage_vals, escape_vals, label, colors[i],
                x_label="Arena Coverage (%)", y_label="Escape Success (%)", title="",
                scatter=True, text_index=i, stats=self.stats, close=(i == len(self.mouse_labels) - 1))
            ax.set_ylim(0, 105)
        self.imgs.append(fig)
            
    def plot_distance_from_wall(self, seconds_before_escape=2, threshold=3.0, min_fraction_near_wall=0.8):

        colors = self.colors
        n_pts = int(seconds_before_escape * self.fps)

        per_group = []   # collect everything in one pass for downstream plots
        percentages = [] # for the bar plot
        columns = []
        trials = []
        event_id = []

        # -------- extract + prepare per group --------
        for mt in self.mouse_types:
            # true positions -> distance to nearest edge (cm)
            true_locs, _ = extract.extract_data_col(self.event_locs_file, data_start=155, data_end=605,
                escape=True, get_escape_index=True, process_coords=True, escape_col=4, mouse_type=mt)
            wall_dists = [[calc.point_to_rect(p, self.corners, skip=None) * self.conversion_factor for p in trial]
                for trial in true_locs]
            event_ids    = extract.extract_data_rows(self.event_locs_file,  data_row=0, escape=False, mouse_type=mt)
            # take last N frames and stretch to same length
            last_n = [t[-n_pts:] for t in wall_dists]
            trials = coordinates.stretch_trials(last_n, target_length=n_pts)

            # Δ from first frame
            norm_trials = np.array([np.asarray(t, float) - np.asarray(t, float)[0] for t in trials]) if trials else np.empty((0, n_pts))
            arr_trials = [np.asarray(t, float) for t in trials]
            usage = [np.nanmean(t) for t in arr_trials]  # mean distance per trial
            flags = [(t[0] < threshold) and (np.mean(t < threshold) >= min_fraction_near_wall) for t in arr_trials]

            idx = sorted(range(len(trials)), key=lambda i: (not flags[i], usage[i]))  # sort by sustained first, then mean asc
            sorted_trials = [trials[i] for i in idx]
            sorted_usage  = [usage[i]  for i in idx]
            split_idx = next((k for k, i in enumerate(idx) if not flags[i]), None)

            percentage = (100 * sum(flags) / len(trials)) if trials else 0.0
            percentages.append(percentage)
            columns.append([percentage])
            event_id.append(event_id)

            per_group.append(dict(
                mouse_type=mt,
                trials=trials,
                norm_trials=norm_trials,
                usage=usage,
                flags=flags,
                percentage=percentage,
                sorted_trials=sorted_trials,
                sorted_usage=sorted_usage,
                split_idx=split_idx,
                event_ids=event_ids))
        
        files.create_csv(columns, os.path.join(self.output_folder, 'sustained_wall_usage.csv'), self.mouse_types)
        trials_by_type = [g["trials"]    for g in per_group]   # list[list[trial]]
        event_ids_by_type = [g["event_ids"] for g in per_group]  # list[list[event_id]]

        files.create_trial_csv(
            trials_by_type,
            event_ids_by_type,
            self.mouse_types,
            os.path.join(self.output_folder, "distance_from_wall.csv"),
        )

        # -------- Plot Δ distance panels per group --------
        for i, g in enumerate(per_group):
            if g["norm_trials"].size == 0:
                continue
            t = np.arange(g["norm_trials"].shape[1]) / self.fps

            fig, ax = plt.subplots(figsize=(6, 5))
            self.imgs.append(fig)

            for tr in g["norm_trials"]:
                ax.plot(t, tr, alpha=0.3, color=colors[i])
            ax.plot(t, np.nanmean(g["norm_trials"], axis=0), color='black', linewidth=2, label='Mean')

            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_title(f"Δ Distance from Wall - {self.mouse_labels[i]}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance Change (cm)")
            ax.legend()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            
        # -------- Plot sustained wall use bar chart --------
        fig, ax = plt.subplots(figsize=(1 * len(per_group), 4), layout='tight')
        self.imgs.append(fig)

        plot.plot_bar(fig, ax, data_groups=percentages,
            x_label="", y_label="% Trials with Sustained Wall Use",
            bar_labels=self.mouse_labels, colors=colors,
            ylim=(0,35), bar_width=0.1, error_bars=False,
            title="Sustained Wall Use", stats=False)

        # -------- Plot heatmap grid (sorted by sustained then mean) --------
        fig, axes = plt.subplots(2, len(per_group), figsize=(6 * len(per_group),7.5),
            gridspec_kw={'height_ratios': [1, 2]}, sharex='col')
        self.imgs.append(fig)

        for i, g in enumerate(per_group):
            ax_slice = axes[:, i]
            add_cbar = (i == len(per_group) - 1)

            plot.cmap_plot_with_average(
                fig, ax_slice,
                data1=g["sorted_trials"],
                sort_data1=g["sorted_usage"],
                title1=self.mouse_labels[i],
                length=seconds_before_escape * self.fps,
                ylabel="Distance from wall (cm)",
                ylim=None,
                cbar_label="Distance from wall (cm)",
                cmap="BuPu_r",
                fps=self.fps,
                vmin=0,
                vmax=7,
                add_cbar=add_cbar)

            if g["split_idx"] is not None:
                heatmap_ax = ax_slice[1]
                heatmap_ax.axhline(g["split_idx"] - 0.5, color='red',  linestyle='--', linewidth=3)

    def plot_path_similarity_and_area(self):

        # ---------------- extract + stretch + compute -----------------
        stretched_before_all, stretched_stim_all = [], []
        path_sim_all, path_sim_norm_all, path_area_all, path_area_norm_all = [], [], [], []
        path_length_all = []
        total_path_length_all = []

        for i, mt in enumerate(self.mouse_types):
            before, _ = extract.extract_data_col(
                self.prev_esc_locs_file, data_start=6, data_end=None,
                escape=True, process_coords=True, get_escape_index=True,
                escape_col=4, mouse_type=mt)
            stim, _ = extract.extract_data_col(
                self.event_locs_file, data_start=155,
                escape=True, process_coords=True, get_escape_index=True,
                escape_col=4, mouse_type=mt)
            before = [trial for trial in before]
            stim = [[p for p in trial if not np.isnan(p).any()] for trial in stim]

            sb = coordinates.stretch_trials(before, target_length=450)
            ss = coordinates.stretch_trials(stim,   target_length=450)
            stretched_before_all.append(sb)
            stretched_stim_all.append(ss)

            # metrics (computed once)
            similarity = [simil.compute_path_retracing_distance(b,s) for b, s in zip(sb, ss)]
            path_sim_all.append(similarity)
            areas_scaled = [simil.polygon_area_between_paths(b, s) * self.area_conversion for b, s in zip(sb, ss)]
            path_area_all.append(areas_scaled)
            # distances between first/last stim coords (per trial), scaled
            Pl = [np.linalg.norm(np.asarray(s[-1]) - np.asarray(s[0])) * (self.conversion_factor) for s in stim]
            path_length_all.append(Pl)
            #total distance travelled in path
            Tpl = [
                (sum(calc.calc_dist_between_points(s[i], s[i - 1]) for i in range(1, len(s)))
                * self.conversion_factor)
                for s in stim]
            total_path_length_all.append(Tpl)

            # normalized areas
            path_area_norm_all.append([a / L if (L and np.isfinite(L)) else np.nan
                                    for a, L in zip(areas_scaled, Tpl)])
            path_sim_norm_all.append([d / L if (L and np.isfinite(L)) else np.nan
                                    for d, L in zip(similarity, Tpl)])
        
        files.create_csv(path_sim_all, os.path.join(self.output_folder, 'dist_sim.csv'), columns=self.mouse_types)
        files.create_csv(path_sim_norm_all, os.path.join(self.output_folder, 'dist_sim_norm.csv'), columns=self.mouse_types)
        files.create_csv(path_area_all, os.path.join(self.output_folder, 'area_sim.csv'), columns=self.mouse_types)
        files.create_csv(path_area_norm_all, os.path.join(self.output_folder, 'area_sim_norm.csv'), columns=self.mouse_types)
        files.create_csv(path_length_all, os.path.join(self.output_folder, 'path_length.csv'), columns=self.mouse_types)
        files.create_csv(total_path_length_all, os.path.join(self.output_folder, 'total_path_length.csv'), columns=self.mouse_types)
        
        #--------Plotting---------
        """#optional: plot per-trial panels    
        # per-trial panels with closest lines + score
        for i, mt in enumerate(self.mouse_labels):
            plot.plot_trial_grid_paths(
                stretched_before_all[i],
                stretched_stim_all[i],
                title=f"Mouse Type: {mt}",
                scores=path_sim_all[i],
                show_lines=True,
                self_obj=self)
        
        # per-trial polygon area panels
        for i, mt in enumerate(self.mouse_labels):
            plot.plot_trial_grid_paths(
                stretched_before_all[i],
                stretched_stim_all[i],
                title=f"Mouse Type: {mt}",
                drawer=lambda before, stim, ax, area, score: simil.area_drawer(before, stim, ax, area, score),
                areas=path_area_all[i],
                scores=None, self_obj=self)"""
                
        # Path length vs Total path length — per mouse type
        fig, ax = plt.subplots(figsize=(6,4), layout="tight"); self.imgs.append(fig)
        for i, (Ls, Tls, lbl, col) in enumerate(zip(path_length_all, total_path_length_all, self.mouse_labels, self.colors)):
            plot.regression_plot(
                fig, ax, Ls, Tls, lbl, col,
                x_label="Distance to Nest (cm)",
                y_label="Total path length (cm)",
                title="",
                stats=self.stats, scatter=True, close=(i == len(self.mouse_labels) - 1))
        
        fig, ax = plt.subplots(figsize=(6,4), layout="tight"); self.imgs.append(fig)
        for i, (Ls, Tls, lbl, col) in enumerate(zip(path_sim_all, path_area_all, self.mouse_labels, self.colors)):
            plot.regression_plot(
                fig, ax, Ls, Tls, lbl, col,
                x_label="Retracing distance (cm)",
                y_label="Retracing Area (cm2)",
                title="",
                stats=self.stats, scatter=True, close=(i == len(self.mouse_labels) - 1))

        # plot bar, histogram, regression for each group
        for data_list, name, norm, area, skew, err in [
            (path_sim_all, "Path Retracing Distance", "", False, self.stats, True),
            (path_sim_norm_all, "Path Retracing Distance", " (norm)", False, self.stats, True),
            (path_area_all, "Path Retracing Area", "", True, self.stats, True),
            (path_area_norm_all, "Path Retracing Area", " (norm)", True, self.stats, True),]:
            
            fig, ax = plt.subplots(figsize=(max(3,len(self.mouse_types)),4), layout="tight"); self.imgs.append(fig)
            if area: fig.suptitle(f"{name}{norm}")
            plot.plot_bar(fig, ax, data_list, "",
                        f"{'Polygon Area Between Paths' if area else name}{norm}",
                        bar_labels=self.mouse_labels, colors=self.colors,
                        bar_width=0.2, ylim=(0,None), error_bars=err)
            
            fig, ax = plt.subplots(figsize=(6,4), layout="tight"); self.imgs.append(fig)
            plot.plot_histogram(fig, ax, data_list=data_list, labels=self.mouse_labels, colors=self.colors,
                                bins=30, xlabel=f"{'Polygon Area Between Paths' if area else name}{norm}",
                                ylabel="Percentage Frequency", show_median=True, xlim=None, alpha=0.55, print_skew=True)
            #ax.set_xlim(0,1.5)

            fig, ax = plt.subplots(figsize=(6,4), layout="tight"); self.imgs.append(fig)
            for i,(L,vals,c,lbl) in enumerate(zip(total_path_length_all,data_list,self.colors,self.mouse_labels)):
                plot.regression_plot(fig, ax, L, vals, lbl, c,
                                    x_label="Total path length (cm)",
                                    y_label=f"{name}{norm} (cm{'2' if area else ''})",
                                    title=f"Path Length vs {name}{norm}", text_index=i, legend_loc='upper left',
                                    stats=self.stats, scatter=False, close=(i == len(self.mouse_labels) - 1))

            q_list = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]


            fig, axs = plt.subplots(
                len(q_list), 2,
                figsize=(8, 2.5 * len(q_list)),
                layout="tight"
            )
            self.imgs.append(fig)

            for i, q in enumerate(q_list):

                # BEST q%
                y0_b, y1_b = simil.get_hist_percentage(data_list, q=q, tail="below")
                ratios_b = np.divide(y1_b, y0_b, out=np.full_like(y0_b, np.nan), where=y0_b > 0)

                plot.plot_bar(
                    fig, axs[i, 0], ratios_b.tolist(),
                    "", "Tail / All share (×)",
                    bar_labels=self.mouse_labels, colors=self.colors,
                    bar_width=0.1, ylim=(0, None), error_bars=False, stats=False
                )
                axs[i, 0].set_title(f"Best {q}%")


                # WORST q%
                y0_w, y1_w = simil.get_hist_percentage(data_list, q=q, tail="above")
                ratios_w = np.divide(y1_w, y0_w, out=np.full_like(y0_w, np.nan), where=y0_w > 0)

                plot.plot_bar(
                    fig, axs[i, 1], ratios_w.tolist(),
                    "", "Tail / All share (×)",
                    bar_labels=self.mouse_labels, colors=self.colors,
                    bar_width=0.1, ylim=(0, None), error_bars=False, stats=False
                )
                axs[i, 1].set_title(f"Worst {q}%")

            q=20
            
            fig, ax = plt.subplots(figsize=((1*len(self.mouse_labels)), 4),layout="tight")
            self.imgs.append(fig)
            y0_b, y1_b = simil.get_hist_percentage(data_list, q=q, tail="below")
            ratios_b = np.divide(y1_b, y0_b, out=np.full_like(y0_b, np.nan), where=y0_b > 0)

            plot.plot_bar(
                fig, ax, ratios_b.tolist(),
                "", "Tail / All share (×)",
                bar_labels=self.mouse_labels, colors=self.colors,
                bar_width=0.1, ylim=(0, None), error_bars=False, stats=False)

            # WORST q%
            fig, ax = plt.subplots(figsize=((1*len(self.mouse_labels)), 4),layout="tight")
            self.imgs.append(fig)
            y0_w, y1_w = simil.get_hist_percentage(data_list, q=q, tail="above")
            ratios_w = np.divide(y1_w, y0_w, out=np.full_like(y0_w, np.nan), where=y0_w > 0)
            plot.plot_bar(
                    fig, ax, ratios_w.tolist(),
                    "", "Tail / All share (×)",
                    bar_labels=self.mouse_labels, colors=self.colors,
                    bar_width=0.1, ylim=(0, None), error_bars=False, stats=False)
                    
        """# ---------- Baseline sliding-window (20 frames = 10 before + 10 after) ----------
        baseline_dist_all, baseline_area_all = [], []
        baseline_dist_norm_all, baseline_area_norm_all = [], []
        baseline_path_length_all =[]
        baseline_total_path_length_all = []

        for mt in self.mouse_types:
            baseline_trials = extract.extract_data_col(
                self.global_locs_file, nested=True, data_start=4, data_end=None,
                process_coords=True, escape_col=None, mouse_type=mt)
            d, a, dn, an, Pl, Tpl = simil.sliding_window_metrics(
                baseline_trials, half=10,
                area_conversion=self.area_conversion,
                conv_factor=getattr(self, "conversion_factor", 1.0))
            baseline_dist_all.append(d)
            baseline_area_all.append(a)
            baseline_dist_norm_all.append(dn)
            baseline_area_norm_all.append(an)
            baseline_path_length_all.append(Pl)
            baseline_total_path_length_all.append(Tpl)

        # Save CSVs (one column per mouse type)
        files.create_csv(baseline_dist_all,       os.path.join(self.output_folder, "baseline_dist_sim.csv"),       columns=self.mouse_types)
        files.create_csv(baseline_dist_norm_all,  os.path.join(self.output_folder, "baseline_dist_sim_norm.csv"),  columns=self.mouse_types)
        files.create_csv(baseline_area_all,       os.path.join(self.output_folder, "baseline_area_sim.csv"),       columns=self.mouse_types)
        files.create_csv(baseline_area_norm_all,  os.path.join(self.output_folder, "baseline_area_sim_norm.csv"),  columns=self.mouse_types)
        
        fig, ax = plt.subplots(); self.imgs.append(fig)
        for Ls, Tls, lbl, col in zip(baseline_path_length_all, baseline_total_path_length_all, self.mouse_labels, self.colors):
            plot.regression_plot(
                fig, ax, Ls, Tls, lbl, col,
                x_label="Path length (cm)",
                y_label="Total path length (cm)",
                title="",
                stats=self.stats, scatter=False, csv_path=self.folder, close=(i == len(self.mouse_labels) - 1))
            
        fig, ax = plt.subplots(); self.imgs.append(fig)
        for Ls, Tls, lbl, col in zip(baseline_dist_all, baseline_area_all, self.mouse_labels, self.colors):
            plot.regression_plot(
                fig, ax, Ls, Tls, lbl, col,
                x_label="Retracing Distance (cm)",
                y_label="Retracing Area (cm)",
                title="",
                stats=self.stats, scatter=True, csv_path=self.folder, close=(i == len(self.mouse_labels) - 1))
        
        fig, ax = plt.subplots(); self.imgs.append(fig)
        for Ls, Tls, lbl, col in zip(baseline_path_length_all, baseline_area_all, self.mouse_labels, self.colors):
            plot.regression_plot(
                fig, ax, Ls, Tls, lbl, col,
                x_label="Path length (cm)",
                y_label="Total path length (cm)",
                title="",
                stats=self.stats, scatter=False, csv_path=self.folder, close=(i == len(self.mouse_labels) - 1))
            
        # Plots: bars + hist for distance and area (raw + norm)
        for data_list, name, norm, area, skew, err in [
            (baseline_dist_all,      "Baseline Path Retracing Distance", "",        False, True, True),
            (baseline_dist_norm_all, "Baseline Path Retracing Distance", " (norm)", False, True, True),
            (baseline_area_all,      "Baseline Path Retracing Area",     "",        True,  True, True),
            (baseline_area_norm_all, "Baseline Path Retracing Area",     " (norm)", True,  True, True),
            ]:
            fig, ax = plt.subplots(figsize=(max(3,len(self.mouse_types)),3), layout="tight"); self.imgs.append(fig)
            if area: fig.suptitle(f"{name}{norm}")
            plot.plot_bar(fig, ax, data_list, "Mouse Type",
                        f"{'Polygon Area Between Paths' if area else name}{norm}",
                        bar_labels=self.mouse_labels, colors=self.colors,
                        bar_width=0.2, ylim=(0,None), error_bars=err)

            fig, ax = plt.subplots(figsize=(6,4), layout="tight"); self.imgs.append(fig)
            plot.plot_histogram(fig, ax, data_list=data_list, labels=self.mouse_labels, colors=self.colors,
                                bins=30, xlabel=f"{'Polygon Area Between Paths' if area else name}{norm}",
                                ylabel="Percentage Frequency", show_median=True, alpha=0.55, print_skew=skew)"""

    def save_figs(self):
        if self.save_pdf:
            if self.imgs:
                files.save_report(self.imgs, self.folder)
        if self.save_imgs:
            if self.imgs:
                files.save_images(self.imgs, self.folder)
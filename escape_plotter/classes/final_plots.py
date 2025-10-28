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
    def __init__(self, folder, settings, mouse_types, save_pdf=True, save_imgs=True):

        for k, v in settings["dimensions"].items():
            setattr(self, k, v)
            
        for k, v in settings["video"].items():
            setattr(self, k, v)
            
        self.mouse_types = mouse_types
        self.mouse_labels = [str(m) for m in mouse_types]
        self.folder = folder
        image_file = "images/arena.tif"
        self.background_image = mpimg.imread(image_file)
        
        self.colors = ['mediumslateblue', 'navy', 'silver', 'slategray']
        #self.colors = ['mediumslateblue', 'silver', 'cadetblue']

        self.imgs = []
        self.save_pdf = save_pdf
        self.save_imgs = save_imgs

        self.parent_folder = os.path.dirname(self.folder)
        
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
        baseline_locs = [extract.extract_data_col(self.global_locs_file, nested=True, data_start=4, data_end=3604, process_coords=True, escape_col=None, mouse_type=mt) for mt in self.mouse_types]
        event_locs = [extract.extract_data_col(self.event_locs_file, nested=False, data_start=155, data_end=None, process_coords=True, escape=True, get_escape_index=True, escape_col=4, mouse_type=mt) for mt in self.mouse_types]
        
        dist_ev = [extract.extract_data_col(self.event_distances_file, data_start=155, escape=True,
                                                get_escape_index=True, escape_col=4, mouse_type=mt)for mt in self.mouse_types]
        dist_ev = [coordinates.stretch_trials(x[0], target_length=450) for x in dist_ev]
        
        #flatten for coord heatmaps
        flat_baseline_locs = [sum(mt_list, []) for mt_list in baseline_locs]
        flat_event_locs = [sum(mt_list, []) for mt_list in event_locs]
        
        self.nest_counts = [[sum(np.isnan(p).any() for p in trial) / self.fps for trial in trials]
            for trials in baseline_locs]

        self.nest_pct = [
            [100.0 * sum(escape.is_nest_frame(p, self.exit_roi) for p in trial) / max(1, len(trial)) for trial in trials]
            for trials in baseline_locs]

        self.arena_pct = [
            [100.0 - nest for nest in nests_one_group]
            for nests_one_group in self.nest_pct]
        
        #--- Plotting ---
        coords_args = dict(
            xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=10, 
            xmin=80, xmax=790, ymin=700, ymax=70, smooth=1, show=False, close=True)

        # Plot baseline and event heatmaps
        fig, axes = plt.subplots(len(self.mouse_types), 2, figsize=(15, 5 * len(self.mouse_types)))
        self.imgs.append(fig)
        for i in range(len(self.mouse_types)):
            plot.plot_coords(fig, axes[i][0], flat_baseline_locs[i], **coords_args)
            axes[i][0].set_title(f"{self.mouse_types[i]} - baseline")

            plot.plot_coords(fig, axes[i][1], flat_event_locs[i], **coords_args)
            axes[i][1].set_title(f"{self.mouse_types[i]} - events")
            
        #--- Plot time in nest vs arena bar graphs ---
        grouped_data = list(zip(self.nest_pct, self.arena_pct))
        fig, ax = plt.subplots(figsize=(1.5*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)

        plot.plot_grouped_bar(fig, ax,
            grouped_data,
            xticks=self.mouse_types,
            labels=("Nest", "Arena"),
            colors=colors,
            error_bars=True,
            ylim=(0,100),
            y_label="Time (%)",
            bar_width=0.1)
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        
        plot.plot_bar(fig, ax, self.nest_pct, "Mouse Type", "Time (%)", self.mouse_labels, colors,
                    ylim=(0, None), bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        #--- Plot distance from wall timecourses ---
        fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
        self.imgs.append(fig)
        plot.plot_speed_timecourses(
            fig, ax,
            speeds_by_type=dist_ev,
            mouse_types=self.mouse_types,
            color="tab:green",
            frame_time=30,
            title="Distance from wall - baseline",
            y_label="Distance (cm)",
            x_label="Normalised Time",
            ylim=(0,50),
            show=False, close=True)
        
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
            ba = extract.extract_data_col(self.global_angles_file, data_start=4, data_end=3604, mouse_type=mt)
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
            
        fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
        self.imgs.append(fig)
        plot.plot_speed_timecourses(
            fig, ax,
            speeds_by_type=stretched_ba,
            mouse_types=self.mouse_types,
            colors=["cornflowerblue", "blue", "darkblue"],
            frame_time=30,
            title="Facing angles (mean) - baseline",
            y_label="Facing angle (degrees)",
            x_label="Normalised Time",
            ylim=(-180,0),
            show=False, close=True)
                    
        fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
        self.imgs.append(fig)
        plot.plot_speed_timecourses(
            fig, ax,
            speeds_by_type=stretched_ta,
            mouse_types=self.mouse_types,
            colors=["cornflowerblue", "blue", "darkblue"],
            frame_time=450,
            title="Escape facing angles (mean ± SD)",
            y_label="Facing angle (degrees)",
            x_label="Normalised Time",
            ylim=(-180,0),
            show=False, close=True)
        
        if len(self.mouse_types) > 3:
            fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
            self.imgs.append(fig)
            plot.plot_speed_timecourses(
                fig, ax,
                speeds_by_type=stretched_ta[:2],
                mouse_types=self.mouse_types[:2],
                colors=["cornflowerblue", "blue", "darkblue"],
                frame_time=450,
                title="Escape facing angles (mean ± SD)",
                y_label="Facing angle (degrees)",
                x_label="Normalised Time",
                ylim=(-180,0),
                show=False, close=True) 
            
            fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
            self.imgs.append(fig)
            plot.plot_speed_timecourses(fig, ax,
                speeds_by_type=stretched_ta[2:],
                mouse_types=self.mouse_types[2:],
                colors=["cornflowerblue", "blue"],
                frame_time=450,
                title="Escape facing angles (mean ± SD)",
                y_label="Facing angle (degrees)",
                ylim=(-180,0),
                x_label="Normalised Time",
                show=False, close=True) 
        
        #----Plot Histograms, Bar plots, Violin plots ----
        baseline_angles = [extract.extract_data_col(
                        self.global_angles_file, nested=False,
                        data_start=4, data_end=3604,
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

            fig, ax = plt.subplots(figsize=(5, 4), layout='tight')
            self.imgs.append(fig)
            for i, label in enumerate(self.mouse_labels):
                ax.plot(bin_centers, group_percents[i], label=label, color=colors[i])
            ax.set_xlabel("Angular deviation (degrees)")
            ax.set_ylabel("Percentage of samples (%)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
            plt.close()

            # Plot Bar plot
            fig, ax = plt.subplots(figsize=(len(self.mouse_types)*2, 4), layout='tight')
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

        baseline_locs_esc = [extract.extract_data_col(self.global_locs_file, nested=True, data_start=4, data_end=3604,
                                    process_coords=True, escape_col=None, get_escape_index=True, mouse_type=mt)
            for mt in self.mouse_types]
        
        times_success, fail_flags = coordinates.get_baseline_stats(baseline_locs_esc, fps=self.fps)

        # ---- Time to find nest -----------------------------
        fig1, ax1 = plt.subplots(figsize=(1*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig1)
        plot.plot_bar(fig=fig1, ax=ax1, data_groups=times_success,
            x_label="Mouse type", y_label="Time to find nest (s)",
            bar_labels=self.mouse_types, colors=colors,
            ylim=(0,None), bar_width=0.2,
            error_bars=True, stats=True, title=None)

        # ----  Failure to find nest ----------------------------------------
        fig2, ax2 = plt.subplots(figsize=(1*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig2)
        plot.plot_bar(fig=fig2, ax=ax2,data_groups=fail_flags,
            x_label="Mouse type",y_label="% Failure to find nest",
            bar_labels=self.mouse_types,colors=colors,
            ylim=(0, 100),bar_width=0.2, 
            error_bars=False, stats=False,title=None)

        #--- Plotting ---
        # --- Bar plots ---
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3),  layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, data['escape_avg'], "Mouse Type", "% Escape Success",
                    self.mouse_labels, colors,
                    ylim=None, bar_width=0.2, points=True, error_bars=False, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3)   , layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, data['time'], "Mouse Type", "Time (s)",
                    self.mouse_labels, colors,
                    ylim=None, bar_width=0.2, points=False, error_bars=True, show=False, close=True)

        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3),  layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, data['time_angle'], "Mouse Type", "Average Time to Face Exit (s)",
                    self.mouse_labels, colors,
                    ylim=None, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        # --- Age vs Escape (regression) ---
        fig, axes = plt.subplots(1, len(self.mouse_types), figsize=(5 * len(self.mouse_types), 5))
        self.imgs.append(fig)
        for ax, age, esc, color, label in zip(axes, data['age'], data['escape_avg'], colors, self.mouse_types):
            sns.regplot(x=age, y=esc, ax=ax, scatter_kws={'color': color, 'alpha': 0.7, 's': 10},
                        line_kws={'color': color}, ci=None)
            ax.set(xlabel="Mouse Age", ylabel="Escape Success (%)", title=str(label))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.close()

        # --- Distance vs Time regression ---
        fig, ax = plt.subplots()
        self.imgs.append(fig)

        for i, (dist, time, color, label) in enumerate(zip(data['distance'], data['time'], colors, self.mouse_labels)):
            plot.regression_plot(
                fig, ax,
                dist[0], time,
                label, color,
                x_label="Distance from Exit (cm)",
                y_label="Time to Escape (s)",
                title="Distance vs Time to Escape",
                text_index=i, stats=True, scatter=False, csv_path=self.folder)
        plt.tight_layout()
        plt.close()

        # --- True/False location scatter + heatmap ---
        true_locs, false_locs = zip(*[(loc[0], loc[1]) for loc in data['true_false_locs']])
        
        fig, axes = plt.subplots(2, len(self.mouse_types), figsize=(7 * len(self.mouse_types), 10))
        self.imgs.append(fig)
        for i, mouse in enumerate(self.mouse_types):
            for row, locs, label in [(0, true_locs[i], "escape"), (1, false_locs[i], "no escape")]:
                ax = axes[row, i]
                ax.set_title(f'{mouse} Mice - {label}')
                ax.imshow(self.background_image, cmap="gray", extent=[0, 850, 755, 0], aspect='auto', zorder=0)
                plot.scatter_plot_with_stats(fig, ax, locs, point_color='tab:blue', mean_marker='o',
                                x_limits=(0, 850), y_limits=(755, 0), show=False, close=True)
        plt.tight_layout()
        plt.close()

    def plot_traj_data(self):
        
        #--- Extraction ---
        true_locs = []
        false_locs = []
        
        for mouse_type in self.mouse_types:
            true_loc, false_loc = extract.extract_data_col(self.event_locs_file,
                data_start=155, escape=True, process_coords=True,
                get_escape_index=True, escape_col=4, mouse_type=mouse_type)
            true_locs.append(coordinates.stretch_trials(true_loc, target_length=450))
            false_locs.append(coordinates.stretch_trials(false_loc, target_length=450))
            print(f"Mouse type: {mouse_type}, Trials with escape: {len(true_loc)}, Trials without escape: {len(false_loc)}")

        #--- Plot trajectories ---
        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, axes = plt.subplots(len(self.mouse_types), 2, figsize=(14, 5 * len(self.mouse_types)))
        self.imgs.append(fig)
        fig.suptitle("Trajectories After Stimulus")

        for i, mouse_type in enumerate(self.mouse_types):
            ax_escape = axes[i][0]
            ax_escape.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            ax_escape.set_title(f"{mouse_type} - escape")
            plot.time_plot(fig, ax_escape, true_locs[i], fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)

            ax_no_escape = axes[i][1]
            ax_no_escape.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            ax_no_escape.set_title(f"{mouse_type} - no escape")
            plot.time_plot(fig, ax_no_escape, false_locs[i], fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True)

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

        self.true_tort = all_tort
        self.true_total_dists = all_dists

        # --- plot ---
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plt.tight_layout()

        plot.plot_bar(fig, ax, all_tort,
            "Mouse type", "Tortuosity (log)",
            bar_labels=self.mouse_types, colors=colors,
            bar_width=0.1, error_bars=True, log_y=True)

        fig, ax = plt.subplots(figsize=(6, 4), layout='tight')
        self.imgs.append(fig)
        plot.plot_histogram(fig, ax, data_list=all_tort,
            labels=self.mouse_types,colors=self.colors,xlabel="Tortuosity")

    def plot_behaviour(self):

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

        palette = ['#84206b', '#ae5097', '#ffd4f4', '#c6abeb']

        # --- plot ---
        for title, per_mouse in categories.items():
            fig, axes = plt.subplots(len(per_mouse), 1, figsize=(6, 4 * len(per_mouse)))
            self.imgs.append(fig)
            fig.suptitle(f"Behavior Proportions - {title}")
            axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]

            for i, ax in enumerate(axes):
                plot.plot_pie_chart(fig, ax,
                    data=(per_mouse[i] or {}),
                    title=f"{self.mouse_types[i]}",
                    colors=palette,
                    autopct='%1.1f%%')

    def plot_speed_data(self):
        colors = self.colors
        smooth = False  # Toggle for smoothing

        # --- Extract baseline and event speed data ---
        baseline_speeds = [extract.extract_data_col(self.global_speeds_file, data_start=4, data_end=3604,  mouse_type=mt)
                   for mt in self.mouse_types]
        event_speeds   = [extract.extract_data_col(self.event_speeds_file,  data_start=155, escape=True,
                                                    get_escape_index=True, escape_col=4, mouse_type=mt)[0]
                            for mt in self.mouse_types]
        
        global_avg_speeds = _speed.get_avg_speeds(baseline_speeds)
        event_avg_speeds  = _speed.get_avg_speeds(event_speeds)
        global_max_speeds = _speed.get_max_speeds(baseline_speeds)
        event_max_speeds  = _speed.get_max_speeds(event_speeds)

        # --- True / false speeds ---
        tf_speeds = [extract.extract_data_col(self.event_speeds_file, data_start=5, data_end=605,
                                            escape=True, escape_col=4, mouse_type=mt)
                    for mt in self.mouse_types]
        true_speeds  = [x[0] for x in tf_speeds]
        false_speeds = [x[1] for x in tf_speeds]

        tf_ages = [extract.extract_data_rows(self.escape_stats_file, data_row=7, escape=True, mouse_type=mt)
                for mt in self.mouse_types]
        true_ages  = [x[0] for x in tf_ages]
        false_ages = [x[1] for x in tf_ages]

        # --- Stretch trials for timecourse plots ---
        tf_stretched = [extract.extract_data_col(self.event_speeds_file, data_start=155, escape=True,
                                                get_escape_index=True, escape_col=4, mouse_type=mt)
                        for mt in self.mouse_types]
        stretched_ts = [coordinates.stretch_trials(x[0], target_length=450) for x in tf_stretched]
        stretched_fs = [coordinates.stretch_trials(x[1], target_length=450) for x in tf_stretched]

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

        # --- Plot bar charts ---
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, global_avg_speeds, "Mouse Type", "Baseline Average Speed (pps)", self.mouse_labels,
                        colors, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, event_avg_speeds, "Mouse Type", "Events Average Speed (pps)", self.mouse_labels,
                        colors, bar_width=0.2, points=False, error_bars=True, show=False, close=True)
            
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, global_max_speeds, "Mouse Type", "Baseline Max Speed (pps)", self.mouse_labels,
                        colors, bar_width=0.2, ylim=(0,None), points=False, error_bars=True, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(1 * len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, event_max_speeds, "Mouse Type", "Event Max Speed (pps)", self.mouse_labels,
                        colors, bar_width=0.2, ylim=(0,None), points=False, error_bars=True, show=False, close=True)

        # --- Plot raw heatmaps ---
        fig, axes = plt.subplots(2 * len(self.mouse_types), 2, figsize=(10, 5 * len(self.mouse_types)),
                                gridspec_kw={'height_ratios': [1, 2] * len(self.mouse_types)}, sharex='col')
        fig.suptitle("Speed Over Stimulus Event")

        for i, mt in enumerate(self.mouse_types):
            ax_slice = axes[2 * i:2 * i + 2, :]
            plot.cmap_plot(fig, ax_slice, true_speeds[i], false_speeds[i], true_ages[i], false_ages[i],
                            title1=f"{mt} - Escape", title2=f"{mt} - No Escape",
                            ylabel="Speed (cms)", ylim=(0, 10), vmin=0, vmax=30, cmap="viridis",
                            cbar_label="Speeds (cms)", fps=30, length=None,
                            cbar_dim=[0.92, 0.51 - i * 0.4, 0.015, 0.22], smooth=smooth, norm=False)
        self.imgs.append(fig)
        plt.close()

        # --- Plot normalised heatmaps ---
        fig, axes = plt.subplots(2 * len(self.mouse_types), 2, figsize=(10, 5 * len(self.mouse_types)),
                                gridspec_kw={'height_ratios': [1, 2] * len(self.mouse_types)}, sharex='col')
        for i, mt in enumerate(self.mouse_types):
            ax_slice = axes[2 * i:2 * i + 2, :]
            plot.cmap_plot(fig, ax_slice, stretched_ts[i], stretched_fs[i], true_ages[i], false_ages[i],
                                title1=f"{mt} - Escape", title2=f"{mt} - No Escape",
                                ylabel="Speed (cm/s)", ylim=(0, 20), vmin=0, vmax=50,
                                cmap="viridis", cbar_label="Speeds (cm/s)", fps=30, length=None,
                                cbar_dim=[0.92, 0.51 - i * 0.4, 0.015, 0.22], smooth=smooth, norm=True)
        self.imgs.append(fig)
        plt.close()

        #--- Plot regression plots ---
        fig, ax = plt.subplots()
        self.imgs.append(fig)

        for i, (baseline_data, event_data) in enumerate(zip(global_avg_speeds, matched_event_means)):
            color = colors[i]
            label = self.mouse_labels[i]
            plot.regression_plot(
                fig=fig, ax=ax, x=baseline_data, y=event_data, label=label, color=color,
                x_label="Speed at Baseline (cm/s)", y_label="Speed at Event (cm/s)", title="", text_index=i, stats=True,scatter=False, csv_path=self.folder,
                show=False, close=False)
        ax.legend(title="Mouse Type")
        
        #--- Plot speed timecourses ---
        fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
        self.imgs.append(fig)
        plot.plot_speed_timecourses(
            fig, ax,
            speeds_by_type=baseline_speeds,
            mouse_types=self.mouse_types,
            color="red",
            frame_time=30,
            title="Speed (mean) - baseline",
            y_label="Speed (cm/s)",
            x_label="Normalised Time",
            ylim=(0,20), smooth=20, shade=False,
            show=False, close=True)
        
        if len(self.mouse_types) <=3:
            fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
            self.imgs.append(fig)
            plot.plot_speed_timecourses(
                fig, ax,
                speeds_by_type=stretched_ts,
                mouse_types=self.mouse_types,
                color="red",
                frame_time=450,
                title="True-Escape Speeds (mean ± SD)",
                y_label="Speed (cm/s)",
                x_label="Normalised Time",
                show=False, close=True) 
        
        if len(self.mouse_types) > 3:
            fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
            self.imgs.append(fig)
            plot.plot_speed_timecourses(
                fig, ax,
                speeds_by_type=stretched_ts[:2],
                mouse_types=self.mouse_types[:2],
                color="red",
                frame_time=450,
                title="True-Escape Speeds (mean ± SD)",
                y_label="Speed (cm/s)",
                ylim=(0,50),
                x_label="Normalised Time",
                show=False, close=True)
            
            fig, ax = plt.subplots(figsize=(5, 3.5), layout='tight')
            self.imgs.append(fig)
            plot.plot_speed_timecourses(
                fig, ax,
                speeds_by_type=stretched_ts[2:],
                mouse_types=self.mouse_types[2:],
                color="red",
                frame_time=450,
                title="True-Escape Speeds (mean ± SD)",
                y_label="Speed (cm/s)",
                ylim=(0,50),
                x_label="Normalised Time",
                show=False, close=True)
        
    def plot_arena_coverage_data(self):
        colors = self.colors
        
        baseline_locs = [extract.extract_data_col(self.global_locs_file, data_start=4, data_end=3604, process_coords=True, mouse_type=mt)for mt in self.mouse_types]
        event_locs = [extract.extract_data_col(self.event_locs_file, data_start=155, escape=True, process_coords=True,
                                                                    get_escape_index=True, escape_col=4, mouse_type=mt)[0]for mt in self.mouse_types]

        baseline_coverage = [coordinates.calculate_arena_coverage(locs) for locs in baseline_locs]
        event_coverage = [coordinates.calculate_arena_coverage(locs) for locs in event_locs]
        
        # --- Baseline coverage, nest, escape correlations ---
        escape_success = [extract.extract_data_rows(self.escape_success_file, data_row=4, mouse_type=mt)for mt in self.mouse_types]
        baseline_ids = [extract.extract_data_rows(self.global_locs_file, data_row=0, mouse_type=mt)for mt in self.mouse_types]

        # --- Nest counts ---
        self.nest_counts = [[sum(np.isnan(p).any() for p in trial) / self.fps for trial in trials]
            for trials in baseline_locs]

        coverage_dicts = [extract.build_dicts(i, c) for i, c in zip(baseline_ids, baseline_coverage)]
        escape_dicts   = [extract.build_dicts(i, e) for i, e in zip(baseline_ids, escape_success)]
        nest_dicts     = [extract.build_dicts(i, n) for i, n in zip(baseline_ids, self.nest_counts)]

        # --- Arena Coverage bar Plots ---
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(fig, ax, baseline_coverage,
            "Mouse Type", "Arena Coverage (%)",
            bar_labels=self.mouse_types, colors=colors, bar_width=0.2,
            error_bars=True, ylim=(0,100))
        
        fig, ax = plt.subplots(figsize=(1*len(self.mouse_types), 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
            fig, ax, event_coverage,
            "Mouse Type", "Arena Coverage (%)",
            bar_labels=self.mouse_types,
            colors=colors, bar_width=0.2,
            error_bars=True)
        
        #--- Regression plots ---
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(self.mouse_labels):
            nest_vals = list(nest_dicts[i].values())
            escape_vals = list(escape_dicts[i].values())
            plot.regression_plot(fig, ax,
                nest_vals, escape_vals,label, colors[i],
                x_label="Time spent in nest (s)", y_label="Escape Success (%)",
                title="Nest Time vs Escape Success",
                text_index=i, stats=True, scatter=False, csv_path=self.folder)
            ax.set_ylim(0, 105)
        self.imgs.append(fig)
            
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, label in enumerate(self.mouse_labels):
            coverage_vals = list(coverage_dicts[i].values())
            escape_vals = list(escape_dicts[i].values())
            plot.regression_plot(fig, ax,
                coverage_vals, escape_vals, label, colors[i],
                x_label="Arena Coverage (%)", y_label="Escape Success (%)",
                title="Coverage vs Escape Success", text_index=i, stats=True)
        self.imgs.append(fig)
            
    def plot_distance_from_wall(self, seconds_before_escape=2, threshold=3.0, min_fraction_near_wall=0.8):

        colors = self.colors
        n_pts = int(seconds_before_escape * self.fps)

        per_group = []   # collect everything in one pass for downstream plots
        percentages = [] # for the bar plot

        # -------- extract + prepare per group --------
        for mt in self.mouse_types:
            # true positions -> distance to nearest edge (cm)
            true_locs, _ = extract.extract_data_col(self.event_locs_file, data_start=155, data_end=605,
                escape=True, get_escape_index=True, process_coords=True, escape_col=4, mouse_type=mt)
            wall_dists = [[calc.point_to_rect(p, self.corners, skip=None) * self.conversion_factor for p in trial]
                for trial in true_locs]

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

            per_group.append(dict(
                mouse_type=mt,
                trials=trials,
                norm_trials=norm_trials,
                usage=usage,
                flags=flags,
                percentage=percentage,
                sorted_trials=sorted_trials,
                sorted_usage=sorted_usage,
                split_idx=split_idx))

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

            print(f"{self.mouse_labels[i]} - Sustained Wall Use: {g['percentage']:.2f}% ({int(sum(g['flags']))}/{len(g['flags'])})")

        # -------- Plot sustained wall use bar chart --------
        fig, ax = plt.subplots(figsize=(1 * len(per_group), 3), layout='tight')
        self.imgs.append(fig)

        plot.plot_bar(fig, ax, data_groups=percentages,
            x_label="Mouse Type", y_label="% Trials",
            bar_labels=self.mouse_labels, colors=colors,
            ylim=None, bar_width=0.1, error_bars=False,
            title="Sustained Wall Use", stats=False)

        # -------- Plot heatmap grid (sorted by sustained then mean) --------
        fig, axes = plt.subplots(2, len(per_group), figsize=(5 * len(per_group), 7),
            gridspec_kw={'height_ratios': [1, 2]}, sharex='col')
        self.imgs.append(fig)

        for i, g in enumerate(per_group):
            ax_slice = axes[:, i]
            plot.cmap_plot(fig, ax_slice, data1=g["sorted_trials"], sort_data1=g["sorted_usage"],
                title1=f"{g['mouse_type']}", length=seconds_before_escape * self.fps,
                ylabel="Distance (cm)", ylim=None,
                cbar_label="Distance (cm)", cmap="BuPu_r",
                fps=self.fps, vmin=0, vmax=7,
                cbar_dim=[0.92, 0.51 - i * 0.4, 0.015, 0.22])
            if g["split_idx"] is not None:
                heatmap_ax = ax_slice[1]
                heatmap_ax.axhline(g["split_idx"] - 0.5, color='red', linestyle='--', linewidth=3)

    def plot_path_similarity_and_area(self):

        colors, labels = self.colors, self.mouse_labels

        # ---------------- extract + stretch + compute -----------------
        stretched_before_all, stretched_stim_all = [], []
        path_sim_all, path_area_all, path_area_norm_all = [], [], []

        for i, mt in enumerate(self.mouse_types):
            before, _ = extract.extract_data_col(
                self.prev_esc_locs_file, data_start=6, data_end=None,
                escape=True, process_coords=True, get_escape_index=True,
                escape_col=4, mouse_type=mt)
            stim, _ = extract.extract_data_col(
                self.event_locs_file, data_start=155,
                escape=True, process_coords=True, get_escape_index=True,
                escape_col=4, mouse_type=mt)

            sb = coordinates.stretch_trials(before, target_length=450)
            ss = coordinates.stretch_trials(stim,   target_length=450)
            stretched_before_all.append(sb)
            stretched_stim_all.append(ss)

            # metrics (computed once)
            path_sim_all.append([simil.compute_path_retracing_distance(b, s) for b, s in zip(sb, ss)])
            areas_scaled = [simil.polygon_area_between_paths(b, s) * self.area_conversion for b, s in zip(sb, ss)]
            path_area_all.append(areas_scaled)
            Ls = self.true_total_dists[i]
            path_area_norm_all.append([a / L if L else np.nan for a, L in zip(areas_scaled, Ls)])

        # ---------------- Plotting ----------------
        fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
        self.imgs.append(fig)
        plot.plot_histogram(fig, ax,
                data_list=path_sim_all,
                labels=labels,
                colors=colors,
                bins=30,
                xlabel="Path Retracing Distance (cm)",
                ylabel="Percentage Frequency",
                show_median=True,
                alpha=0.55)

        # regression: path length vs similarity
        fig, ax = plt.subplots()
        self.imgs.append(fig)
        for i, (path_len, ps, color, label) in enumerate(zip(self.true_total_dists, path_sim_all, colors, labels)):
            plot.regression_plot(fig, ax,
                    path_len, ps,
                    label, color,
                    x_label="Path length (cm)",
                    y_label="Path Retracing Distance (cm)",
                    title="Path Length vs Path Retracing Distance",
                    text_index=i, stats=True, scatter=False, csv_path=self.folder)
        
        # bars
        fig, ax = plt.subplots(figsize=(len(self.mouse_types) * 1, 3), layout='tight')
        self.imgs.append(fig)
        plot.plot_bar(
                fig, ax, path_sim_all,
                "Mouse Type", "Path Retracing Distance",
                bar_labels=labels, colors=colors, bar_width=0.2,
                ylim=(0, None))

        # per-trial panels with closest lines + score
        for i, mt in enumerate(self.mouse_types):
            plot.plot_trial_grid_paths(
                stretched_before_all[i],
                stretched_stim_all[i],
                title=f"Mouse Type: {mt}",
                scores=path_sim_all[i],
                show_lines=True,
                self_obj=self)

        # ---------------- area visuals ----------------
        # histogram (raw)
        fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
        self.imgs.append(fig)
        plot.plot_histogram(fig, ax, data_list=path_area_all,
                labels=labels, colors=colors, bins=30,
                xlabel="Polygon Area Between Paths",
                ylabel="Percentage Frequency",
                show_median=True, alpha=0.55)

        # bars (raw)
        fig, ax = plt.subplots(figsize=(len(self.mouse_types), 3), layout="tight")
        self.imgs.append(fig)
        fig.suptitle("Path Retracing Area")
        plot.plot_bar(fig, ax, path_area_all,
                "Mouse Type", "Polygon Area Between Paths",
                bar_labels=labels, colors=colors,
                bar_width=0.2, ylim=(0, None))

        # histogram (normalized)
        fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
        self.imgs.append(fig)
        plot.plot_histogram(fig, ax, data_list=path_area_norm_all,
                labels=labels, colors=colors, bins=30,
                xlabel="Polygon Area Between Paths (normalized)",
                ylabel="Percentage Frequency",
                show_median=True, alpha=0.55)

        # bars (normalized)
        fig, ax = plt.subplots(figsize=(len(self.mouse_types), 3), layout="tight")
        self.imgs.append(fig)
        fig.suptitle("Path Retracing Area (norm)")
        plot.plot_bar(fig, ax, path_area_norm_all,
                "Mouse Type", "Polygon Area Between Paths",
                bar_labels=labels, colors=colors,
                bar_width=0.2, ylim=(0, None))

        # per-trial polygon area panels
        for i, mt in enumerate(self.mouse_types):
            plot.plot_trial_grid_paths(
                stretched_before_all[i],
                stretched_stim_all[i],
                title=f"Mouse Type: {mt}",
                drawer=lambda before, stim, ax, area, score: simil.area_drawer(before, stim, ax, area, score),
                areas=path_area_all[i],
                scores=None, self_obj=self)

    def save_figs(self):
        if self.save_pdf:
            if self.imgs:
                files.save_report(self.imgs, self.folder)
        if self.save_imgs:
            if self.imgs:
                files.save_images(self.imgs, self.folder)
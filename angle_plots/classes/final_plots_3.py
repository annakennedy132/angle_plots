import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.interpolate import interp1d

from angle_plots.processing import coordinates, extract, plots, stats, calc, behaviour
from angle_plots.utils import files

import seaborn as sns
from scipy.spatial import cKDTree

class FinalPlots3:
    def __init__(self, folder, settings, mouse_type_1, mouse_type_2, mouse_type_3, save_figs=True, save_imgs=True):
        
        for k, v in settings["video"].items():
            setattr(self, k, v)
            
        for k, v in settings["tracking"].items():
            setattr(self, k, v)
            
        self.mouse_type_1 = mouse_type_1
        self.mouse_type_2 = mouse_type_2
        self.mouse_type_3 = mouse_type_3

        self.folder = folder
        image_file = "images/arena.tif"
        self.background_image = mpimg.imread(image_file)

        self.figs = []
        self.imgs = []
        self.save_figs = save_figs
        self.save_imgs = save_imgs

        self.parent_folder = os.path.dirname(self.folder)
        
        self.global_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_locs.csv")), None)
        self.global_speeds_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_speeds.csv")), None)
        self.global_distances_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_distances.csv")), None)
        self.after_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("after_angles.csv")), None)
        self.during_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("during_angles.csv")), None)
        self.prev_esc_locs_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("collated_prev_esc_locs.csv")), None)
        self.event_locs_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("locs.csv")), None)
        self.event_distances_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("event_distances.csv")), None)
        self.event_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("event_angles.csv")), None)
        self.event_speeds_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("event_speeds.csv")), None)
        self.stim_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("stim.csv")), None)
        self.escape_stats_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("escape-stats.csv")), None)
        self.escape_success_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("collated_escape_success.csv")), None)

    def plot_coord_data(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        
        baseline_locs = [
            extract.extract_data_col_bymousetype(
                self.global_locs_file, nested=False, data_start=3, data_end=5400, 
                process_coords=True, escape_col=None, mouse_type=mt
            ) for mt in mouse_types
        ]
        
        event_locs = [
            extract.extract_data_col_bymousetype(
                self.event_locs_file, nested=False, data_start=154, escape=False, 
                process_coords=True, escape_col=None, mouse_type=mt
            ) for mt in mouse_types
        ]

        coords_args = dict(
            xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=10, 
            xmin=80, xmax=790, ymin=700, ymax=80, show=False, close=True
        )

        # Plot baseline and event side-by-side
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        self.imgs.append(fig)
        for i in range(3):
            plots.plot_coords(fig, axes[i][0], baseline_locs[i], **coords_args)
            axes[i][0].set_title(f"{mouse_types[i]} - baseline")

            plots.plot_coords(fig, axes[i][1], event_locs[i], **coords_args)
            axes[i][1].set_title(f"{mouse_types[i]} - events")

        # Plot baseline heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        self.imgs.append(fig)
        fig.suptitle("Heatmap of Coordinates at Baseline")
        for i, ax in enumerate(axes):
            plots.plot_coords(fig, ax, baseline_locs[i], colorbar=(i == 2), **coords_args)
            ax.set_title(f"{mouse_types[i]} - baseline")

        # Plot event heatmaps
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        self.imgs.append(fig)
        fig.suptitle("Heatmap of Coordinates During Events")
        for i, ax in enumerate(axes):
            plots.plot_coords(fig, ax, event_locs[i], colorbar=(i == 2), **coords_args)
            ax.set_title(f"{mouse_types[i]} - events")

    def plot_angle_data(self):
        fig, axes = plt.subplots(3, 4, figsize=(20, 15), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)

        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        angle_sources = [
            {
                'title': 'Baseline - first 3 minutes',
                'file': self.global_angles_file,
                'data_start': 3,
                'data_end': 5400,
            },
            {
                'title': 'Before Stimulus',
                'file': self.event_angles_file,
                'data_start': 4,
                'data_end': 154,
            },
            {
                'title': 'During Stimulus / Time to Escape',
                'file': self.during_angles_file,
                'data_start': 4,
                'data_end': None,
            },
            {
                'title': 'After Stimulus / Exit from Nest',
                'file': self.after_angles_file,
                'data_start': 4,
                'data_end': None,
            }
        ]

        for i, mouse_type in enumerate(mouse_types):
            for j, source in enumerate(angle_sources):
                ax = axes[i][j]
                ax.set_title(source['title'])

                self.angles = extract.extract_data_col_bymousetype(
                    source['file'],
                    nested=False,
                    data_start=source['data_start'],
                    data_end=source['data_end'],
                    escape_col=None,
                    mouse_type=mouse_type
                )

                # Apply optional direction/zero only for specific subplots if needed
                kwargs = {'bins': 36, 'show': False, 'close': True}
                if j == 0 and i > 0:  # Baseline plots for mouse_type 2 and 3
                    kwargs['direction'] = 1
                    kwargs['zero'] = "E"

                plots.plot_polar_chart(fig, ax, self.angles, **kwargs)

    def plot_avgs_data(self):
        frame_time = 1. / self.fps
        norm_event_time = np.arange(-self.event["t_minus"], self.event["length"] + self.event["t_plus"], frame_time)

        # Extract data
        true_angles = []
        false_angles = []
        true_dist = []
        false_dist = []
        true_speeds = []
        false_speeds = []
        
        for mtype in [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]:
            t_angle, f_angle = extract.extract_data_col_bymousetype(self.event_angles_file, data_start=4, escape=True, escape_col=3, mouse_type=mtype)
            t_dist, f_dist = extract.extract_data_col_bymousetype(self.event_distances_file, data_start=4, escape=True, escape_col=3, mouse_type=mtype)
            t_speed, f_speed = extract.extract_data_col_bymousetype(self.event_speeds_file, data_start=4, escape=True, escape_col=3, mouse_type=mtype)
            
            true_angles.append(t_angle)
            false_angles.append(f_angle)
            true_dist.append(t_dist)
            false_dist.append(f_dist)
            true_speeds.append(t_speed)
            false_speeds.append(f_speed)

        def compute_avg_trace(data_list):
            return [
                np.nanmean([lst[i] if i < len(lst) else np.nan for lst in data_list])
                for i in range(max(map(len, data_list)))
            ] if data_list else []

        def compute_avg_angle_trace(data_list):
            return [
                0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in data_list])
                for i in range(max(map(len, data_list)))
            ] if data_list else []

        # Compute averages: [mouse1_escape, mouse1_no_escape, mouse2_..., mouse3_...]
        avg_esc_angle_data = [compute_avg_angle_trace(t) for t in true_angles] + [compute_avg_angle_trace(f) for f in false_angles]
        avg_esc_dist_data  = [compute_avg_trace(t) for t in true_dist] + [compute_avg_trace(f) for f in false_dist]
        avg_esc_speeds_data = [compute_avg_trace(t) for t in true_speeds] + [compute_avg_trace(f) for f in false_speeds]

        titles = ["Facing angle", "Distance from Exit (cm)", "Speed"]
        colours = ["blue", "green", "red"]
        data_limits = [(-185, 20), (0, 55), (0, 300)]
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]

        for df, title, colour, limits in zip([avg_esc_angle_data, avg_esc_dist_data, avg_esc_speeds_data], titles, colours, data_limits):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(f"Average {title} at Stim Event")
            self.figs.append(fig)
            self.imgs.append(fig)

            for i, ax in enumerate(axes):
                ax.set_title(mouse_types[i])
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel(title)
                ax.set_ylim(limits)
                ax.axvspan(0, 10, alpha=0.3, color='lightgray')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)

                ax.plot(norm_event_time, df[i], color=colour, label="escape")
                ax.plot(norm_event_time, df[i+3], color=colour, alpha=0.5, label="no escape")
                ax.legend()
            
    def plot_stats_data(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        
        # Data extraction keys: tuple of (file, data_row, optional escape flag)
        extraction_params = {
            'time':         (self.escape_stats_file, 6, False),
            'distance':     (self.escape_stats_file, 8, False),
            'escape_avg':   (self.escape_success_file, 3, False),
            'prev_nest':    (self.escape_stats_file, 7, True),
            'age':          (self.escape_success_file, 2, False),
            'time_angle':   (self.escape_stats_file, 9, False),
            'true_false_locs': (self.escape_stats_file, 4, True)
        }

        # Extract data for each mouse type and variable
        data = {key: [] for key in extraction_params}
        for mouse in mouse_types:
            for key, (file, row, escape) in extraction_params.items():
                if escape:
                    vals = extract.extract_data_rows_bymousetype(file, data_row=row, escape=True, mouse_type=mouse)
                else:
                    vals = extract.extract_data_rows_bymousetype(file, data_row=row, mouse_type=mouse)
                data[key].append(vals)
        
        # Convert times to numpy arrays and filter out times >= 15s
        for i, time_arr in enumerate(data['time']):
            arr = np.array(time_arr)
            arr[arr >= 15] = np.nan
            data['time'][i] = arr
        
        # -------- Plot 1: Escape probability, Time to escape, Time to face exit --------
        # -------- Plot 1: Escape probability, Time to escape, Time to face exit --------
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 4))
        self.imgs.append(fig)

        mouse_colors = ['teal', 'gainsboro', 'salmon']
        mouse_labels = [str(m) for m in mouse_types]

        plots.plot_bar(fig, axes[0], data['escape_avg'], "Mouse Type", "Escape probability (%)",
                    mouse_labels, mouse_colors,
                    ylim=(0, 102), bar_width=0.2, points=True, error_bars=False, show=False, close=True)

        plots.plot_bar(fig, axes[1], data['time'], "Mouse Type", "Time to Escape (s)",
                    mouse_labels, mouse_colors,
                    ylim=None, bar_width=0.2, points=False, error_bars=True, show=False, close=True)

        plots.plot_bar(fig, axes[2], data['time_angle'], "Mouse Type", "Average Time to Face Exit (s)",
                    mouse_labels, mouse_colors,
                    ylim=(0, 8), bar_width=0.2, points=False, error_bars=True, show=False, close=True)

        # Create an overall legend on the figure level
        from matplotlib.patches import Patch

        legend_handles = [Patch(color=color, label=label) for color, label in zip(mouse_colors, mouse_labels)]
        fig.legend(handles=legend_handles, loc='upper center', ncol=3, fontsize='medium', frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for the legend

        # -------- Plot 2: Age vs Escape Success with regression --------
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        self.imgs.append(fig)
        colors = ['teal', 'gainsboro', 'salmon']

        for ax, mouse, age, esc, color in zip(axes, mouse_types, data['age'], data['escape_avg'], colors):
            ax.set_xlabel("Mouse Age")
            ax.set_ylabel("Escape Success (%)")
            ax.set_title(str(mouse))
            sns.regplot(x=age, y=esc, ax=ax, scatter=True,
                        scatter_kws={'color': color, 'alpha': 0.7, 's': 10},
                        line_kws={'color': color}, ci=None)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # -------- Plot 3: Distance from exit vs Time to escape --------
        # -------- Plot 3: Distance from exit vs Time to escape --------
        fig, ax = plt.subplots()
        self.imgs.append(fig)

        ax.set_xlabel("Distance from Exit at Stimulus (cm)")
        ax.set_ylabel("Time to Escape (s)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        mouse_colors = ['teal', 'gainsboro', 'salmon']
        mouse_labels = [str(m) for m in mouse_types]

        # Plot regression for each mouse type
        for dist_data, time_data, color, label in zip(data['distance'], data['time'], mouse_colors, mouse_labels):
            sns.regplot(x=dist_data, y=time_data, ax=ax, scatter=True,
                        scatter_kws={'color': color, 'alpha': 0.7, 's': 10},
                        line_kws={'color': color}, label=label)

        # Add legend for mouse types
        ax.legend(title="Mouse Type")

        plt.tight_layout()

        # -------- Plot 4: True/False locations on background --------
        fig, axes = plt.subplots(2, 3, figsize=(21, 10))
        self.imgs.append(fig)
        x_limits = (0, 850)
        y_limits = (755, 0)

        true_locs = [locs[0] for locs in data['true_false_locs']]
        false_locs = [locs[1] for locs in data['true_false_locs']]

        for i, mouse in enumerate(mouse_types):
            # Escape plots
            axes[0, i].set_title(f'{mouse} Mice - escape')
            axes[0, i].imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            plots.scatter_plot_with_stats(fig, axes[0, i], true_locs[i], point_color='tab:blue', mean_marker='o',
                                        x_limits=x_limits, y_limits=y_limits, show=False, close=True)

            # No escape plots
            axes[1, i].set_title(f'{mouse} Mice - no escape')
            axes[1, i].imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            plots.scatter_plot_with_stats(fig, axes[1, i], false_locs[i], point_color='tab:blue', mean_marker='o',
                                        x_limits=x_limits, y_limits=y_limits, show=False, close=True)

        plt.tight_layout()

        # -------- Plot 5: Time Since Last Visiting Nest (Grouped Bar) --------
        fig, ax = plt.subplots(figsize=(6, 4))
        self.imgs.append(fig)

        prev_nest_data = [(data['prev_nest'][i][0], data['prev_nest'][i][1]) for i in range(3)]

        plots.plot_grouped_bar(
            fig, ax, prev_nest_data,
            xticks=[str(m) for m in mouse_types],
            labels=[("escape", "no escape")] * 3,
            colors=["teal", "cadetblue", "darkgray", "lightgray", "salmon", "lightpink"],
            bar_width=0.1, error_bars=True, points=False, show=False, close=True
        )
    
    def plot_traj_data(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        true_locs = []
        false_locs = []

        # Extract data for each mouse type
        for mouse_type in mouse_types:
            true_loc, false_loc = extract.extract_data_col_bymousetype(
                self.event_locs_file,
                data_start=154,
                escape=True,
                process_coords=True,
                get_escape_index=True,
                escape_col=3,
                mouse_type=mouse_type
            )
            true_locs.append(coordinates.normalize_length(true_loc))
            false_locs.append(coordinates.normalize_length(false_loc))

        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        self.imgs.append(fig)
        fig.suptitle("Trajectories After Stimulus")

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        for i, mouse_type in enumerate(mouse_types):
            # Escape plot
            ax_escape = axes_flat[i*2]
            ax_escape.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            ax_escape.set_title(f"{mouse_type} - escape")
            plots.time_plot(fig, ax_escape, true_locs[i], fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)

            # No escape plot
            ax_no_escape = axes_flat[i*2 + 1]
            ax_no_escape.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
            ax_no_escape.set_title(f"{mouse_type} - no escape")
            
            # Add colorbar only on the last subplot (optional)
            cbar_args = dict(cbar_dim=[0.92, 0.11, 0.015, 0.22], colorbar=True) if i == 2 else dict(colorbar=False)
            plots.time_plot(fig, ax_no_escape, false_locs[i], fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, **cbar_args)

    def plot_tort_data(self):
        min_path_length = 5
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]

        true_locs_list, false_locs_list = [], []
        prev_true_locs_list, prev_false_locs_list = [], []

        for mtype in mouse_types:
            true, false = extract.extract_data_col_bymousetype(
                self.event_locs_file, data_start=154, escape=True, process_coords=True,
                get_escape_index=True, escape_col=3, mouse_type=mtype
            )
            true_locs_list.append(true)
            false_locs_list.append(false)

            prev_true, prev_false = extract.extract_data_col_bymousetype(
                self.prev_esc_locs_file, data_start=4, escape=True,
                escape_col=3, mouse_type=mtype
            )
            prev_true_locs_list.append(prev_true)
            prev_false_locs_list.append(prev_false)

        # --- Tortuosity for escape paths ---
        true_dist_ratios, false_dist_ratios = [], []

        for loc_set_list in [true_locs_list, false_locs_list]:
            all_ratios = []
            for locs in loc_set_list:
                dist_ratios = []
                for points in locs:
                    points = [p for p in points if not np.isnan(p).any()]
                    if len(points) < 2:
                        continue
                    total_dist = sum(
                        calc.calc_dist_between_points(points[i], points[i - 1]) for i in range(1, len(points))
                    )
                    path_len = calc.calc_dist_between_points(points[-1], points[0])
                    if path_len >= min_path_length:
                        dist_ratios.append(total_dist / path_len)
                all_ratios.append(dist_ratios)
            true_dist_ratios = all_ratios if loc_set_list is true_locs_list else true_dist_ratios
            false_dist_ratios = all_ratios if loc_set_list is false_locs_list else false_dist_ratios

        # --- Plot 1: bar plot of true escape paths ---
        fig, ax = plt.subplots(figsize=(5, 4))
        self.imgs.append(fig)
        fig.suptitle("Tortuosity of Escape Path")
        plots.plot_bar(
            fig, ax, true_dist_ratios, "Mouse type", "Tortuosity of Escape Path (log)",
            bar_labels=mouse_types, colors=['teal', 'lightgray', 'salmon'],
            bar_width=0.2, error_bars=True, points=False, log_y=True,
            ylim=None, title=None, show=False, close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Tortuosity (log)")

        # --- Plot 2: grouped bar plot of true/false ---
        fig, ax = plt.subplots(figsize=(5, 5))
        self.imgs.append(fig)
        fig.suptitle("Tortuosity of Escape Path")
        grouped_data = list(zip(true_dist_ratios, false_dist_ratios))
        plots.plot_grouped_bar(
            fig, ax, grouped_data, mouse_types,
            labels=["escape", "no escape"] * 3,
            colors=["teal", "cadetblue", "darkgray", "lightgray", "salmon", "lightpink"],
            bar_width=0.1, error_bars=True, log_y=True,
            ylim=None, show=False, close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Tortuosity")

        # --- Plot 3: histogram ---
        fig, ax = plt.subplots(figsize=(6, 4))
        self.imgs.append(fig)
        fig.suptitle("Tortuosity of Escape Path")
        for ratios, label, color in zip(true_dist_ratios, mouse_types, ['teal', 'gray', 'salmon']):
            if len(ratios) == 0:
                continue
            weights = np.ones_like(ratios) * 100 / len(ratios)
            ax.hist(ratios, bins=20, alpha=0.45, label=label, color=color, weights=weights)
        ax.set_xlabel("Tortuosity")
        ax.set_ylabel("Frequency (%)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
        plt.tight_layout()

        # --- Previous tortuosity (1D path) ---
        prev_true_ratios, prev_false_ratios = [], []
        for locs_list in [prev_true_locs_list, prev_false_locs_list]:
            all_ratios = []
            for locs in locs_list:
                torts = []
                for vals in locs:
                    vals = [v for v in vals if not np.isnan(v)]
                    if not vals or vals[0] <= 700:
                        continue
                    total_diff = sum(abs(vals[i] - vals[i - 1]) for i in range(1, len(vals)))
                    path_len = abs(vals[-1] - vals[0])
                    if path_len >= min_path_length:
                        torts.append(total_diff / path_len)
                all_ratios.append(torts)
            prev_true_ratios = all_ratios if locs_list is prev_true_locs_list else prev_true_ratios
            prev_false_ratios = all_ratios if locs_list is prev_false_locs_list else prev_false_ratios

        # --- Plot 4: grouped bar for previous escape paths ---
        fig, ax = plt.subplots(figsize=(6, 5))
        self.imgs.append(fig)
        fig.suptitle("Tortuosity of Path Since Last Visiting the Nest")
        prev_grouped_data = list(zip(prev_true_ratios, prev_false_ratios))
        plots.plot_grouped_bar(
            fig, ax, prev_grouped_data, mouse_types,
            labels=["escape", "no escape"] * 3,
            colors=["teal", "cadetblue", "darkgray", "lightgray", "salmon", "lightpink"],
            bar_width=0.1, error_bars=True, log_y=True,
            ylim=None, show=False, close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Tortuosity of Path Since Last Visiting Nest (log)")

    def plot_behavior(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]

        # Extract data for each mouse and condition
        baseline_angles = [
            extract.extract_data_col_bymousetype(self.global_angles_file, data_start=4, data_end=5400, mouse_type=mt)
            for mt in mouse_types
        ]
        baseline_locs = [
            extract.extract_data_col_bymousetype(self.global_locs_file, data_start=4, data_end=5400, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]
        event_angles = [
            extract.extract_data_col_bymousetype(self.event_angles_file, data_start=154, mouse_type=mt)
            for mt in mouse_types
        ]
        event_locs = [
            extract.extract_data_col_bymousetype(self.event_locs_file, data_start=154, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]
        true_angles = []
        false_angles = []
        true_locs = []
        false_locs = []
        for mt in mouse_types:
            ta, fa = extract.extract_data_col_bymousetype(self.event_angles_file, data_start=154, escape=True, escape_col=3, mouse_type=mt)
            true_angles.append(ta)
            false_angles.append(fa)
            tl, fl = extract.extract_data_col_bymousetype(self.event_locs_file, data_start=154, escape=True, escape_col=3, process_coords=True, mouse_type=mt)
            true_locs.append(tl)
            false_locs.append(fl)

        # Prepare all sets for processing
        angle_sets = baseline_angles + event_angles + true_angles + false_angles
        locs_sets = baseline_locs + event_locs + true_locs + false_locs

        # Containers for behaviors per mouse and condition
        global_behaviors = [{} for _ in mouse_types]
        event_behaviors = [{} for _ in mouse_types]
        true_behaviors = [{} for _ in mouse_types]
        false_behaviors = [{} for _ in mouse_types]

        # Process behavior analysis for each set
        for i, (angle_set, locs_set) in enumerate(zip(angle_sets, locs_sets)):
            behavior_percentages_list = []
            for angles, locs in zip(angle_set, locs_set):
                behavior_percentages = behaviour.analyse_behavior(angles, locs, fps=30)
                behavior_percentages_list.append(behavior_percentages)

            mean_behavior_percentages = {
                behavior: np.nanmean([
                    behavior_data[behavior]
                    for behavior_data in behavior_percentages_list
                    if behavior_data is not None and behavior in behavior_data
                ])
                for behavior in (behavior_percentages_list[0] if behavior_percentages_list and behavior_percentages_list[0] is not None else {})
            }

            # Assign mean_behavior_percentages to correct dict based on i and mouse index
            mouse_idx = i % 3
            category_idx = i // 3  # 0: baseline, 1: event, 2: true, 3: false

            if category_idx == 0:
                global_behaviors[mouse_idx] = mean_behavior_percentages
            elif category_idx == 1:
                event_behaviors[mouse_idx] = mean_behavior_percentages
            elif category_idx == 2:
                true_behaviors[mouse_idx] = mean_behavior_percentages
            elif category_idx == 3:
                false_behaviors[mouse_idx] = mean_behavior_percentages

        categories = {
            "Baseline (Global)": global_behaviors,
            "Event (All)": event_behaviors,
            "Escape": true_behaviors,
            "No Escape": false_behaviors,
        }

        colors = ['#84206b', '#ae5097', '#ffd4f4', '#c6abeb']

        for category_name, behavior_list in categories.items():
            fig, axes = plt.subplots(len(behavior_list), 1, figsize=(6, 4 * len(behavior_list)))
            fig.suptitle(f"Behavior Proportions - {category_name}")

            if len(behavior_list) == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                behavior_data = behavior_list[i]
                if behavior_data:
                    ax.pie(
                        behavior_data.values(),
                        labels=behavior_data.keys(),
                        autopct='%1.1f%%',
                        colors=colors,
                        startangle=90
                    )
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center')

                ax.set_title(f"{mouse_types[i]}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            self.imgs.append(fig)
                
    def plot_arena_coverage_data(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]

        baseline_locs = [
            extract.extract_data_col_bymousetype(self.global_locs_file, data_start=4, data_end=5400, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]
        event_locs = [
            extract.extract_data_col_bymousetype(self.event_locs_file, data_start=154, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]

        baseline_coverage = [calc.calculate_arena_coverage(locs) for locs in baseline_locs]
        event_coverage = [calc.calculate_arena_coverage(locs) for locs in event_locs]
        
        all_dists = [[] for _ in range(3)]  # For baseline distances for each mouse type
        event_dists = [[] for _ in range(3)]  # For event distances for each mouse type

        # Combine loc sets and output containers
        all_locs_sets = [baseline_locs, event_locs]
        all_dists_sets = [all_dists, event_dists]

        # Loop over both baseline and event locs
        for locs_set_group, dists_group in zip(all_locs_sets, all_dists_sets):
            for i, locs_set in enumerate(locs_set_group):  # i corresponds to mouse type index
                for locs_list in locs_set:
                    # Clean locs
                    locs_list = [loc for loc in locs_list if isinstance(loc, (tuple, list)) and len(loc) == 2 and not np.isnan(loc).any()]
                    if len(locs_list) < 2:
                        continue
                    total_distance_covered = sum(
                        calc.calc_dist_between_points(locs_list[j], locs_list[j - 1])
                        for j in range(1, len(locs_list))
                    )
                    dists_group[i].append(total_distance_covered)

        # Apply conversion factor
        conversion_factor = 46.5 / 645
        all_dists = [[d * conversion_factor for d in dist_list] for dist_list in all_dists]
        event_dists = [[d * conversion_factor for d in dist_list] for dist_list in event_dists]

        # --- Arena Coverage Plot ---
        fig, ax = plt.subplots(figsize=(5, 4))
        self.imgs.append(fig)
        fig.suptitle("Arena Coverage at Baseline")
        plots.plot_bar(
            fig, ax,
            [np.mean(cov) for cov in baseline_coverage],  # average coverage per mouse type
            "Mouse Type", "Arena Coverage (%)",
            bar_labels=mouse_types,
            colors=["darkorange", "deeppink", "#f6d746"],  # adjust colors as needed
            bar_width=0.2,
            error_bars=True,
            points=False,
            log_y=False,
            show=False,
            close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Arena Coverage (%)")

        # --- Distance Covered Plot ---
        fig, ax = plt.subplots(figsize=(5, 4))
        self.imgs.append(fig)
        fig.suptitle("Total Distance Covered")
        plots.plot_bar(
            fig, ax,
            [np.mean(dist) for dist in all_dists],
            "Mouse Type", "Total Distance Covered (cm)",
            bar_labels=mouse_types,
            colors=["darkorange", "deeppink", "#f6d746"],
            bar_width=0.2,
            error_bars=True,
            points=False,
            log_y=False,
            show=False,
            close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Distance (cm)")


        # --- Grouped Bar Plot: Baseline vs Event Coverage ---
        fig, ax = plt.subplots(figsize=(5, 5))
        self.imgs.append(fig)
        fig.suptitle("Arena Coverage: Baseline vs Event")

        grouped_coverage = list(zip(
            [np.mean(b) for b in baseline_coverage],
            [np.mean(e) for e in event_coverage]
        ))

        plots.plot_grouped_bar(
            fig, ax,
            grouped_coverage,
            mouse_types,
            labels=["Baseline", "Event"] * len(mouse_types),
            colors=["darkorange", "orange", "deeppink", "hotpink", "#f6d746", "#ffd700"],
            bar_width=0.1,
            error_bars=True,
            log_y=False,
            show=False,
            close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Arena Coverage (%)")


        # --- Grouped Bar Plot: Distance Baseline vs Event ---
        fig, ax = plt.subplots(figsize=(5, 5))
        self.imgs.append(fig)
        fig.suptitle("Distance Covered: Baseline vs Event")

        grouped_distance = list(zip(
            [np.mean(d) for d in all_dists],
            [np.mean(d) for d in event_dists]
        ))

        plots.plot_grouped_bar(
            fig, ax,
            grouped_distance,
            mouse_types,
            labels=["Baseline", "Event"] * len(mouse_types),
            colors=["darkorange", "orange", "deeppink", "hotpink", "#f6d746", "#ffd700"],
            bar_width=0.1,
            error_bars=True,
            log_y=False,
            show=False,
            close=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Distance Covered (cm)")
        
    def plot_location_data(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        centre_roi = [200, 570, 620, 260]

        # Extract data for each mouse type
        event_locs = [
            extract.extract_data_col_bymousetype(self.event_locs_file, nested=True, data_start=154, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]
        baseline_locs = [
            extract.extract_data_col_bymousetype(self.global_locs_file, nested=True, data_start=3, data_end=5404, process_coords=True, mouse_type=mt)
            for mt in mouse_types
        ]

        # Analyze locations for center and edge
        event_centre_edge = [behaviour.analyse_locs(locs, 30, centre_roi) for locs in event_locs]
        baseline_centre_edge = [behaviour.analyse_locs(locs, 30, centre_roi) for locs in baseline_locs]

        # Categorize locations
        categorized_event_locs = [behaviour.categorise_location(locs) for locs in event_locs]

        # Colors for the three mouse types
        colors = ["deeppink", "#ff69b4", "darkorange", "#ffa500", "#f6d746", "#fffacd"]  # Deep pink, lighter pink, orange, lighter orange, yellow, lighter yellow

        # --- Plot 1: Percentage Time Spent at Centre vs Edge (Baseline) ---
        fig, ax = plt.subplots(figsize=(5, 4.5))
        self.imgs.append(fig)
        fig.suptitle("Percentage Time Spent at Centre of Arena vs Edge (Baseline)")
        plots.plot_grouped_bar(
            fig, ax,
            baseline_centre_edge,
            xticks=mouse_types,
            labels=["centre", "edge"] * len(mouse_types),
            colors=colors,
            bar_width=0.1,
            error_bars=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Time Spent (%)")

        # --- Plot 2: Percentage Time Spent at Centre vs Edge (Event) ---
        fig, ax = plt.subplots(figsize=(5, 4.5))
        self.imgs.append(fig)
        fig.suptitle("Percentage Time Spent at Centre of Arena vs Edge (Event)")
        plots.plot_grouped_bar(
            fig, ax,
            event_centre_edge,
            xticks=mouse_types,
            labels=["centre", "edge"] * len(mouse_types),
            colors=colors,
            bar_width=0.1,
            error_bars=True
        )
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Time Spent (%)")

    def stretch_trials(self, trials, target_length=60):
        stretched_trials = []
        for trial in trials:
            trial = np.array(trial, dtype=np.float64)
            orig_len = len(trial)

            if orig_len == 0:
                stretched_trials.append([np.nan] * target_length)
                continue

            if orig_len == target_length:
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
                # Fill starting NaNs with first valid value
                if np.isnan(stretched[0]):
                    first_valid = np.nanmin(stretched)
                    stretched[0] = first_valid
                for i in range(1, len(stretched)):
                    if np.isnan(stretched[i]):
                        stretched[i] = stretched[i - 1]
            else:
                for row in stretched:
                    if np.isnan(row[0]):
                        first_valid = np.nanmin(row)
                        row[0] = first_valid
                    for i in range(1, len(row)):
                        if np.isnan(row[i]):
                            row[i] = row[i - 1]

            stretched_trials.append(stretched.tolist())

        return stretched_trials

    def plot_distance_from_wall(self):
        mouse_types = [self.mouse_type_1, self.mouse_type_2, self.mouse_type_3]
        corners = [(119, 140), (750, 158), (756, 660), (100, 653)]
        conversion_factor = 46.5 / 645

        # Extract data for each mouse type
        true_locs = []
        false_locs = []
        for mouse_type in mouse_types:
            true_loc, false_loc = extract.extract_data_col_bymousetype(
                self.event_locs_file, data_start=4, data_end=604, escape=True, process_coords=True, escape_col=3, mouse_type=mouse_type
            )
            true_locs.append(true_loc)
            false_locs.append(false_loc)

        # Calculate distances from walls
        true_wall_dist = [
            [[calc.point_to_rect(point, corners) for point in sublist] for sublist in locs]
            for locs in true_locs
        ]
        false_wall_dist = [
            [[calc.point_to_rect(point, corners) for point in sublist] for sublist in locs]
            for locs in false_locs
        ]

        # Apply conversion factor
        true_wall_dist = [
            [[dist * conversion_factor for dist in trial] for trial in locs]
            for locs in true_wall_dist
        ]
        false_wall_dist = [
            [[dist * conversion_factor for dist in trial] for trial in locs]
            for locs in false_wall_dist
        ]

        # Plot heatmaps for each mouse type
        fig, axes = plt.subplots(3, 2, figsize=(14, 15))
        self.imgs.append(fig)
        fig.suptitle("Distance to Nearest Wall")

        for i, mouse_type in enumerate(mouse_types):
            # Escape heatmap
            plots.cmap_plot(
                fig, axes[i][0], true_wall_dist[i],
                ylabel="Distance (cm)", cmap="coolwarm_r", fps=30, vmin=0, vmax=250
            )
            axes[i][0].set_title(f"{mouse_type} - Escape")  # Set title outside cmap_plot

            # No escape heatmap
            plots.cmap_plot(
                fig, axes[i][1], false_wall_dist[i],
                ylabel="Distance (cm)", cmap="coolwarm_r", fps=30, vmin=0, vmax=250
            )
            axes[i][1].set_title(f"{mouse_type} - No Escape")  # Set title outside cmap_plot

        # Flatten the lists of lists into single 1D arrays for histogram plotting
        flat_true_wall_dist = [np.hstack(locs) for locs in true_wall_dist]
        flat_false_wall_dist = [np.hstack(locs) for locs in false_wall_dist]

        # Plot histograms for each mouse type
        fig, ax = plt.subplots(figsize=(6, 4))
        self.imgs.append(fig)
        fig.suptitle("Distance to Nearest Wall - Histogram")

        # Colors for the three mouse types
        colors = ["deeppink", "#ff69b4", "darkorange", "#ffa500", "#f6d746", "#fffacd"]

        for i, mouse_type in enumerate(mouse_types):
            if len(flat_true_wall_dist[i]) > 0:
                weights = np.ones_like(flat_true_wall_dist[i]) * 100 / len(flat_true_wall_dist[i])
                ax.hist(flat_true_wall_dist[i], bins=30, alpha=0.7, label=f"{mouse_type} - Escape", color=colors[i * 2], weights=weights)

            if len(flat_false_wall_dist[i]) > 0:
                weights = np.ones_like(flat_false_wall_dist[i]) * 100 / len(flat_false_wall_dist[i])
                ax.hist(flat_false_wall_dist[i], bins=30, alpha=0.5, label=f"{mouse_type} - No Escape", color=colors[i * 2 + 1])

        ax.set_xlabel("Distance (cm)")
        ax.set_ylabel("Percentage Frequency (%)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.legend()
        plt.tight_layout()
        
    def plot_path_similarity(self):
        # --- Extract data ---
        wt_true_before_locs, wt_false_before_locs, blind_true_before_locs, blind_false_before_locs = extract.extract_data(
            self.event_locs_file, data_start=4, data_end=154, escape=True, process_coords=True, get_escape_index=True, escape_col=3)

        wt_true_stim_locs, wt_false_stim_locs, blind_true_stim_locs, blind_false_stim_locs = extract.extract_data(
            self.event_locs_file, data_start=154, escape=True, process_coords=True, get_escape_index=True, escape_col=3)

        # --- Example path (WT True, first trace) ---
        before_locs = np.array(wt_true_before_locs[1])[::-1]
        stim_locs = np.array(wt_true_stim_locs[1])

        # Remove NaNs
        before_locs = before_locs[~np.isnan(before_locs).any(axis=1)]
        stim_locs = stim_locs[~np.isnan(stim_locs).any(axis=1)]

        # Normalize
        normalized = coordinates.normalize_length([before_locs, stim_locs])
        before_locs = np.array(normalized[0])
        stim_locs = np.array(normalized[1])

        # Build KDTree for stim_locs
        stim_tree = cKDTree(stim_locs)

        # Find nearest neighbors
        distances, indices = stim_tree.query(before_locs, k=1)

        # --- Plot the paths ---
        fig, ax1 = plt.subplots(figsize=(8, 6))
        self.figs.append(fig)

        x_limits = (0, 850)
        y_limits = (755, 0)

        ax1.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        ax1.plot(before_locs[:, 0], before_locs[:, 1], color='darkmagenta', marker='o', linestyle='-', label='Before Stimulus', markersize=2)
        ax1.plot(stim_locs[:, 0], stim_locs[:, 1], color='orange', marker='o', linestyle='-', label='After Stimulus', markersize=2)

        for i, idx in enumerate(indices):
            x_vals = [before_locs[i][0], stim_locs[idx][0]]
            y_vals = [before_locs[i][1], stim_locs[idx][1]]
            ax1.plot(x_vals, y_vals, color='k', linestyle=(0, (1, 0.5)), alpha=0.6)

        ax1.set_title("Path Comparison: Before vs After Stimulus")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.legend()

        # Remove top and right spines
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Compute similarity score
        wt_true_scores = calc.compute_nearest_euclidean_similarity(wt_true_before_locs, wt_true_stim_locs)

        # Add similarity score text to plot
        similarity_text = f"Similarity Score: {1 -(np.mean(wt_true_scores)):.3f}"
        ax1.text(0.05, 0.95, similarity_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        plt.tight_layout()
        self.imgs.append(fig)

        wt_false_scores = calc.compute_nearest_euclidean_similarity(wt_false_before_locs, wt_false_stim_locs)
        blind_true_scores = calc.compute_nearest_euclidean_similarity(blind_true_before_locs, blind_true_stim_locs)
        blind_false_scores = calc.compute_nearest_euclidean_similarity(blind_false_before_locs, blind_false_stim_locs)

        fig, ax = plt.subplots(figsize=(5, 5))
        self.figs.append(fig)
        self.imgs.append(fig)
        fig.suptitle("Path Similarity Score")
        
        plots.plot_grouped_bar_chart(
            fig, ax,
            wt_true_scores, wt_false_scores, blind_true_scores, blind_false_scores,
            xticks=["WT", f"{self.mouse_type}"],
            labels=["escape", "no escape", "escape", "no escape"],
            colors=["teal", "cadetblue", "darkgray", "lightgray"],
            bar_width=0.1,
            error_bars=True,
            points=False,
            log_y=True,
            ylim=None,
            show=False,
            close=False
        )
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_ylabel("Path Similarity Score (1 / (1 + avg dist))")
        plt.tight_layout()
                
                # Combine all scores into a DataFrame for seaborn plotting
        data = {
            'score': np.concatenate([wt_true_scores, wt_false_scores, blind_true_scores, blind_false_scores]),
            'group': ['WT_escape'] * len(wt_true_scores) +
                    ['WT_no_escape'] * len(wt_false_scores) +
                    [f'{self.mouse_type}_escape'] * len(blind_true_scores) +
                    [f'{self.mouse_type}_no_escape'] * len(blind_false_scores)
        }

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(7, 5))
        self.imgs.append(fig)
        fig.suptitle("Path Similarity Score")

        sns.violinplot(x='group', y='score', data=df, ax=ax, hue='group', palette=["teal", "cadetblue", "darkgray", "lightgray"], legend=False)


        ax.set_ylabel("Path Similarity Score (1 / (1 + avg dist))")
        ax.set_xlabel("Group")
        ax.set_yscale("log")
        ax.set_title("Distribution of Path Similarity Scores")

        plt.tight_layout()
        
    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder)
        
        if self.save_imgs:
            if self.imgs:
                files.save_images(self.imgs, self.folder)

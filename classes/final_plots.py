import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import files, utils
from processing import plots, data, angles
import matplotlib.image as mpimg

class FinalPlots:
    def __init__(self, folder, mouse_type, save_figs=True):
        self.fps = 30
        self.length = 10
        self.t_minus = 5
        self.t_plus = 5

        self.folder = folder
        image_file = "/Users/annakennedy/Documents/projects/angle_plots/images/arena.tif"
        self.background_image = mpimg.imread(image_file)

        self.figs = []
        self.save_figs = save_figs

        self.parent_folder = os.path.dirname(self.folder)
        self.global_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_data")), None)
        
        self.global_angles_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("locs.csv")), None)
        self.global_distances_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("distances.csv")), None)
        self.global_speeds_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("speeds.csv")), None)
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
        self.avg_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_angles.csv")), None)
        self.avg_dist_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_distances.csv")), None)
        self.avg_speeds_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_speeds.csv")), None)

        self.mouse_type = mouse_type
    
    def plot_global_data(self):

        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, nested=False, data_start=3, data_end=5400, process_coords=True, escape_col=None)
        all_wt_locs, all_blind_locs = data.extract_data(self.global_locs_file, nested=True, data_start=3, process_coords=True, escape_col=None)
        self.wt_baseline_angles, self.blind_baseline_angles = data.extract_data(self.global_angles_file, nested=False, data_start=3, data_end=5400, escape_col=None)
        self.wt_baseline_distances, self.blind_baseline_distances = data.extract_data(self.global_distances_file, nested=False, data_start=3, data_end=5400, escape_col=None)
        wt_locs, blind_locs = data.extract_data(self.event_locs_file, nested=False, data_start=154, escape=False, process_coords=True, escape_col=None)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)
        plots.plot_coords(fig, ax1, wt_baseline_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=700, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        plots.plot_coords(fig, ax2, blind_baseline_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=700, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)
        plots.plot_coords(fig, ax1, wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        plots.plot_coords(fig, ax2, blind_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)

    def plot_event_data(self):
        wt_before_angles, blind_before_angles = data.extract_data(self.event_angles_file, nested=False, data_start=4, data_end=154)
        wt_after_angles, blind_after_angles = data.extract_data(self.after_angles_file, nested=False, data_start=4)
        wt_during_angles, blind_during_angles = data.extract_data(self.during_angles_file, nested=False, data_start=4)
        
        # Plot polar plots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_polar_chart(fig, ax1, self.wt_baseline_angles, bins=36, show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_polar_chart(fig, ax2, wt_before_angles, bins=36, show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_polar_chart(fig, ax3, wt_during_angles, bins=36, show=False, close=True)
        ax4.set_title("After Stimulus / Exit from Nest")
        plots.plot_polar_chart(fig, ax4, wt_after_angles, bins=36, show=False, close=True)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_polar_chart(fig, ax1, self.blind_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_polar_chart(fig, ax2, blind_before_angles, bins=36, show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_polar_chart(fig, ax3, blind_during_angles, bins=36, show=False, close=True)
        ax4.set_title("After Stimulus / Exit from Nest")
        plots.plot_polar_chart(fig, ax4, blind_after_angles, bins=36, show=False, close=True)

    def plot_avgs_data(self):
        wt_avg_angles, blind_avg_angles = data.extract_data(self.avg_angles_file, data_start=3, escape=False)
        wt_true_angles, blind_true_angles, wt_false_angles, blind_false_angles = data.extract_data(self.event_angles_file, data_start=4, escape=True, escape_col=3)
        wt_avg_dist, blind_avg_dist = data.extract_data(self.avg_dist_file, data_start=3, escape=False)
        wt_true_dist, blind_true_dist, wt_false_dist, blind_false_dist = data.extract_data(self.event_distances_file, data_start=4, escape=True, escape_col=3)
        wt_avg_speeds, blind_avg_speeds = data.extract_data(self.avg_speeds_file, data_start=3, escape=False)
        wt_true_speeds, blind_true_speeds, wt_false_speeds, blind_false_speeds = data.extract_data(self.event_speeds_file, data_start=4, escape=True, escape_col=3)
        
        stim_data = [0]*150 + [1]*300 + [0]*150
        frame_time = (1./self.fps)
        norm_event_time = np.arange(-self.t_minus, (self.length + self.t_plus), frame_time)

        # Compute averages using list comprehensions
        avg_angle_data = [
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_avg_angles]) for i in range(max(map(len, wt_avg_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in blind_avg_angles]) for i in range(max(map(len, blind_avg_angles)))]
        ]
        
        avg_dist_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_avg_dist]) for i in range(max(map(len, wt_avg_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_avg_dist]) for i in range(max(map(len, blind_avg_dist)))]
        ]

        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_avg_speeds]) for i in range(max(map(len, wt_avg_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_avg_speeds]) for i in range(max(map(len, blind_avg_speeds)))]
        ]

        avg_esc_angle_data = [
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_true_angles]) for i in range(max(map(len, wt_true_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_false_angles]) for i in range(max(map(len, wt_false_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in blind_true_angles]) for i in range(max(map(len, blind_true_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in blind_false_angles]) for i in range(max(map(len, blind_false_angles)))]
        ]
        
        avg_esc_dist_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_true_dist]) for i in range(max(map(len, wt_true_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_false_dist]) for i in range(max(map(len, wt_false_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_true_dist]) for i in range(max(map(len, blind_true_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_false_dist]) for i in range(max(map(len, blind_false_dist)))],
        ]

        avg_esc_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_true_speeds]) for i in range(max(map(len, wt_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_false_speeds]) for i in range(max(map(len, wt_false_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_true_speeds]) for i in range(max(map(len, blind_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_false_speeds]) for i in range(max(map(len, blind_false_speeds)))],
        ]

        titles = ["Facing angle", "Distance from Exit (cm)", "Speed"]
        colours = ["tab:blue", "mediumseagreen", "red"]
        data_limits = [(-185, 20), (0, 55), (0,255)]

        for df, title, colour, limits in zip([avg_angle_data, avg_dist_data, avg_speeds_data], titles, colours, data_limits):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT", f"{self.mouse_type}"]
            for i, (values, ax, subtitle) in enumerate(zip(df, axes, subtitles)):
                plots.two_plots(fig, ax, norm_event_time, values, stim_data, colour, 
                                "Time (seconds)", title, "Stimulus", limits, title=subtitle)
            self.figs.append(fig)
        
        for df, title, colour, limits in zip([avg_speeds_data], ["Speed"], ["red"], [(0, 700)]):
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT", f"{self.mouse_type}"]

            for i, (values, ax, subtitle) in enumerate(zip(df, axes, subtitles)):
                # Plot individual speed sets in light grey for WT or blind
                individual_speeds = wt_avg_speeds if i == 0 else blind_avg_speeds
                for speed_trace in individual_speeds:
                    ax.plot(norm_event_time[:len(speed_trace)], speed_trace, color='lightgrey', alpha=0.5)
                # Call two_plots to overlay the average and stimulus
                plots.two_plots(fig, ax, norm_event_time, values, stim_data, colour, 
                                "Time (seconds)", title, "Stimulus", limits, title=subtitle)
            
            self.figs.append(fig)

        for df, title, colour, limits in zip([avg_esc_angle_data, avg_esc_dist_data, avg_esc_speeds_data], titles, colours, data_limits):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT - escape", "WT - no escape", f"{self.mouse_type} - escape", f"{self.mouse_type} - no escape"]
            for i, (values, subtitle) in enumerate(zip(df, subtitles)):
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                plots.two_plots(fig, ax, norm_event_time, values, stim_data, colour, "Time (seconds)", title, "Stimulus", limits, title=subtitle)
            self.figs.append(fig)

    def plot_stats_data(self):
        wt_time, blind_time = data.extract_data_rows(self.escape_stats_file, data_row=6)
        wt_dist, blind_dist = data.extract_data_rows(self.escape_stats_file, data_row=8)
        wt_esc_avg, blind_esc_avg = data.extract_data_rows(self.escape_success_file, data_row=3)
        wt_true_data, wt_false_data, blind_true_data, blind_false_data = data.extract_data_rows(self.escape_stats_file, data_row=7, escape=True)
        wt_age, blind_age = data.extract_data_rows(self.escape_success_file, data_row=2)
        wt_time_angle, blind_time_angle = data.extract_data_rows(self.escape_stats_file, data_row=9)

        wt_time = np.array(wt_time)
        blind_time = np.array(blind_time)
        wt_time[wt_time >= 15] = np.nan
        blind_time[blind_time >= 15] = np.nan

        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax,
                                wt_esc_avg, 
                                blind_esc_avg,
                                "Mouse Type", 
                                "Escape probability (%)", 
                                "WT", 
                                f"{self.mouse_type}", 
                                color1='tab:blue', 
                                color2='mediumseagreen',
                                ylim=(0,102),
                                bar_width=0.2,
                                points=True,
                                error_bars=True,
                                show=False,
                                close=True)
        
        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax,
                                wt_time, 
                                blind_time,
                                "Mouse Type", 
                                "Time to Escape (s)", 
                                "WT", 
                                f"{self.mouse_type}", 
                                color1='tab:blue', 
                                color2='mediumseagreen',
                                ylim=None,
                                bar_width=0.2,
                                points=True,
                                error_bars=True,
                                show=False,
                                close=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.figs.append(fig)
        plots.plot_scatter_trendline(fig, ax1,
                                    wt_age, 
                                    wt_esc_avg, 
                                    "Mouse Age", 
                                    "Escape Success Rate (%)",
                                    "WT",
                                    color='tab:blue', 
                                    marker_size=20, 
                                    show=False, 
                                    close=True)
        
        plots.plot_scatter_trendline(fig, ax2,
                                    blind_age, 
                                    blind_esc_avg, 
                                    "Mouse Age", 
                                    "Escape Success Rate (%)",
                                    f"{self.mouse_type}",
                                    color='mediumseagreen', 
                                    marker_size=20, 
                                    show=False, 
                                    close=True)

        fig, ax = plt.subplots()
        self.figs.append(fig)
        plots.plot_two_scatter_trendline(fig, ax,
                                        wt_dist, 
                                        wt_time, 
                                        blind_dist, 
                                        blind_time, 
                                        "Distance From Exit at Stim (cm)", 
                                        "Time to Escape (s)",
                                        "WT", 
                                        f"{self.mouse_type}",
                                        group1_color='tab:blue', 
                                        group2_color='mediumseagreen', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)

        fig, ax = plt.subplots()
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_data, 
                                    wt_false_data, 
                                    blind_true_data, 
                                    blind_false_data, 
                                    ["WT - escape", "WT - no escape", f"{self.mouse_type} - escape", f"{self.mouse_type} - no escape"],
                                    "Mouse Type", 
                                    "Time Since Previous Escape",
                                    colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35, 
                                    show=False, 
                                    close=True)
        
        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_time_angle, blind_time_angle, "Mouse Type", "Average Time to Face Exit (s)", "WT", f"{self.mouse_type}",
                                color1='tab:blue', color2='mediumseagreen', ylim=(0,8), bar_width=0.2, show=False, close=True)

        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data_rows(self.escape_stats_file, data_row=4, escape=True)
        
        # Define x and y limits
        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.figs.append(fig)
        ax1.set_title('WT Mice - escape')
        ax1.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax1, wt_true_locs, point_color='tab:blue', background_color=None, mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax2.set_title('WT Mice - no escape')
        ax2.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax2, wt_false_locs, point_color='tab:blue', background_color=None, mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax3.set_title(f'{self.mouse_type} Mice - escape')
        ax3.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax3, blind_true_locs, point_color='tab:blue', background_color=None, mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax4.set_title(f'{self.mouse_type} Mice - no escape')
        ax4.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax4, blind_false_locs, point_color='tab:blue', background_color=None, mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        plt.tight_layout()

    def plot_tort_data(self):
        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data(
            self.event_locs_file, data_start=154, escape=True, process_coords=True, get_escape_index=True, escape_col=3)
        min_path_length = 5

        wt_true_dist_ratio = []
        wt_false_dist_ratio = []
        blind_true_dist_ratio = []
        blind_false_dist_ratio = []
        distance_sets = [wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs]

        for distance_set in distance_sets:
            distance_diff_list = []
            path_length_list = []
            distance_ratio_list = []

            for distance_list in distance_set:
                # Remove any lists that contain NaNs and ensure there are enough points to calculate distances
                distance_list = [distance for distance in distance_list if not np.isnan(distance).any()]
                if len(distance_list) < 2:
                    continue  # Skip lists with fewer than two valid points

                # Calculate total distance covered
                total_distance_covered = sum(
                    utils.calc_dist_between_points(distance_list[i], distance_list[i - 1])
                    for i in range(1, len(distance_list))
                )
                distance_diff_list.append(total_distance_covered)

                # Calculate path length (absolute difference between start and end points)
                path_length = utils.calc_dist_between_points(distance_list[-1], distance_list[0])
                path_length_list.append(path_length)
                
                if path_length >= min_path_length:
                    # Calculate distance tortuosity
                    dist_ratio = total_distance_covered / path_length
                    distance_ratio_list.append(dist_ratio)

            # Assign to the appropriate list based on the distance set
            if distance_set == wt_true_locs:
                wt_true_dist_ratio = distance_ratio_list
            elif distance_set == wt_false_locs:
                wt_false_dist_ratio = distance_ratio_list
            elif distance_set == blind_true_locs:
                blind_true_dist_ratio = distance_ratio_list
            elif distance_set == blind_false_locs:
                blind_false_dist_ratio = distance_ratio_list

        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(
            fig, ax, wt_true_dist_ratio, blind_true_dist_ratio,
            "Mouse type", "Tortuosity of Escape Path (log)",
            "WT", f"{self.mouse_type}",
            color1='tab:blue', color2='mediumseagreen', ylim=None,
            bar_width=0.2, points=True, log_y=True,
            title=None, show=False, close=True)
        
        fig, ax = plt.subplots(figsize=(8,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(
            fig, ax, wt_true_dist_ratio, wt_false_dist_ratio, 
            blind_true_dist_ratio, blind_false_dist_ratio,
            ["WT - escape", "WT - no escape", f"{self.mouse_type} - escape", f"{self.mouse_type} - no escape"],
            "Mouse Type", "Tortuosity of escape path / after stimulus (log)",
            colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
            bar_width=0.35, log_y=True, show=False, close=True)

    def plot_prev_tort(self):
        wt_true_prev_esc, wt_false_prev_esc, blind_true_prev_esc, blind_false_prev_esc = data.extract_data(self.prev_esc_locs_file, data_start=4, escape=True, escape_col=3)
        min_path_length = 5 

        wt_true_tort = []
        wt_false_tort = []
        blind_true_tort = []
        blind_false_tort = []

        distance_sets = [wt_true_prev_esc, wt_false_prev_esc, blind_true_prev_esc, blind_false_prev_esc]
        
        for distance_set in distance_sets:
            
            tort_list = []

            for distance_list in distance_set:

                distance_list = [distance for distance in distance_list if not np.isnan(distance)]
                if distance_list and distance_list[0] < 5:
                    total_distance_difference = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                    path_length = abs(distance_list[-1] - distance_list[0])
                    if path_length != 0 and path_length >= min_path_length:  # To avoid division by zero
                        dist_ratio = total_distance_difference / path_length
                        tort_list.append(dist_ratio)
            
            if distance_set == wt_true_prev_esc:
                wt_true_tort = tort_list
            elif distance_set == wt_false_prev_esc:
                wt_false_tort = tort_list
            elif distance_set == blind_true_prev_esc:
                blind_true_tort = tort_list
            elif distance_set == blind_false_prev_esc:
                blind_false_tort = tort_list
        
        fig, ax = plt.subplots(figsize=(8,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_tort, 
                                    wt_false_tort, 
                                    blind_true_tort, 
                                    blind_false_tort, 
                                    ["WT - escape", "WT - no escape", f"{self.mouse_type} - escape", f"{self.mouse_type} - no escape"],
                                    "Mouse Type", 
                                    "Tortuosity of Path From Previous Escape (log)",
                                    colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35,
                                    log_y=True, 
                                    show=False, 
                                    close=True)

    def plot_traj_data(self):
        wt_true_locs, blind_true_locs, wt_false_locs, blind_false_locs = data.extract_data(self.event_locs_file, escape=True, process_coords=True, get_escape_index=True, escape_col=3)

        norm_true_wt_locs = angles.normalize_length(wt_true_locs)
        norm_true_blind_locs = angles.normalize_length(blind_true_locs)
        norm_false_wt_locs = angles.normalize_length(wt_false_locs)
        norm_false_blind_locs = angles.normalize_length(blind_false_locs)

        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(14,10))
        self.figs.append(fig)
        ax1.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax1, norm_true_wt_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax2.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax2, norm_false_wt_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax3.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax3, norm_true_blind_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax4.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax4, norm_false_blind_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)

    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import files
from processing import plots, data, angles

class FinalPlots:
    def __init__(self, folder, save_figs=True):
        self.fps = 30
        self.length = 10
        self.t_minus = 5
        self.t_plus = 5

        self.folder = folder

        self.figs = []
        self.save_figs = save_figs

        self.parent_folder = os.path.dirname(self.folder)
        self.global_file = next((os.path.join(self.parent_folder, file) for file in os.listdir(self.parent_folder) if file.endswith("global_data")), None)
        
        self.global_angles_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("locs.csv")), None)
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

    def plot_global_data(self):

        wt_baseline_locs, rd1_baseline_locs = data.extract_data_columns(self.global_locs_file, data_start=3, data_end=5400)
        self.wt_baseline_angles, self.rd1_baseline_angles = data.extract_data_columns(self.global_angles_file, data_start=3, data_end=5400)
        wt_locs, rd1_locs = data.extract_data_columns(self.event_locs_file, data_start=4, escape=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)
        plots.plot_str_coords(fig, ax1, wt_baseline_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=1000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        plots.plot_str_coords(fig, ax2, rd1_baseline_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=1000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)
        plots.plot_str_coords(fig, ax1, wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=100, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        plots.plot_str_coords(fig, ax2, rd1_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=100, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)

    def plot_event_data(self):

        wt_before_angles, rd1_before_angles = data.extract_data_columns(self.event_angles_file, data_start=4, data_end=154, escape=False)
        wt_after_angles, rd1_after_angles = data.extract_data_columns(self.after_angles_file, data_start=4, escape=False)
        wt_during_angles, rd1_during_angles = data.extract_data_columns(self.during_angles_file, data_start=4, escape=False)
        wt_true_after_angles, wt_false_after_angles, rd1_true_after_angles, rd1_false_after_angles = data.extract_data_columns(self.after_angles_file, data_start=4, escape=True)
        wt_true_during_angles, wt_false_during_angles, rd1_true_during_angles, rd1_false_during_angles = data.extract_data_columns(self.during_angles_file, data_start=4, escape=True)

        #plot polar plots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_str_polar_chart(fig, ax1, self.wt_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_str_polar_chart(fig, ax2, wt_before_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_str_polar_chart(fig, ax3, wt_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title("After Stimulus / Exit from Nest")
        plots.plot_str_polar_chart(fig, ax4, wt_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_str_polar_chart(fig, ax1, self.rd1_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_str_polar_chart(fig, ax2, rd1_before_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_str_polar_chart(fig, ax3, rd1_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title("After Stimulus / Exit from Nest")
        plots.plot_str_polar_chart(fig, ax4, rd1_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles of WT and RD1 Mice Before Events / Time to Escape')
        self.figs.append(fig)
        ax1.set_title('WT - escape')
        plots.plot_str_polar_chart(fig, ax1, wt_true_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('WT - no escape')
        plots.plot_str_polar_chart(fig, ax2, wt_false_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title('RD1 - escape')
        plots.plot_str_polar_chart(fig, ax3, rd1_true_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title('RD1 - no escape')
        plots.plot_str_polar_chart(fig, ax4, rd1_false_during_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles of WT and RD1 Mice After Events / Return from Nest')
        self.figs.append(fig)
        ax1.set_title('WT - escape')
        plots.plot_str_polar_chart(fig, ax1, wt_true_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('WT - no escape')
        plots.plot_str_polar_chart(fig, ax2, wt_false_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title('RD1 - escape')
        plots.plot_str_polar_chart(fig, ax3, rd1_true_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title('RD1 - no escape')
        plots.plot_str_polar_chart(fig, ax4, rd1_false_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

    def plot_avgs_data(self):
        wt_avg_angles, rd1_avg_angles = data.extract_avg_data(self.avg_angles_file, escape=False)
        wt_true_angles, rd1_true_angles, wt_false_angles, rd1_false_angles = data.extract_avg_data(self.event_angles_file, escape=True)
        wt_avg_dist, rd1_avg_dist = data.extract_avg_data(self.avg_dist_file, escape=False)
        wt_true_dist, rd1_true_dist, wt_false_dist, rd1_false_dist = data.extract_avg_data(self.event_distances_file, escape=True)
        wt_avg_speeds, rd1_avg_speeds = data.extract_avg_data(self.avg_speeds_file, escape=False)
        wt_true_speeds, rd1_true_speeds, wt_false_speeds, rd1_false_speeds = data.extract_avg_data(self.event_speeds_file, escape=True)
        
        stim_data = [0]*150 + [1]*300 + [0]*150
        frame_time = (1./self.fps)
        norm_event_time = np.arange(-self.t_minus, (self.length + self.t_plus), frame_time)

        # Compute averages using list comprehensions
        avg_angle_data = [
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_avg_angles]) for i in range(max(map(len, wt_avg_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in rd1_avg_angles]) for i in range(max(map(len, rd1_avg_angles)))]
        ]
        
        avg_dist_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_avg_dist]) for i in range(max(map(len, wt_avg_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_avg_dist]) for i in range(max(map(len, rd1_avg_dist)))]
        ]

        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_avg_speeds]) for i in range(max(map(len, wt_avg_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_avg_speeds]) for i in range(max(map(len, rd1_avg_speeds)))]
        ]

        avg_esc_angle_data = [
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_true_angles]) for i in range(max(map(len, wt_true_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in wt_false_angles]) for i in range(max(map(len, wt_false_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in rd1_true_angles]) for i in range(max(map(len, rd1_true_angles)))],
            [0 - np.nanmean([abs(lst[i]) if i < len(lst) else np.nan for lst in rd1_false_angles]) for i in range(max(map(len, rd1_false_angles)))]
        ]
        
        avg_esc_dist_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_true_dist]) for i in range(max(map(len, wt_true_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_false_dist]) for i in range(max(map(len, wt_false_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_true_dist]) for i in range(max(map(len, rd1_true_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_false_dist]) for i in range(max(map(len, rd1_false_dist)))],
        ]

        avg_esc_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_true_speeds]) for i in range(max(map(len, wt_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_false_speeds]) for i in range(max(map(len, wt_false_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_true_speeds]) for i in range(max(map(len, rd1_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in rd1_false_speeds]) for i in range(max(map(len, rd1_false_speeds)))],
        ]

        titles = ["Facing angle", "Distance from Exit (cm)", "Speed"]
        colours = ["tab:blue", "mediumseagreen", "red"]
        data_limits = [(-185, 20), (0, 55), (0,255)]

        for df, title, colour, limits in zip([avg_angle_data, avg_dist_data, avg_speeds_data], titles, colours, data_limits):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT", "rd1"]
            for values, ax, subtitle in zip(df, axes, subtitles):
                plots.two_plots(fig, ax, norm_event_time, values, stim_data, colour, "Time (seconds)", title, "Stimulus", limits, title=subtitle, )
            self.figs.append(fig)

        for df, title, colour, limits in zip([avg_esc_angle_data, avg_esc_dist_data, avg_esc_speeds_data], titles, colours, data_limits):
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT - escape", "WT - no escape", "rd1 - escape", "rd1 - no escape"]
            for i, (values, subtitle) in enumerate(zip(df, subtitles)):
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                plots.two_plots(fig, ax, norm_event_time, values, stim_data, colour, "Time (seconds)", title, "Stimulus", limits, title=subtitle)
            self.figs.append(fig)

    def plot_stats_data(self):
        wt_time, rd1_time = data.extract_data_rows(self.escape_stats_file, data_row=6)
        wt_dist, rd1_dist = data.extract_data_rows(self.escape_stats_file, data_row=8)
        wt_esc_avg, rd1_esc_avg = data.extract_data_rows(self.escape_success_file, data_row=3)
        wt_true_data, wt_false_data, rd1_true_data, rd1_false_data = data.extract_data_rows(self.escape_stats_file, data_row=7, escape=True)
        wt_true_locs, wt_false_locs, rd1_true_locs, rd1_false_locs = data.extract_data_rows(self.escape_stats_file, data_row=4, escape=True)
        wt_age, rd1_age = data.extract_data_rows(self.escape_success_file, data_row=2)
        wt_time_angle, rd1_time_angle = data.extract_data_rows(self.escape_stats_file, data_row=9)

        wt_time = np.array(wt_time)
        rd1_time = np.array(rd1_time)
        wt_time[wt_time >= 15] = np.nan
        rd1_time[rd1_time >= 15] = np.nan

        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax,
                                wt_esc_avg, 
                                rd1_esc_avg,
                                "Mouse Type", 
                                "Escape probability (%)", 
                                "WT", 
                                "rd1", 
                                color1='tab:blue', 
                                color2='mediumseagreen',
                                ylim=(0,102),
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
                                        "Average Escape Success Rate (%)",
                                        "WT",
                                        color='tab:blue', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)
        plots.plot_scatter_trendline(fig, ax2,
                                        rd1_age, 
                                        rd1_esc_avg, 
                                        "Mouse Age", 
                                        "Average Escape Success Rate (%)",
                                        "RD1",
                                        color='mediumseagreen', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)

        fig, ax = plt.subplots()
        self.figs.append(fig)
        plots.plot_two_scatter_trendline(fig, ax,
                                        wt_dist, 
                                        wt_time, 
                                        rd1_dist, 
                                        rd1_time, 
                                        "Distance From Exit at Stim (cm)", 
                                        "Time to Escape (s)",
                                        "WT", 
                                        "rd1",
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
                                    rd1_true_data, 
                                    rd1_false_data, 
                                    ["WT - escape", "WT - no escape", "rd1 - escape", "rd1 - no escape"],
                                    "Mouse Type", 
                                    "Time Since Previous Escape",
                                    colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35, 
                                    show=False, 
                                    close=True)
        
        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_time_angle, rd1_time_angle, "Mouse Type", "Average Time to Face Exit (s)", "WT", "RD1",
                            color1='tab:blue', color2='mediumseagreen', ylim=(0,8), bar_width=0.2, show=False, close=True)
        

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.figs.append(fig)
        ax1.set_title('WT Mice - escape')
        plots.scatter_plot_with_stats(fig, ax1, wt_true_locs, point_color='tab:blue', background_color='black', mean_marker='o', x_limits=(80,790), y_limits=(670,70), show=False, close=True)
        ax2.set_title('WT Mice - no escape')
        plots.scatter_plot_with_stats(fig, ax2, wt_false_locs, point_color='tab:blue', background_color='black', mean_marker='o', x_limits=(80,790), y_limits=(670,70), show=False, close=True)
        ax3.set_title('rd1 Mice - escape')
        plots.scatter_plot_with_stats(fig, ax3, rd1_true_locs, point_color='tab:blue', background_color='black', mean_marker='o', x_limits=(80,790), y_limits=(670,70), show=False, close=True)
        ax4.set_title('rd1 Mice - no escape')
        plots.scatter_plot_with_stats(fig, ax4, rd1_false_locs, point_color='tab:blue', background_color='black', mean_marker='o', x_limits=(80,790), y_limits=(670,70), show=False, close=True)

    def plot_tort_data(self):
        wt_distances, rd1_distances = data.extract_escape_data(self.event_distances_file)

        wt_dist_ratio = []
        rd1_dist_ratio = []
        distance_sets = [wt_distances, rd1_distances]

        for distance_set in distance_sets:

            distance_diff_list = []
            path_length_list = []
            distance_ratio_list = []

            for distance_list in distance_set:

                total_distance_covered = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                distance_diff_list.append(total_distance_covered)

                # Calculate path length (absolute difference between start and end points)
                path_length = abs(distance_list[-1] - distance_list[0])
                path_length_list.append(path_length)

                #calculate distance tortuosity
                dist_ratio = total_distance_covered / path_length
                distance_ratio_list.append(dist_ratio)

                if distance_set == wt_distances:
                    wt_dist_ratio = distance_ratio_list
                elif distance_set == rd1_distances:
                    rd1_dist_ratio = distance_ratio_list

        wt_true_distances, wt_false_distances, rd1_true_distances, rd1_false_distances = data.extract_tort_data(self.event_distances_file, start_row=154)
        wt_true_dist = []
        rd1_true_dist = []
        wt_false_dist = []
        rd1_false_dist = []
        distance_sets = [wt_true_distances, wt_false_distances, rd1_true_distances, rd1_false_distances]

        for distance_set in distance_sets:

            distance_diff_list = []
            path_length_list = []

            for distance_list in distance_set:

                total_distance_covered = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                distance_diff_list.append(total_distance_covered)

                # Calculate path length (absolute difference between start and end points)
                path_length = abs(distance_list[-1] - distance_list[0])
                path_length_list.append(path_length)

                if distance_set == wt_true_distances:
                    wt_true_dist = distance_diff_list
                elif distance_set == rd1_true_distances:
                    rd1_true_dist = distance_diff_list
                elif distance_set == wt_false_distances:
                    wt_false_dist = distance_diff_list
                elif distance_set == rd1_false_distances:
                    rd1_false_dist = distance_diff_list
                
        fig4, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig4)
        plots.plot_bar_two_groups(fig4,
                                  ax, 
                                    wt_dist_ratio,  
                                    rd1_dist_ratio, 
                                    "Mouse type", 
                                    "Total Distance Covered / Path Length (log)",
                                    "WT", 
                                    "rd1",
                                    color1='tab:blue',
                                    color2='mediumseagreen',
                                    ylim=None,
                                    bar_width=0.2,
                                    points=True,
                                    log_y=True,
                                    title=None,
                                    show=False,
                                    close=True)
        
        fig, ax = plt.subplots(figsize=(8,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                        wt_true_dist, 
                                        wt_false_dist, 
                                        rd1_true_dist, 
                                        rd1_false_dist, 
                                        ["WT - escape", "WT - no escape", "rd1 - escape", "rd1 - no escape"],
                                        "Mouse Type", 
                                        "Distance Covered After Stimulus",
                                        colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                        bar_width=0.35,
                                        log_y=False, 
                                        show=False, 
                                        close=True)


    def plot_prev_tort(self):
        wt_true_prev_esc, wt_false_prev_esc, rd1_true_prev_esc, rd1_false_prev_esc = data.extract_tort_data(self.prev_esc_locs_file, start_row=5)
        
        wt_true_tort = []
        wt_false_tort = []
        rd1_true_tort = []
        rd1_false_tort = []

        distance_sets = [wt_true_prev_esc, wt_false_prev_esc, rd1_true_prev_esc, rd1_false_prev_esc]
        
        for distance_set in distance_sets:
            
            tort_list = []

            for distance_list in distance_set:
                total_distance_difference = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                path_length = abs(distance_list[-1] - distance_list[0])
                dist_ratio = total_distance_difference / path_length
                tort_list.append(dist_ratio)
            
            if distance_set == wt_true_prev_esc:
                wt_true_tort = tort_list
            elif distance_set == wt_false_prev_esc:
                wt_false_tort = tort_list
            elif distance_set == rd1_true_prev_esc:
                rd1_true_tort = tort_list
            elif distance_set == rd1_false_prev_esc:
                rd1_false_tort = tort_list
        
        fig, ax = plt.subplots(figsize=(8,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                        wt_true_tort, 
                                        wt_false_tort, 
                                        rd1_true_tort, 
                                        rd1_false_tort, 
                                        ["WT - escape", "WT - no escape", "rd1 - escape", "rd1 - no escape"],
                                        "Mouse Type", 
                                        "Tortuosity of Path From Previous Escape (log)",
                                        colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                        bar_width=0.35,
                                        log_y=True, 
                                        show=False, 
                                        close=True)
        
    def plot_traj_data(self):
        wt_true_locs, rd1_true_locs = data.extract_escape_locs(self.event_locs_file, escape=True)
        wt_false_locs, rd1_false_locs = data.extract_escape_locs(self.event_locs_file, escape=False)

        norm_true_wt_locs = angles.normalize_length(wt_true_locs)
        norm_true_rd1_locs = angles.normalize_length(rd1_true_locs)
        norm_false_wt_locs = angles.normalize_length(wt_false_locs)
        norm_false_rd1_locs = angles.normalize_length(rd1_false_locs)

        rotated_true_wt_locs = []
        rotated_true_rd1_locs = []
        rotated_false_wt_locs = []
        rotated_false_rd1_locs = []
        stretched_true_wt_locs = []
        stretched_true_rd1_locs = []
        stretched_false_wt_locs = []
        stretched_false_rd1_locs = []

        exit_coord = (770, 370)
        min_wt_true_coord = None
        min_rd1_true_coord = None
        min_wt_false_coord = None
        min_rd1_false_coord = None

        for coord_set in norm_true_wt_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_true_wt_locs.append(coords)
            first_coord = coords[0]
            if min_wt_true_coord is None or first_coord[0] < min_wt_true_coord[0]:
                min_wt_true_coord = first_coord
            stretched_coords = stretch_traj(coords, min_wt_true_coord)
            stretched_true_wt_locs.append(stretched_coords)

        for coord_set in norm_true_rd1_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_true_rd1_locs.append(coords)
            first_coord = coords[0]
            if min_rd1_true_coord is None or first_coord[0] < min_rd1_true_coord[0]:
                min_rd1_true_coord = first_coord
            stretched_coords = stretch_traj(coords, min_rd1_true_coord)
            stretched_true_rd1_locs.append(stretched_coords)

        for coord_set in norm_false_wt_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_false_wt_locs.append(coords)
            first_coord = coords[0]
            if min_wt_false_coord is None or first_coord[0] < min_wt_false_coord[0]:
                min_wt_false_coord = first_coord
            stretched_coords = stretch_traj(coords, min_wt_false_coord)
            stretched_false_wt_locs.append(stretched_coords)

        for coord_set in norm_false_rd1_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_false_rd1_locs.append(coords)
            first_coord = coords[0]
            if min_rd1_false_coord is None or first_coord[0] < min_rd1_false_coord[0]:
                min_rd1_false_coord = first_coord
            stretched_coords = stretch_traj(coords, min_rd1_false_coord)
            stretched_false_rd1_locs.append(stretched_coords)

        fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(14,10))
        self.figs.append(fig)
        plots.time_plot(fig, ax1, norm_true_wt_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax2, norm_false_wt_locs, fps=30, show=False, close=True)
        plots.time_plot(fig, ax3, norm_true_rd1_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax4, norm_false_rd1_locs, fps=30, show=False, close=True)

        fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(14,10))
        self.figs.append(fig)
        plots.time_plot(fig, ax1, rotated_true_wt_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax2, rotated_false_wt_locs, fps=30, show=False, close=True)
        plots.time_plot(fig, ax3, rotated_true_rd1_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax4, rotated_false_rd1_locs, fps=30, show=False, close=True)

        fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(14,10))
        self.figs.append(fig)
        plots.time_plot(fig, ax1, stretched_true_wt_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax2, stretched_false_wt_locs, fps=30, show=False, close=True)
        plots.time_plot(fig, ax3, stretched_true_rd1_locs, fps=30, show=False, close=True, colorbar=False)
        plots.time_plot(fig, ax4, stretched_false_rd1_locs, fps=30, show=False, close=True)

    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder, "data")

def stretch_traj(coords, target_start_coord):
        
        target_start_x = target_start_coord[0]
        stretch_factor = abs(coords[-1][0] - target_start_x) / (coords[-1][0] - coords[0][0])
        stretched_coords = [(coord[0] * stretch_factor, coord[1]) for coord in coords]
        translation = stretched_coords[0][0] - target_start_x
        translated_coords = [((coord[0] - translation), coord[1]) for coord in stretched_coords]

        return translated_coords

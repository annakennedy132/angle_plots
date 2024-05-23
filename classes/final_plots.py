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

        self.global_figs = []
        self.event_figs = []
        self.avg_figs = []
        self.stats_figs = []
        self.tort_figs = []
        self.traj_figs = []
        self.save_figs = save_figs

        self.parent_folder = os.path.dirname(self.folder)
        self.global_file = next((os.path.join(self.parent_folder, file) for file in os.listdir(self.parent_folder) if file.endswith("global_data")), None)
        
        self.global_angles_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("locs.csv")), None)
        self.after_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("after_angles.csv")), None)
        self.during_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("during_angles.csv")), None)
        self.event_locs_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("locs.csv")), None)
        self.event_distances_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("event_distances.csv")), None)
        self.event_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("event_angles.csv")), None)
        self.avg_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("angles_avg.csv")), None)
        self.avg_dist_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("distances_avg.csv")), None)
        self.avg_angles_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("angles_avg_escape_success.csv")), None)
        self.avg_dist_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("distances_avg_escape_success.csv")), None)
        self.avg_speed_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("speeds_avg.csv")), None)
        self.avg_speed_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("speeds_avg_escape_success.csv")), None)
        self.stim_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("stim.csv")), None)
        self.escape_stats_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("escape-stats.csv")), None)
        self.escape_success_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("collated_escape_success.csv")), None)

    def plot_event_data(self):

        wt_after_angles, rd1_after_angles = data.extract_data_columns(self.after_angles_file, data_start=4, escape=False)
        wt_during_angles, rd1_during_angles = data.extract_data_columns(self.during_angles_file, data_start=4, escape=False)
        wt_locs, rd1_locs = data.extract_data_columns(self.event_locs_file, data_start=4, escape=False)

        wt_true_after_angles, wt_false_after_angles, rd1_true_after_angles, rd1_false_after_angles = data.extract_data_columns(self.after_angles_file, data_start=4, escape=True)
        wt_true_during_angles, wt_false_during_angles, rd1_true_during_angles, rd1_false_during_angles = data.extract_data_columns(self.during_angles_file, data_start=4, escape=True)
        wt_true_locs, wt_false_locs, rd1_true_locs, rd1_false_locs = data.extract_data_columns(self.event_locs_file, data_start=4, escape=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        plt.suptitle(f'Heatmaps of Coords Comparing WT and RD1 Mice During Stimulus Events')
        self.event_figs.append(fig)
        ax1.set_title('WT Mice')

        plots.plot_str_coords(fig, ax1, wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
        ax2.set_title('RD1 Mice')
        plots.plot_str_coords(fig, ax2, rd1_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
 
        #plot polar plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles of WT and RD1 Mice During Events / Time to Escape')
        self.event_figs.append(fig)
        ax1.set_title('WT Mice')

        plots.plot_str_polar_chart(fig, ax1, wt_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('RD1 Mice')
        plots.plot_str_polar_chart(fig, ax2, rd1_during_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles WT and RD1 Mice After Events / Return from Nest')
        self.event_figs.append(fig)
        ax1.set_title('WT Mice')

        plots.plot_str_polar_chart(fig, ax1, wt_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('RD1 Mice')
        plots.plot_str_polar_chart(fig, ax2, rd1_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles of WT and RD1 Mice During Events / Time to Escape')
        self.event_figs.append(fig)
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
        self.event_figs.append(fig)
        ax1.set_title('WT - escape')

        plots.plot_str_polar_chart(fig, ax1, wt_true_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('WT - no escape')
        plots.plot_str_polar_chart(fig, ax2, wt_false_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title('RD1 - escape')
        plots.plot_str_polar_chart(fig, ax3, rd1_true_after_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title('RD1 - no escape')
        plots.plot_str_polar_chart(fig, ax4, rd1_false_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        plt.suptitle(f'Heatmaps of Coords Comparing Escape Success / WT and RD1 Mice During Events')
        self.event_figs.append(fig)
        ax1.set_title('WT - escape')

        plots.plot_str_coords(fig, ax1, wt_true_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
        ax2.set_title('WT - no escape')
        plots.plot_str_coords(fig, ax2, wt_false_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
        ax3.set_title('RD1 - escape')
        plots.plot_str_coords(fig, ax3, rd1_true_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
        ax4.set_title('RD1 - no escape')
        plots.plot_str_coords(fig, ax4, rd1_false_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)

    def plot_avgs_data(self):
        
        stim_data = pd.read_csv(self.stim_file)["stim"].tolist()
        frame_time = (1./self.fps)
        norm_event_time = np.arange(-self.t_minus, (self.length + self.t_plus), frame_time)

        avg_data = [self.avg_angles_file, self.avg_dist_file, self.avg_speed_file]
        avg_data_esc = [self.avg_angles_esc_file, self.avg_dist_esc_file, self.avg_speed_esc_file]
        titles = ["Facing angle", "Distance from Exit", "Speed"]
        colours = ["tab:blue", "mediumseagreen", "tab:red"]
        data_limits = [(-185, 20), (0, 800), (0, 250)]
        
        for df, title, colour, limits in zip(avg_data, titles, colours, data_limits):
            df_data = pd.read_csv(df)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f"Average {title} at Stim Event")
            subtitles = ["WT", "RD1"]
            
            for column, ax, subtitle in zip(["wt", "rd1"], axes, subtitles):
                data = df_data[column].tolist()
                plots.two_plots(fig, ax, norm_event_time, data, stim_data, colour, "Time (seconds)", f"{title}", "Stimulus", limits, title=subtitle)
            
            self.avg_figs.append(fig)

        for df, title, colour, limits in zip(avg_data_esc, titles, colours, data_limits):
            df_data = pd.read_csv(df)

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"Average {title} at Stim Event")

            subtitles = ["WT - escape", "WT - no escape", "RD1 - escape", "RD1 - no escape"]

            for i, (column, subtitle) in enumerate(zip(["wt_true", "wt_false", "rd1_true", "rd1_false"], subtitles)):
                row = i // 2
                col = i % 2
                ax = axes[row, col]
                data = df_data[column].tolist()
                plots.two_plots(fig, ax, norm_event_time, data, stim_data, colour, "Time (seconds)", f"{title}", "Stimulus", limits, title=subtitle)

            self.avg_figs.append(fig)

    def plot_stats_data(self):
        wt_age, rd1_age = data.extract_data_rows(self.escape_stats_file, row=2, bool_row=3)
        wt_time, rd1_time = data.extract_data_rows(self.escape_stats_file, row=6, bool_row=3)
        wt_last_esc, rd1_last_esc = data.extract_data_rows(self.escape_stats_file, row=7, bool_row=3)
        wt_dist, rd1_dist = data.extract_data_rows(self.escape_stats_file, row=8, bool_row=3)
        wt_time_angle, rd1_time_angle = data.extract_data_rows(self.escape_stats_file, row=9, bool_row=3)
        wt_all_age, rd1_all_age = data.extract_data_rows(self.escape_success_file, row=2, bool_row=None)
        wt_esc_avg, rd1_esc_avg = data.extract_data_rows(self.escape_success_file, row=3, bool_row=None)
        wt_true_data, wt_false_data, rd1_true_data, rd1_false_data = data.extract_data_rows_esc(self.escape_stats_file, row=7)
        wt_true_data_angle, wt_false_data_angle, rd1_true_data_angle, rd1_false_data_angle = data.extract_data_rows_esc(self.escape_stats_file, row=9)

        wt_time = np.array(wt_time)
        rd1_time = np.array(rd1_time)
        wt_time[wt_time >= 15] = np.nan
        rd1_time[rd1_time >= 15] = np.nan

        fig, ax = plt.subplots()
        self.stats_figs.append(fig)
        plots.plot_bar_two_groups(fig, ax,
                                wt_esc_avg, 
                                rd1_esc_avg,
                                "Mouse Type", 
                                "Average Escape Success Rate (%)", 
                                "Average Escape Success in WT and RD1 Mice", 
                                "WT", 
                                "RD1", 
                                color1='tab:blue', 
                                color2='mediumseagreen',
                                show=False,
                                close=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.stats_figs.append(fig)
        plots.plot_scatter_trendline(fig, ax1,
                                        wt_all_age, 
                                        wt_esc_avg, 
                                        "Mouse Age", 
                                        "Average Escape Success Rate (%)",
                                        "WT",
                                        color='tab:blue', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)
        plots.plot_scatter_trendline(fig, ax2,
                                        rd1_all_age, 
                                        rd1_esc_avg, 
                                        "Mouse Age", 
                                        "Average Escape Success Rate (%)",
                                        "RD1",
                                        color='mediumseagreen', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.stats_figs.append(fig)
        bin_edges1 = np.arange(0, 5, 0.5)
        plots.plot_binned_bar_chart(fig, ax1,
                                    wt_all_age,
                                    wt_esc_avg, 
                                    bin_edges=bin_edges1, 
                                    xlabel="Mouse Age", 
                                    ylabel="Escape Success Rate (%)", 
                                    title="Mouse Age vs Escape Success in WT Mice", 
                                    color='tab:blue',
                                    show=False,
                                    close=True)
        bin_edges2 = np.arange(0, 9, 1)
        plots.plot_binned_bar_chart(fig, ax2,
                                    rd1_all_age,
                                    rd1_esc_avg, 
                                    bin_edges=bin_edges2, 
                                    xlabel="Mouse Age", 
                                    ylabel="Escape Success Rate (%)", 
                                    title="Mouse Age vs Escape Success in RD1 Mice", 
                                    color='mediumseagreen',
                                    show=False,
                                    close=True)

        fig, ax = plt.subplots()
        self.stats_figs.append(fig)
        plots.plot_two_scatter_trendline(fig, ax,
                                        wt_dist, 
                                        wt_time, 
                                        rd1_dist, 
                                        rd1_time, 
                                        "Distance From Exit at Stim", 
                                        "Time to Escape (s)", 
                                        "Effect of Distance From Door on Escape Time in WT and RD1 Mice", 
                                        "WT", 
                                        "RD1",
                                        group1_color='tab:blue', 
                                        group2_color='mediumseagreen', 
                                        marker_size=20, 
                                        show=False, 
                                        close=True)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
        self.stats_figs.append(fig)
        plots.plot_scatter_trendline(fig, ax1,
                                        wt_age, 
                                        wt_time,
                                        "Mouse Age", 
                                        "Time to Escape (s)",
                                        "WT",
                                        color='tab:blue',
                                        marker_size=20, 
                                        show=False,
                                        close=True)
        plots.plot_scatter_trendline(fig, ax2,
                                        rd1_age, 
                                        rd1_time,
                                        "Mouse Age", 
                                        "Time to Escape (s)",
                                        "RD1",
                                        color="mediumseagreen",
                                        marker_size=20, 
                                        show=False, 
                                        close=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.stats_figs.append(fig)
        bin_edges1 = np.arange(0, 9, 0.5)
        plots.plot_binned_bar_chart(fig, ax1,
                                    wt_age,
                                    wt_time, 
                                    bin_edges=bin_edges1, 
                                    xlabel="Mouse Age", 
                                    ylabel="Average Time to Escape (s)", 
                                    title="Mouse Age vs Escape Time in WT Mice", 
                                    color='tab:blue',
                                    y_limit=(0,15),
                                    show=False,
                                    close=True)
        bin_edges2 = np.arange(0, 9, 0.5)
        plots.plot_binned_bar_chart(fig, ax2,
                                    rd1_age,
                                    rd1_time, 
                                    bin_edges=bin_edges2, 
                                    xlabel="Mouse Age", 
                                    ylabel="Average Time to Escape (s)", 
                                    title="Mouse Age vs Escape Time in RD1 Mice", 
                                    color='mediumseagreen',
                                    y_limit=(0,15),
                                    show=False,
                                    close=True)
        
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
        self.stats_figs.append(fig)
        plots.plot_scatter_trendline(fig, ax1,
                                        wt_last_esc, 
                                        wt_time,
                                        "Time Since Last Escape (s)", 
                                        "Time to Escape at Stimulus (s)",
                                        "WT",
                                        color='tab:blue',
                                        marker_size=20, 
                                        show=False, 
                                        close=True)
        plots.plot_scatter_trendline(fig, ax2,
                                        rd1_last_esc, 
                                        rd1_time,
                                        "Time Since Last Escape (s)", 
                                        "Time to Escape at Stimulus (s)",
                                        "RD1",
                                        color="mediumseagreen",
                                        marker_size=20, 
                                        show=False, 
                                        close=True)
        
        fig, ax = plt.subplots()
        self.stats_figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_data, 
                                    wt_false_data, 
                                    rd1_true_data, 
                                    rd1_false_data, 
                                    ["WT - escape", "WT - no escape", "RD1 - escape", "RD1 - no escape"],
                                    "Mouse Type", 
                                    "Average Time Since Previous Escape", 
                                    "Effect of Time Since Previous Escape on Chance of Escape at Stimulus", 
                                    colors=['blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35, 
                                    show=False, 
                                    close=True)
        
        fig, ax = plt.subplots()
        self.stats_figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_time_angle, rd1_time_angle, "Mouse Type", "Average Time to Face Exit (s)", "Average Time to Face Exit After Stimulus in WT and RD1 Mice", "WT", "RD1",
                            color1='tab:blue', color2='mediumseagreen', ylim=(0,8), bar_width=0.5, show=False, close=True)
        
        fig, ax = plt.subplots()
        self.stats_figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_data_angle, 
                                    wt_false_data_angle, 
                                    rd1_true_data_angle, 
                                    rd1_false_data_angle, 
                                    ["WT - escape", "WT - no escape", "RD1 - escape", "RD1 - no escape"],
                                    "Mouse Type", 
                                    "Time to Face Exit (s)", 
                                    "Effect of Time to Face Exit on Chance of Escape at Stimulus", 
                                    colors=['mediumblue', 'blue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35, 
                                    show=False, 
                                    close=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.stats_figs.append(fig)
        bin_edges1 = np.arange(0, 9, 0.5)
        plots.plot_binned_bar_chart(fig, ax1,
                                    wt_age,
                                    wt_time_angle, 
                                    bin_edges=bin_edges1, 
                                    xlabel="Mouse Age", 
                                    ylabel="Average Time to Face Exit (s)", 
                                    title="Mouse Age vs Time to Face Exit After Stimulus in WT Mice", 
                                    color='tab:blue',
                                    y_limit=(0,15),
                                    show=False,
                                    close=True)
        bin_edges2 = np.arange(0, 9, 0.5)
        plots.plot_binned_bar_chart(fig, ax2,
                                    rd1_age,
                                    rd1_time_angle, 
                                    bin_edges=bin_edges2, 
                                    xlabel="Mouse Age", 
                                    ylabel="Average Time to Face Exit (s)", 
                                    title="Mouse Age vs Time to Face Exit After Stimulus in RD1 mice", 
                                    color='mediumseagreen',
                                    y_limit=(0,15),
                                    show=False,
                                    close=True)

    def plot_tort_data(self):
        wt_distances, rd1_distances = data.extract_escape_data(self.event_distances_file)
        wt_angles, rd1_angles = data.extract_escape_data(self.event_angles_file)

        wt_angle_difference = []
        rd1_angle_difference = []
        wt_distance_difference = []
        rd1_distance_difference = []
        wt_dist_ratio = []
        wt_angle_ratio = []
        rd1_dist_ratio = []
        rd1_angle_ratio = []

        angle_sets = [wt_angles, rd1_angles]
        distance_sets = [wt_distances, rd1_distances]

        for angle_set, distance_set in zip(angle_sets, distance_sets):

            angle_diff_list = []
            distance_diff_list = []
            path_length_list = []
            angle_change_list = []
            distance_ratio_list = []
            angle_ratio_list = []

            for angle_list, distance_list in zip(angle_set, distance_set):
                total_angle_difference = sum(abs(angle_list[i] - angle_list[i-1]) for i in range(1, len(angle_list)))
                angle_diff_list.append(total_angle_difference)

                total_distance_difference = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                distance_diff_list.append(total_distance_difference)

                # Calculate path length (absolute difference between start and end points)
                path_length = abs(distance_list[-1] - distance_list[0])
                path_length_list.append(path_length)

                # Calculate angle length (absolute difference between start and end angles)
                angle_length = abs(angle_list[-1] - angle_list[0])
                angle_change_list.append(angle_length)

                #calculate distance tortuosity
                dist_ratio = path_length / total_distance_difference
                distance_ratio_list.append(dist_ratio)

                #calculate angle tortuosity
                angle_ratio = total_angle_difference / angle_length
                angle_ratio_list.append(angle_ratio)

            if angle_set == wt_angles:
                wt_angle_difference = angle_diff_list
                wt_distance_difference = distance_diff_list
                wt_dist_ratio = distance_ratio_list
                wt_angle_ratio = angle_ratio_list
            elif angle_set == rd1_angles:
                rd1_angle_difference = angle_diff_list
                rd1_distance_difference = distance_diff_list
                rd1_dist_ratio = distance_ratio_list
                rd1_angle_ratio = angle_ratio_list

        wt_dist_ratio_avg = np.mean(wt_dist_ratio)
        wt_angle_ratio_avg =np.mean(wt_angle_ratio)
        rd1_dist_ratio_avg = np.mean(rd1_dist_ratio)
        rd1_angle_ratio_avg =  np.mean(rd1_angle_ratio)

        wt_ang_per_dist = wt_angle_ratio_avg / wt_dist_ratio_avg
        rd1_ang_per_dist = rd1_angle_ratio_avg / rd1_dist_ratio_avg

        fig, ax = plt.subplots()
        self.tort_figs.append(fig)
        plots.plot_two_lines(fig, ax, 
                            wt_angle_difference, 
                            wt_distance_difference, 
                            "Total Changes in Facing Angle", 
                            "Total Distance Covered", 
                            "tab:blue", 
                            "darkblue", 
                            xlabel='Escape', 
                            ylabel='Change', 
                            title='WT', 
                            ylim=(0,2500))
        
        fig, ax = plt.subplots()
        self.tort_figs.append(fig)
        plots.plot_two_lines(fig, ax, 
                            rd1_angle_difference, 
                            rd1_distance_difference, 
                            "Total changes in facing angle", 
                            "Total distance covered", 
                            "mediumseagreen", 
                            "seagreen", 
                            xlabel='Escape', 
                            ylabel='Change', 
                            title='RD1', 
                            ylim=(0,2500))
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.tort_figs.append(fig1)
        fig1.suptitle("Measure of tortuosity for escape paths - overall distance covered / total distance covered")
        plots.plot_one_line(fig1, ax1, wt_dist_ratio, "WT", "darkblue", xlabel='Escape', ylabel='Ratio', title='WT', ylim=(0, 2))
        plots.plot_one_line(fig1, ax2, rd1_dist_ratio, "RD1", "seagreen", xlabel='Escape', ylabel='Ratio', title='RD1', ylim=(0,2))

        fig3, ax = plt.subplots()
        self.tort_figs.append(fig3)
        plots.plot_bar_two_groups(fig3, ax, 
                                    wt_dist_ratio,  
                                    rd1_dist_ratio, 
                                    "Mouse type", 
                                    "Average - overall distance covered / total distance covered", 
                                    "Average tortuosity of escape paths", 
                                    "WT", 
                                    "RD1",
                                    color1='tab:blue', color2='mediumseagreen', ylim=(0,1), bar_width=0.5)

        fig4, ax = plt.subplots()
        self.tort_figs.append(fig4)
        plots.plot_bar_two_groups(fig, ax, wt_ang_per_dist, rd1_ang_per_dist, "Mouse type", "Average angles faced per distance travelled", 
                                "Average angles faced per distance travelled in WT and RD1 mice", "WT", "RD1",
                                color1='tab:blue', color2='mediumseagreen', ylim=None, bar_width=0.5)

    def plot_traj_data(self):
        wt_locs, rd1_locs = data.extract_escape_locs(self.event_locs_file)

        norm_wt_locs = data.normalize_length(wt_locs)
        norm_rd1_locs = data.normalize_length(rd1_locs)

        fig1, ax1 = plt.subplots(figsize=(10,8))
        fig1.suptitle("All escape paths of WT mice")
        self.traj_figs.append(fig1)
        plots.time_plot(fig1, ax1, norm_wt_locs, fps=30, show=False, close=False)
        fig2, ax2 = plt.subplots(figsize=(10,8))
        fig2.suptitle("All escape paths of RD1 mice")
        self.traj_figs.append(fig2)
        plots.time_plot(fig2, ax2, norm_rd1_locs, fps=30, show=True, close=False)

        rotated_wt_coords = []
        rotated_rd1_coords = []
        stretched_wt_coords = []
        stretched_rd1_coords = []

        exit_coord = (770, 370)

        for coord_set in norm_wt_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_wt_coords.append(coords)

        for coord_set in norm_rd1_locs:
            coords = angles.get_rotated_coords(exit_coord, coord_set)
            rotated_rd1_coords.append(coords)

        fig3, ax3 = plt.subplots(figsize=(10,8))
        fig3.suptitle("Normalised escape paths of WT mice")
        self.traj_figs.append(fig3)
        plots.time_plot(fig1, ax3, rotated_wt_coords, fps=30, show=False, close=False)
        fig4, ax4 = plt.subplots(figsize=(10,8))
        fig4.suptitle("Normalised escape paths of RD1 mice")
        self.traj_figs.append(fig4)
        plots.time_plot(fig4, ax4, rotated_rd1_coords, fps=30, show=True, close=False)

        min_wt_coord = None
        min_rd1_coord = None

        for coord_list in rotated_wt_coords:
            first_coord = coord_list[0]
            if min_wt_coord is None or first_coord[0] < min_wt_coord[0]:
                min_wt_coord = first_coord

        for coord_list in rotated_wt_coords:
            stretched_coords = stretch_traj(coord_list, min_wt_coord)
            stretched_wt_coords.append(stretched_coords)
        
        for coord_list in rotated_rd1_coords:
            first_coord = coord_list[0]
            if min_rd1_coord is None or first_coord[0] < min_rd1_coord[0]:
                min_rd1_coord = first_coord

        for coord_list in rotated_rd1_coords:
            stretched_coords = stretch_traj(coord_list, min_rd1_coord)
            stretched_rd1_coords.append(stretched_coords)

        fig5, ax5 = plt.subplots(figsize=(10,8))
        fig5.suptitle("Normalised escape paths of WT mice")
        self.traj_figs.append(fig5)
        plots.time_plot(fig1, ax5, stretched_wt_coords, fps=30, show=False, close=False)
        fig6, ax6 = plt.subplots(figsize=(10,8))
        fig6.suptitle("Normalised escape paths of RD1 mice")
        self.traj_figs.append(fig6)
        plots.time_plot(fig6, ax6, stretched_rd1_coords, fps=30, show=True, close=False)

    def plot_global_data(self):

        global_wt_angles, global_rd1_angles = data.extract_data_columns(self.global_angles_file, data_start=3)
        global_wt_locs, global_rd1_locs = data.extract_data_columns(self.global_locs_file, data_start=3)
        wt_baseline_locs, rd1_baseline_locs = data.extract_data_columns(self.global_locs_file, data_start=3, data_end=5400)
        wt_baseline_angles, rd1_baseline_angles = data.extract_data_columns(self.global_angles_file, data_start=3, data_end=5400)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plt.suptitle(f'Heatmaps of All Coords Comparing WT and RD1 Mice (including events)')
        self.global_figs.append(fig)

        for ax, title, value in zip(axes, ['WT Mice', 'RD1 Mice'], [global_wt_locs, global_rd1_locs]):
            ax.set_title(title)

            plots.plot_str_coords(fig, ax, value, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=2000, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        plt.suptitle(f'Heatmaps of Coords Comparing WT and RD1 Mice (baseline - first 3 minutes)')
        self.global_figs.append(fig)

        for ax, title, value in zip(axes, ['WT Mice', 'RD1 Mice'], [wt_baseline_locs, rd1_baseline_locs]):
            ax.set_title(title)

            plots.plot_str_coords(fig, ax, value, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=720, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing All Facing Angles WT and RD1 Mice')
        self.global_figs.append(fig)

        for ax, title, value in zip(axes, ['WT Mice', 'RD1 Mice'], [global_wt_angles, global_rd1_angles]):
            ax.set_title(title)

            plots.plot_str_polar_chart(fig, ax, value, bins=36, direction=1, zero="E", show=False, close=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw=dict(projection='polar'))
        fig.suptitle(f'Polar Plot Comparing Facing Angles WT and RD1 Mice (baseline - first 3 minutes)')
        self.global_figs.append(fig)

        for ax, title, value in zip(axes, ['WT Mice', 'RD1 Mice'], [ wt_baseline_angles, rd1_baseline_angles]):
            ax.set_title(title)

            plots.plot_str_polar_chart(fig, ax, value, bins=36, direction=1, zero="E", show=False, close=True)


    def save_pdfs(self):
        if self.save_figs:
            if self.global_figs:
                files.save_report(self.global_figs, self.folder, "collated_global_data")
            if len(self.event_figs) > 0:
                files.save_report(self.event_figs, self.folder, "event_data")
            if len(self.avg_figs) > 0:
                files.save_report(self.avg_figs, self.folder, "event_averages")
            if len(self.stats_figs) > 0:
                files.save_report(self.stats_figs, self.folder, "stats_data")
            if len(self.tort_figs) > 0:
                files.save_report(self.tort_figs, self.folder, "tortuosity")

def stretch_traj(coords, target_start_coord):
    # Extract the target start x-coordinate
    target_start_x = target_start_coord[0]
    
    # Calculate the stretching factor
    stretch_factor = abs(coords[-1][0] - target_start_x) / (coords[-1][0] - coords[0][0])
    
    # Stretch the x-coordinates
    stretched_coords = [(coord[0] * stretch_factor, coord[1]) for coord in coords]

    #translate back to the correct starting point
    translation = stretched_coords[0][0] - target_start_x
    translated_coords = [((coord[0] - translation), coord[1]) for coord in stretched_coords]

    return translated_coords


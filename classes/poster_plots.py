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
        self.avg_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("angles_avg.csv")), None)
        self.avg_dist_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("distances_avg.csv")), None)
        self.avg_angles_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("angles_avg_escape_success.csv")), None)
        self.avg_dist_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("distances_avg_escape_success.csv")), None)
        self.avg_speed_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("speeds_avg.csv")), None)
        self.avg_speed_esc_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("speeds_avg_escape_success.csv")), None)
        self.stim_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("stim.csv")), None)
        self.escape_stats_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("escape-stats.csv")), None)
        self.escape_success_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("collated_escape_success.csv")), None)

    def plot_global_data(self):

        wt_baseline_locs, rd1_baseline_locs = data.extract_data_columns(self.global_locs_file, data_start=3, data_end=5400)
        self.wt_baseline_angles, self.rd1_baseline_angles = data.extract_data_columns(self.global_angles_file, data_start=3, data_end=5400)
        wt_locs, rd1_locs = data.extract_data_columns(self.event_locs_file, data_start=4, escape=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)

        for ax, title, value in zip(axes, ['WT Mice - baseline', 'RD1 Mice - baseline'], [wt_baseline_locs, rd1_baseline_locs]):
            ax.set_title(title)
            plots.plot_str_coords(fig, ax, value, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=720, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        self.figs.append(fig)
        ax1.set_title('WT Mice - events')
        plots.plot_str_coords(fig, ax1, wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
        ax2.set_title('RD1 Mice events')
        plots.plot_str_coords(fig, ax2, rd1_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, colorbar=True, show=False, close=True)
         

    def plot_event_data(self):

        wt_before_angles, rd1_before_angles = data.extract_data_columns(self.event_angles_file, data_start=4, data_end=154, escape=False)
        wt_after_angles, rd1_after_angles = data.extract_data_columns(self.after_angles_file, data_start=4, escape=False)
        wt_during_angles, rd1_during_angles = data.extract_data_columns(self.during_angles_file, data_start=4, escape=False)

        #plot polar plots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        plt.suptitle(f"Polar Plots Comparing Facing Angles of WT Mice at Baseline, Before, During and After Events")
        self.event_figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_str_polar_chart(fig, ax1, self.wt_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_str_polar_chart(fig, ax2, wt_before_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_str_polar_chart(fig, ax3, wt_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title("After Stimulus / Return from Nest")
        plots.plot_str_polar_chart(fig, ax4, wt_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5), subplot_kw=dict(projection='polar'))
        plt.suptitle(f"Polar Plots Comparing Facing Angles of RD1 Mice at Baseline, Before, During and After Events")
        self.event_figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_str_polar_chart(fig, ax1, self.rd1_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_str_polar_chart(fig, ax2, rd1_before_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_str_polar_chart(fig, ax3, rd1_during_angles, bins=36, direction=1, zero="E", show=False, close=True)
        ax4.set_title("After Stimulus / Return from Nest")
        plots.plot_str_polar_chart(fig, ax4, rd1_after_angles, bins=36, direction=1, zero="E", show=False, close=True)

    def plot_stats_data(self):
        wt_time, rd1_time = data.extract_data_rows(self.escape_stats_file, row=6, bool_row=3)
        wt_dist, rd1_dist = data.extract_data_rows(self.escape_stats_file, row=8, bool_row=3)
        wt_esc_avg, rd1_esc_avg = data.extract_data_rows(self.escape_success_file, row=3, bool_row=None)
        wt_true_data, wt_false_data, rd1_true_data, rd1_false_data = data.extract_data_rows_esc(self.escape_stats_file, row=7)
        wt_true_locs, wt_false_locs, rd1_true_locs, rd1_false_locs = data.extract_data_rows_esc(self.escape_stats_file, row=4)

        wt_time = np.array(wt_time)
        rd1_time = np.array(rd1_time)
        wt_time[wt_time >= 15] = np.nan
        rd1_time[rd1_time >= 15] = np.nan

        fig, ax = plt.subplots(figsize=(5,5))
        self.figs.append(fig)
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
                                bar_width=0.2,
                                points=True,
                                show=False,
                                close=True)

        fig, ax = plt.subplots()
        self.figs.append(fig)
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

        fig, ax = plt.subplots()
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_data, 
                                    wt_false_data, 
                                    rd1_true_data, 
                                    rd1_false_data, 
                                    ["WT - escape", "WT - no escape", "RD1 - escape", "RD1 - no escape"],
                                    "Mouse Type", 
                                    "Average Time Since Previous Escape", 
                                    "Effect of Time Since Previous Escape on Chance of Escape at Stimulus", 
                                    colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35, 
                                    show=False, 
                                    close=True)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.figs.append(fig)
        ax1.set_title('WT Mice - escape')
        plots.scatter_plot_with_stats(fig, ax1, wt_true_locs, point_color='tab:blue', background_color='black', mean_marker='*', x_limits=(80,790), y_limits=(670,70), show=True, close=True)
        ax2.set_title('WT Mice - no escape')
        plots.scatter_plot_with_stats(fig, ax2, wt_false_locs, point_color='mediumblue', background_color='black', mean_marker='*', x_limits=(80,790), y_limits=(670,70), show=True, close=True)
        ax3.set_title('RD1 Mice - escape')
        plots.scatter_plot_with_stats(fig, ax3, rd1_true_locs, point_color='green', background_color='black', mean_marker='*', x_limits=(80,790), y_limits=(670,70), show=True, close=True)
        ax4.set_title('RD1 Mice - no escape')
        plots.scatter_plot_with_stats(fig, ax4, rd1_false_locs, point_color='mediumseagreen', background_color='black', mean_marker='*', x_limits=(80,790), y_limits=(670,70), show=True, close=True)

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

                total_distance_difference = sum(abs(distance_list[i] - distance_list[i-1]) for i in range(1, len(distance_list)))
                distance_diff_list.append(total_distance_difference)

                # Calculate path length (absolute difference between start and end points)
                path_length = abs(distance_list[-1] - distance_list[0])
                path_length_list.append(path_length)

                #calculate distance tortuosity
                dist_ratio = total_distance_difference / path_length
                distance_ratio_list.append(dist_ratio)

        fig, ax = plt.subplots()
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, 
                                    wt_dist_ratio,  
                                    rd1_dist_ratio, 
                                    "Mouse type", 
                                    "Total Distance Covered / Path Length", 
                                    "Tortuosity of escape paths", 
                                    "WT", 
                                    "RD1",
                                    color1='tab:blue', color2='mediumseagreen', ylim=None, bar_width=0.5, points=True)
        
        fig4, ax = plt.subplots(figsize=(5,5))
        self.figs.append(fig4)
        plots.plot_bar_two_groups(fig4, ax, 
                                    wt_dist_ratio,  
                                    rd1_dist_ratio, 
                                    "Mouse type", 
                                    "Total Distance Covered / Path Length", 
                                    "Tortuosity of escape paths", 
                                    "WT", 
                                    "RD1",
                                    color1='tab:blue', color2='mediumseagreen', ylim=None, bar_width=0.2, points=False)

    def plot_prev_tort(self):
        wt_true_prev_esc, wt_false_prev_esc, rd1_true_prev_esc, rd1_false_prev_esc = data.extract_tort_data(self.prev_esc_locs_file)
        
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
                                        ["WT - escape", "WT - no escape", "RD1 - escape", "RD1 - no escape"],
                                        "Mouse Type", 
                                        "Tortuosity of Path From Previous Escape to Stimulus", 
                                        "Effect of Tortuosity of Path Since Previous Escape on Chance of Escape at Stimulus", 
                                        colors=['blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                        bar_width=0.35,
                                        log_y=True, 
                                        show=False, 
                                        close=True)
        
    def plot_traj_data(self):
        wt_true_locs, rd1_true_locs = data.extract_escape_locs(self.event_locs_file, escape=True)

        norm_true_wt_locs = data.normalize_length(wt_true_locs)
        norm_true_rd1_locs = data.normalize_length(rd1_true_locs)

        fig1, ax1 = plt.subplots(figsize=(10,8))
        fig1.suptitle("Paths of WT mice - escape")
        self.figs.append(fig1)
        plots.time_plot(fig1, ax1, norm_true_wt_locs, fps=30, show=False, close=False)
        fig2, ax2 = plt.subplots(figsize=(10,8))
        fig2.suptitle("Paths of RD1 mice - escape")
        self.figs.append(fig2)
        plots.time_plot(fig2, ax2, norm_true_rd1_locs, fps=30, show=True, close=False)

        wt_false_locs, rd1_false_locs = data.extract_escape_locs(self.event_locs_file, escape=False)

        norm_false_wt_locs = data.normalize_length(wt_false_locs)
        norm_false_rd1_locs = data.normalize_length(rd1_false_locs)

        fig1, ax1 = plt.subplots(figsize=(10,8))
        fig1.suptitle("All paths of WT mice - no escape")
        self.figs.append(fig1)
        plots.time_plot(fig1, ax1, norm_false_wt_locs, fps=30, show=False, close=False)
        fig2, ax2 = plt.subplots(figsize=(10,8))
        fig2.suptitle("All paths of RD1 mice - no escape")
        self.figs.append(fig2)
        plots.time_plot(fig2, ax2, norm_false_rd1_locs, fps=30, show=True, close=False)

    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder, "pdf_data")


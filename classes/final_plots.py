import os
import numpy as np
import matplotlib.pyplot as plt
from utils import files, utils
from processing import plots, data, angles, stats
import matplotlib.image as mpimg
import seaborn as sns

class FinalPlots:
    def __init__(self, folder, mouse_type, save_figs=True, save_imgs=True):
        self.fps = 30
        self.length = 10
        self.t_minus = 5
        self.t_plus = 5
        
        self.mouse_type = mouse_type

        self.folder = folder
        image_file = "images/arena.tif"
        self.background_image = mpimg.imread(image_file)

        self.figs = []
        self.imgs = []
        self.save_figs = save_figs
        self.save_imgs = save_imgs

        self.parent_folder = os.path.dirname(self.folder)
        self.global_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_data")), None)
        
        self.global_angles_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("locs.csv")), None)
        self.global_speeds_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("speeds.csv")), None)
        self.global_distances_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("distances.csv")), None)
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

        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, nested=False, data_start=3, data_end=None, process_coords=True, escape_col=None)
        event_wt_locs, event_blind_locs = data.extract_data(self.event_locs_file, nested=False, data_start=154, escape=False, process_coords=True, escape_col=None)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.figs.append(fig)
        plots.plot_coords(fig, ax1, wt_baseline_locs, xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=1000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        ax1.set_title(("WT - baseline"))
        plots.plot_coords(fig, ax2, blind_baseline_locs, xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=1000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
        ax2.set_title((f"{self.mouse_type} - baseline"))
        plots.plot_coords(fig, ax3, event_wt_locs, xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True, colorbar=False)
        ax3.set_title(("WT - events"))
        plots.plot_coords(fig, ax4, event_blind_locs, xlabel="x", ylabel="y", gridsize=100, vmin=0, vmax=80, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
        ax4.set_title((f"{self.mouse_type} - events"))

    def plot_angle_data(self):
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, nested=False, data_start=3, data_end=5400, escape_col=None)
        wt_before_angles, blind_before_angles = data.extract_data(self.event_angles_file, nested=False, data_start=4, data_end=154)
        wt_after_angles, blind_after_angles = data.extract_data(self.after_angles_file, nested=False, data_start=4)
        wt_during_angles, blind_during_angles = data.extract_data(self.during_angles_file, nested=False, data_start=4)
        
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(projection='polar'))
        self.figs.append(fig)
        ax1.set_title('Baseline - first 3 minutes')
        plots.plot_polar_chart(fig, ax1, wt_baseline_angles, bins=36, show=False, close=True)
        ax2.set_title('Before Stimulus')
        plots.plot_polar_chart(fig, ax2, wt_before_angles, bins=36, show=False, close=True)
        ax3.set_title("During Stimulus / Time to Escape")
        plots.plot_polar_chart(fig, ax3, wt_during_angles, bins=36, show=False, close=True)
        ax4.set_title("After Stimulus / Exit from Nest")
        plots.plot_polar_chart(fig, ax4, wt_after_angles, bins=36, show=False, close=True)
        plots.plot_polar_chart(fig, ax5, blind_baseline_angles, bins=36, direction=1, zero="E", show=False, close=True)
        plots.plot_polar_chart(fig, ax6, blind_before_angles, bins=36, show=False, close=True)
        plots.plot_polar_chart(fig, ax7, blind_during_angles, bins=36, show=False, close=True)
        plots.plot_polar_chart(fig, ax8, blind_after_angles, bins=36, show=False, close=True)

    def plot_avgs_data(self):

        frame_time = (1./self.fps)
        norm_event_time = np.arange(-self.t_minus, (self.length + self.t_plus), frame_time)

        wt_speeds, blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=None)
        wt_angles, blind_angles = data.extract_data(self.event_angles_file, data_start=4, data_end=None)
        wt_dist, blind_dist = data.extract_data(self.event_distances_file, data_start=4, data_end=None)
        wt_true_angles, wt_false_angles, blind_true_angles, blind_false_angles = data.extract_data(self.event_angles_file, data_start=4, escape=True, escape_col=3)
        wt_true_dist, wt_false_dist, blind_true_dist, blind_false_dist = data.extract_data(self.event_distances_file, data_start=4, escape=True, escape_col=3)
        wt_true_speeds, wt_false_speeds, blind_true_speeds, blind_false_speeds = data.extract_data(self.event_speeds_file, data_start=4, escape=True, escape_col=3)
        
        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_speeds]) for i in range(max(map(len, wt_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_speeds]) for i in range(max(map(len, blind_speeds)))]]
        avg_angles_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_angles]) for i in range(max(map(len, wt_angles)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_angles]) for i in range(max(map(len, blind_angles)))]]
        avg_dist_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_dist]) for i in range(max(map(len, wt_dist)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_dist]) for i in range(max(map(len, blind_dist)))]]

        fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,5))
        self.figs.append(fig)

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Speed (pps)")
        ax1.set_ylim(0,250)
        ax1.plot(norm_event_time, avg_speeds_data[0], color="red", label="WT")
        ax1.plot(norm_event_time, avg_speeds_data[1], color="red", alpha=0.5, label=f"{self.mouse_type}")
        ax1.tick_params(axis='y')
        ax1.axvspan(0,10, alpha=0.3, color='lightgray')
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.legend()

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Facing angle")
        ax2.set_ylim(-185,20)
        ax2.plot(norm_event_time, avg_angles_data[0], color="blue", label="WT")
        ax2.plot(norm_event_time, avg_angles_data[1], color="blue", alpha=0.5, label=f"{self.mouse_type}")
        ax2.tick_params(axis='y')
        ax2.axvspan(0,10, alpha=0.3, color='lightgray')
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.legend()
        
        ax3.set_xlabel("Time (seconds)")
        ax3.set_ylabel("Distance from Exit (cm)")
        ax3.set_ylim(0,55)
        ax3.plot(norm_event_time, avg_dist_data[0], color="green", label="WT")
        ax3.plot(norm_event_time, avg_dist_data[1], color="green", alpha=0.5, label=f"{self.mouse_type}")
        ax3.tick_params(axis='y')
        ax3.axvspan(0,10, alpha=0.3, color='lightgray')
        ax3.spines['right'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.legend()

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
        colours = ["blue", "green", "red"]
        data_limits = [(-185, 20), (0, 55), (0,255)]
        
        for df, title, colour, limits in zip([avg_esc_angle_data, avg_esc_dist_data, avg_esc_speeds_data], titles, colours, data_limits):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            fig.suptitle(f"Average {title} at Stim Event")
            self.figs.append(fig)
            
            ax1.set_xlabel("Time (seconds)")
            ax1.set_ylabel("Speed (pps)")
            ax1.set_title("WT")
            ax1.set_ylim(limits)
            ax1.plot(norm_event_time, df[0], color=colour, label="escape")
            ax1.plot(norm_event_time, df[1], color=colour, alpha=0.5, label="no escape")
            ax1.tick_params(axis='y')
            ax1.axvspan(0,10, alpha=0.3, color='lightgray')
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            ax1.legend()
            
            ax2.set_xlabel("Time (seconds)")
            ax2.set_ylabel("Speed (pps)")
            ax2.set_title(f"{self.mouse_type}")
            ax2.set_ylim(limits)
            ax2.plot(norm_event_time, df[2], color=colour, label="escape")
            ax2.plot(norm_event_time, df[3], color=colour, alpha=0.5, label="no escape")
            ax2.tick_params(axis='y')
            ax2.axvspan(0,10, alpha=0.3, color='lightgray')
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax2.legend()
            
    def plot_stats_data(self):
        wt_time, blind_time = data.extract_data_rows(self.escape_stats_file, data_row=6)
        wt_dist, blind_dist = data.extract_data_rows(self.escape_stats_file, data_row=8)
        wt_esc_avg, blind_esc_avg = data.extract_data_rows(self.escape_success_file, data_row=3)
        wt_true_data, wt_false_data, blind_true_data, blind_false_data = data.extract_data_rows(self.escape_stats_file, data_row=7, escape=True)
        wt_age, blind_age = data.extract_data_rows(self.escape_success_file, data_row=2)
        wt_time_angle, blind_time_angle = data.extract_data_rows(self.escape_stats_file, data_row=9)
        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data_rows(self.escape_stats_file, data_row=4, escape=True)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, process_coords=True)
        
        wt_time = np.array(wt_time)
        blind_time = np.array(blind_time)
        wt_time[wt_time >= 15] = np.nan
        blind_time[blind_time >= 15] = np.nan

        #escape probability, time to escape, time to face exit
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6,3.5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax1, wt_esc_avg, blind_esc_avg, "Mouse Type", "Escape probability (%)", "WT", f"{self.mouse_type}",
                                  color1='teal', color2='gainsboro', ylim=(0,102), bar_width=0.2, points=True, error_bars=False, show=False, close=True)
        
        plots.plot_bar_two_groups(fig, ax2, wt_time, blind_time, "Mouse Type", "Time to Escape (s)", "WT", f"{self.mouse_type}", 
                                color1='teal', color2='gainsboro', ylim=None, bar_width=0.2, points=False, error_bars=True, show=False, close=True)

        plots.plot_bar_two_groups(fig, ax3, wt_time_angle, blind_time_angle, "Mouse Type", "Average Time to Face Exit (s)", "WT", f"{self.mouse_type}",
                                color1='teal', color2='gainsboro', ylim=(0,8), bar_width=0.2, points=False, error_bars=True, show=False, close=True)
        
        #age vs escape success
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        self.figs.append(fig)
        ax1.set_xlabel("Mouse Age")
        ax1.set_ylabel("Escape Sucess")
        ax1.set_title("WT")
        sns.regplot(x=wt_age, y=wt_esc_avg, ax=ax1, scatter=True,
                    scatter_kws={'color': "teal", 'alpha': 0.7, 's': 10}, line_kws={'color': "teal"}, ci=None)
        ax2.set_xlabel("Mouse Age")
        ax2.set_ylabel("Escape Sucess")
        ax2.set_title(f"{self.mouse_type}")
        sns.regplot(x=blind_age, y=blind_esc_avg, ax=ax2, scatter=True,
                    scatter_kws={'color': "lightgray", 'alpha': 0.7, 's': 10}, line_kws={'color': "lightgray"}, ci=None)

        # distance from exit
        fig, ax = plt.subplots()
        self.figs.append(fig)
        ax.set_xlabel("Distance from Exit at Stimulus (cm)")
        ax.set_ylabel("Time to Escape (s)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        sns.regplot(x=wt_dist, y=wt_time, ax=ax, scatter=True,
                    scatter_kws={'color': "teal", 'alpha': 0.7, 's': 10}, line_kws={'color': "teal"})
        sns.regplot(x=blind_dist, y=blind_time, ax=ax, scatter=True,
                    scatter_kws={'color': "lightgray", 'alpha': 0.7, 's': 10}, line_kws={'color': "lightgray"})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        self.figs.append(fig)
        x_limits = (0, 850)
        y_limits = (755, 0)
        
        ax1.set_title('WT Mice - escape')
        ax1.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax1, wt_true_locs, point_color='tab:blue', mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax2.set_title('WT Mice - no escape')
        ax2.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax2, wt_false_locs, point_color='tab:blue', mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax3.set_title(f'{self.mouse_type} Mice - escape')
        ax3.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax3, blind_true_locs, point_color='tab:blue', mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        ax4.set_title(f'{self.mouse_type} Mice - no escape')
        ax4.imshow(self.background_image, cmap="gray", extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.scatter_plot_with_stats(fig, ax4, blind_false_locs, point_color='tab:blue', mean_marker='o', x_limits=x_limits, y_limits=y_limits, show=False, close=True)
        plt.tight_layout()

        fig, ax = plt.subplots(figsize=(4,4))
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Time Since Previous Escape (s)")
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, wt_true_data, wt_false_data, blind_true_data, blind_false_data, 
                                     xticks=["WT", f"{self.mouse_type}"], labels=["escape", "no escape", "escape", "no escape"],
                                     colors=["teal", "cadetblue", "darkgray", "lightgray"],
                                    bar_width=0.1, error_bars=True, points=False, show=False, close=True)
        
        wt_times_to_find_nest = []
        blind_times_to_find_nest = []

        exit_roi = [650, 240, 800, 500]
        
        for loc_set in wt_baseline_locs:
            time_to_find_nest = stats.find_global_esc_frame(loc_set, min_escape_frames=5, exit_roi=exit_roi)
            wt_times_to_find_nest.append(time_to_find_nest)

        for loc_set in blind_baseline_locs:
            time_to_find_nest = stats.find_global_esc_frame(loc_set, min_escape_frames=5, exit_roi=exit_roi)
            blind_times_to_find_nest.append(time_to_find_nest)
        
        wt_count = wt_times_to_find_nest.count(None)
        blind_count = blind_times_to_find_nest.count(None)
        wt_percentage_useless = wt_count / len(wt_times_to_find_nest) * 100
        blind_percentage_useless = blind_count / len(blind_times_to_find_nest) * 100

        wt_times_to_find_nest = [x for x in wt_times_to_find_nest if x is not None]
        blind_times_to_find_nest = [x for x in blind_times_to_find_nest if x is not None]

        fig, ax = plt.subplots(figsize=(3,4))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_times_to_find_nest, blind_times_to_find_nest, x_label="mouse type",
                                   y_label="time to find escape (s)", bar1_label="wt", bar2_label="rd1",
                        color1='blue', color2='green', ylim=None, bar_width=0.2, points=True, 
                        log_y=False, error_bars=False, show=False, close=True, title=None)
        
        fig, ax = plt.subplots(figsize=(3,4))
        self.figs.append(fig)
        
        plots.plot_bar_two_groups(fig, ax, wt_percentage_useless, blind_percentage_useless, x_label="Mouse Type", y_label="% Mice that Never Escape", 
                                  bar1_label="WT", bar2_label=f"{self.mouse_type}", color1="blue", color2="mediumseagreen", ylim=None, bar_width=0.2, points=True, 
                                log_y=False, error_bars=False, show=False, close=True, title=None)
        ax.set_title("Percentage of Mice \n that Never Escape")

    def plot_traj_data(self):
        wt_true_locs, blind_true_locs, wt_false_locs, blind_false_locs = data.extract_data(self.event_locs_file, escape=True, process_coords=True, get_escape_index=True, escape_col=3)

        norm_true_wt_locs = angles.normalize_length(wt_true_locs)
        norm_true_blind_locs = angles.normalize_length(blind_true_locs)
        norm_false_wt_locs = angles.normalize_length(wt_false_locs)
        norm_false_blind_locs = angles.normalize_length(blind_false_locs)

        x_limits = (0, 850)
        y_limits = (755, 0)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(14,10))
        self.imgs.append(fig)
        ax1.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax1, norm_true_wt_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax2.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax2, norm_false_wt_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax3.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax3, norm_true_blind_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)
        ax4.imshow(self.background_image, cmap='gray', extent=[*x_limits, *y_limits], aspect='auto', zorder=0)
        plots.time_plot(fig, ax4, norm_false_blind_locs, fps=30, xlim=x_limits, ylim=y_limits, show=False, close=True, colorbar=False)

    def plot_tort_data(self):
        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data(self.event_locs_file, data_start=154, escape=True, process_coords=True, get_escape_index=True, escape_col=3)
        wt_true_prev_esc, wt_false_prev_esc, blind_true_prev_esc, blind_false_prev_esc = data.extract_data(self.prev_esc_locs_file, data_start=4, escape=True, escape_col=3)
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
                distance_list = [distance for distance in distance_list if not np.isnan(distance).any()]
                if len(distance_list) < 2:
                    continue

                total_distance_covered = sum(
                    utils.calc_dist_between_points(distance_list[i], distance_list[i - 1])
                    for i in range(1, len(distance_list)))
                distance_diff_list.append(total_distance_covered)

                path_length = utils.calc_dist_between_points(distance_list[-1], distance_list[0])
                path_length_list.append(path_length)
                
                if path_length >= min_path_length:
                    dist_ratio = total_distance_covered / path_length
                    distance_ratio_list.append(dist_ratio)

            if distance_set == wt_true_locs:
                wt_true_dist_ratio = distance_ratio_list
            elif distance_set == wt_false_locs:
                wt_false_dist_ratio = distance_ratio_list
            elif distance_set == blind_true_locs:
                blind_true_dist_ratio = distance_ratio_list
            elif distance_set == blind_false_locs:
                blind_false_dist_ratio = distance_ratio_list

        fig, ax = plt.subplots(figsize=(3,4))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_true_dist_ratio, blind_true_dist_ratio,
            "Mouse type", "Tortuosity of Escape Path (log)","WT", f"{self.mouse_type}",color1='teal', color2='lightgray',
            bar_width=0.2, error_bars=True, points=False, log_y=True, ylim=None, title=None, show=False, close=True)
        ax.set_ylabel("Tortuosity of Escape")
        
        fig, ax = plt.subplots(figsize=(5,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, wt_true_dist_ratio, wt_false_dist_ratio, blind_true_dist_ratio, blind_false_dist_ratio,
            ["WT", f"{self.mouse_type}"], labels=["escape", "no escape", "escape", "no escape"], colors=["teal", "cadetblue", "darkgray", "lightgray"], 
            bar_width=0.1, error_bars=True, log_y=True, ylim=None, show=False, close=True)
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Tortuosity of Escape Path")

        wt_true_tort = []
        wt_false_tort = []
        blind_true_tort = []
        blind_false_tort = []
        prev_distance_sets = [wt_true_prev_esc, wt_false_prev_esc, blind_true_prev_esc, blind_false_prev_esc]
        
        for distance_set in prev_distance_sets:
            
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
                
        fig, ax = plt.subplots(figsize=(5,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, wt_true_tort, wt_false_tort, blind_true_tort, blind_false_tort,  
                                     ["WT", f"{self.mouse_type}"], labels=["escape", "no escape", "escape", "no escape"],
                                     colors=["teal", "cadetblue", "darkgray", "lightgray"], bar_width=0.1, error_bars=True,
                                    log_y=True, ylim=None, show=False, close=True)
        ax.set_xlabel("Mouse Type")
        ax.set_ylabel("Tortuosity of Path From Previous Escape (log)")

    def plot_behavior(self):
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, data_start=4, data_end=5400)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, data_end=5400, process_coords=True)
        wt_event_angles, blind_event_angles = data.extract_data(self.event_angles_file, data_start=154)
        wt_event_locs, blind_event_locs = data.extract_data(self.event_locs_file, data_start=154, process_coords=True)
        wt_true_angles, wt_false_angles, blind_true_angles, blind_false_angles = data.extract_data(self.event_angles_file, data_start=154, escape=True, escape_col=3)
        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data(self.event_locs_file, data_start=154, escape=True, escape_col=3, process_coords=True)
        
        angle_sets = [wt_baseline_angles, wt_event_angles, blind_baseline_angles, blind_event_angles, wt_true_angles, wt_false_angles, blind_true_angles, blind_false_angles]
        locs_sets = [wt_baseline_locs, wt_event_locs, blind_baseline_locs, blind_event_locs, wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs]

        wt_global_behavior = {}
        wt_event_behavior = {}
        wt_true_behavior = {}
        wt_false_behavior = {}
        blind_global_behavior = {}
        blind_event_behavior = {}
        blind_true_behavior = {}
        blind_false_behavior = {}

        for i, (angle_set, locs_set) in enumerate(zip(angle_sets, locs_sets)):
            behavior_percentages_list = []
            for angles, locs in zip(angle_set, locs_set):
                behavior_percentages = utils.analyse_behavior(angles, locs, fps=30)
                behavior_percentages_list.append(behavior_percentages)

            mean_behavior_percentages = {
            behavior: np.nanmean([
                behavior_data[behavior]
                for behavior_data in behavior_percentages_list
                if behavior_data is not None and behavior in behavior_data
            ])
            for behavior in (behavior_percentages_list[0] if behavior_percentages_list and behavior_percentages_list[0] is not None else {})
            }

            if i == 0:
                wt_global_behavior = mean_behavior_percentages
            elif i == 1:
                wt_event_behavior = mean_behavior_percentages
            elif i == 2:
                blind_global_behavior = mean_behavior_percentages
            elif i == 3:
                blind_event_behavior = mean_behavior_percentages
            elif i == 4:
                wt_true_behavior = mean_behavior_percentages
            elif i == 5:
                wt_false_behavior = mean_behavior_percentages
            elif i == 6:
                blind_true_behavior = mean_behavior_percentages
            elif i == 7:
                blind_false_behavior = mean_behavior_percentages

        fig, axes = plt.subplots(2,4, figsize=(20, 10))
        fig.suptitle('Behavioral Analysis')
        self.figs.append(fig)

        axes[0, 0].pie(wt_global_behavior.values(), labels=wt_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 0].set_title('WT Baseline')

        axes[0, 1].pie(wt_event_behavior.values(), labels=wt_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 1].set_title('WT All')
        
        axes[0, 2].pie(wt_false_behavior.values(), labels=wt_false_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 2].set_title('WT - escape')
        
        axes[0, 3].pie(wt_true_behavior.values(), labels=wt_true_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 3].set_title('WT - no escape')

        axes[1, 0].pie(blind_global_behavior.values(), labels=blind_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 0].set_title(f'{self.mouse_type} Baseline')

        axes[1, 1].pie(blind_event_behavior.values(), labels=blind_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 1].set_title(f'{self.mouse_type} All')

        axes[1, 2].pie(blind_true_behavior.values(), labels=blind_true_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 2].set_title(f'{self.mouse_type}  - escape')
        
        axes[1, 3].pie(blind_false_behavior.values(), labels=blind_false_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 3].set_title(f'{self.mouse_type}  - no escape')

    def plot_speed_data(self):
        global_wt_speeds, global_blind_speeds = data.extract_data(self.global_speeds_file, data_start=4)
        event_wt_speeds, event_blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454)
        
        global_wt_max_speeds = []
        global_blind_max_speeds = []
        global_wt_avg_speeds = []
        global_blind_avg_speeds = []

        event_wt_max_speeds = []
        event_blind_max_speeds = []
        event_wt_avg_speeds = []
        event_blind_avg_speeds = []

        for speeds in event_wt_speeds:
            event_wt_max_speeds.append(max(speeds))
            event_wt_avg_speeds.append(np.nanmean(speeds))

        for speeds in event_blind_speeds:
            event_blind_max_speeds.append(max(speeds))
            event_blind_avg_speeds.append(np.nanmean(speeds))

        for speeds in global_wt_speeds:
            global_wt_max_speeds.append(max(speeds))
            global_wt_avg_speeds.append(np.nanmean(speeds))

        for speeds in global_blind_speeds:
            global_blind_max_speeds.append(max(speeds))
            global_blind_avg_speeds.append(np.nanmean(speeds))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
        self.figs.append(fig)
        
        plots.plot_grouped_bar_chart(fig, ax1, global_wt_avg_speeds, event_wt_avg_speeds, global_blind_avg_speeds, event_blind_avg_speeds,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["baseline", "events", "baseline", "events"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)
        plots.plot_grouped_bar_chart(fig, ax2, global_wt_max_speeds, event_wt_max_speeds, global_blind_max_speeds, event_blind_max_speeds,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["baseline", "events", "baseline", "events"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)
        for ax in [ax1, ax2]:
            ax.set_ylabel("Speed (pps)")
        ax1.set_title("Average Speeds")
        ax2.set_title("Maximum Speeds")
        
        wt_speeds, blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454)
        wt_ages, blind_ages = data.extract_data_rows(self.event_speeds_file, data_row=2, escape=False)
        wt_true_speeds, wt_false_speeds, blind_true_speeds, blind_false_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454, escape=True, escape_col=3)
        wt_true_ages, wt_false_ages, blind_true_ages, blind_false_ages = data.extract_data_rows(self.event_speeds_file, data_row=2, escape=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 3]}, sharex='col')
        
        plots.cmap_plot(fig, axes, wt_speeds, blind_speeds, wt_ages, blind_ages, title1="WT speed", title2=f"{self.mouse_type} speed", 
                        ylabel="Speed (pps)", ylim=(0,250), cmap = "viridis", fps=30)
        self.imgs.append(fig)
        
        fig, axes = plt.subplots(4, 2, figsize=(10,10), gridspec_kw={'height_ratios': [1, 2, 1, 2]}, sharex='col')
        
        plots.cmap_plot(fig, axes[0:2, :], wt_true_speeds, wt_false_speeds, wt_true_ages, wt_false_ages, title1="WT speed - escape", title2="WT speed - no escape",
                       ylabel="Speed (pps)", ylim=(0,250), cmap="viridis", fps=30, cbar_dim=[0.92, 0.51, 0.015, 0.22])
        plots.cmap_plot(fig, axes[2:4, :], blind_true_speeds, blind_false_speeds, blind_true_ages, blind_false_ages, title1=f"{self.mouse_type} speed - escape", title2=f"{self.mouse_type} - no escape",
                       ylabel="Speed (pps)", ylim=(0,250), cmap="viridis", fps=30, cbar_dim=[0.92, 0.11, 0.015, 0.22])
        self.imgs.append(fig)
 
    def plot_arena_coverage_data(self):
        event_wt_locs, event_blind_locs = data.extract_data(self.event_locs_file, nested=True, data_start=154, data_end=None, escape=False, process_coords=True, get_escape_index=False, escape_col=3)
        all_wt_locs, all_blind_locs = data.extract_data(self.global_locs_file, nested=True, data_start=3, process_coords=True, escape_col=None)

        wt_baseline_coverage = utils.calculate_arena_coverage(all_wt_locs)
        blind_baseline_coverage = utils.calculate_arena_coverage(all_blind_locs)
        wt_event_coverage = utils.calculate_arena_coverage(event_wt_locs)
        blind_event_coverage = utils.calculate_arena_coverage(event_blind_locs)

        all_wt_dist =[]
        all_blind_dist=[]
        event_wt_dist =[]
        event_blind_dist =[]
        locs_sets = [all_wt_locs, event_wt_locs, all_blind_locs, event_blind_locs]
        distance_lists = [all_wt_dist, all_blind_dist, event_wt_dist, event_blind_dist]

        for locs_set, dist_set in zip(locs_sets, distance_lists):
            for locs_list in locs_set:
                locs_list = [loc for loc in locs_list if isinstance(loc, (tuple, list)) and len(loc) == 2 and not np.isnan(loc).any()]
                if len(locs_list) < 2:
                    continue  # Skip lists with fewer than two valid points
                total_distance_covered = sum(
                    utils.calc_dist_between_points(locs_list[i], locs_list[i - 1])
                    for i in range(1, len(locs_list))
                )
                dist_set.append(total_distance_covered)
                
        conversion_factor = 46.5 / 645
        all_wt_dist = [d * conversion_factor for d in all_wt_dist]
        all_blind_dist = [d * conversion_factor for d in all_blind_dist]
        event_wt_dist = [d * conversion_factor for d in event_wt_dist]
        event_blind_dist = [d * conversion_factor for d in event_blind_dist]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
        self.figs.append(fig)
        
        plots.plot_grouped_bar_chart(fig, ax1, wt_baseline_coverage, wt_event_coverage, blind_baseline_coverage, blind_event_coverage,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["baseline", "events", "baseline", "events"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)
        plots.plot_grouped_bar_chart(fig, ax2, all_wt_dist, event_wt_dist, all_blind_dist, event_blind_dist,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["baseline", "events", "baseline", "events"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)

        ax1.set_ylabel("Arena Coverage (%)")
        ax2.set_ylabel("Total Distance Covered (cm)")
        
    def plot_location_data(self):
        
        centre_roi = [200, 570, 670, 170]
        
        event_wt_locs, event_blind_locs = data.extract_data(self.event_locs_file, nested=True, data_start=154, data_end=None,escape=False, process_coords=True, get_escape_index=False, escape_col=3)
        all_wt_locs, all_blind_locs = data.extract_data(self.global_locs_file, nested=True, data_start=3, process_coords=True, escape_col=None)
        wt_event_centre, wt_event_edge = utils.analyse_locs(event_wt_locs, 30, centre_roi)
        blind_event_centre, blind_event_edge = utils.analyse_locs(event_blind_locs, 30, centre_roi)
        wt_baseline_centre, wt_baseline_edge = utils.analyse_locs(all_wt_locs, 30, centre_roi)
        blind_baseline_centre, blind_baseline_edge = utils.analyse_locs(all_blind_locs, 30, centre_roi)
        wt_centre, wt_edge, wt_exit, wt_nest = utils.categorise_location(event_wt_locs)
        blind_centre, blind_edge, blind_exit, blind_nest = utils.categorise_location(event_blind_locs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5))
        self.figs.append(fig)
        
        plots.plot_grouped_bar_chart(fig, ax1, wt_baseline_centre, wt_baseline_edge, blind_baseline_centre, blind_baseline_edge,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["centre", "edge", "centre", "edge"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)
        plots.plot_grouped_bar_chart(fig, ax2, wt_event_centre, wt_event_edge, blind_event_centre, blind_event_edge,
                                     xticks=["WT", f"{self.mouse_type}"], labels=["centre", "edge", "centre", "edge"],
                                     colors=["darkorange", "orange", "deeppink", "hotpink"], bar_width=0.1, error_bars=True)

        ax1.set_ylabel("Time Spent (s)")
        ax1.set_title("Baseline")
        ax2.set_title("Events")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        self.figs.append(fig)
        
        plots.plot_bar_four_groups(fig, ax1, wt_centre, wt_edge, wt_exit, wt_nest, xticks=["centre", "edge", "exit region", "nest"], 
                             labels=[], colors=["blue", "green", "yellow", "orange"], 
                             bar_width=0.2, error_bars=False, points=False, log_y=False, ylim=(0,None), show=False, close=True)
        
        plots.plot_bar_four_groups(fig, ax2, blind_centre, blind_edge, blind_exit, blind_nest, xticks=["centre", "edge", "exit region", "nest"], 
                             labels=[], colors=["blue", "green", "yellow", "orange"],
                             bar_width=0.2, error_bars=False, points=False, log_y=False, ylim=(0,None), show=False, close=True)
        
        ax1.set_title("WT")
        ax2.set_title(f"{self.mouse_type}")
        ax1.set_ylabel("Time spent(s)")
        plt.title("Time Spent at Stimulus")

    def plot_distance_from_wall(self):
        wt_event_locs, blind_event_locs = data.extract_data(self.event_locs_file, escape=False, process_coords=True)
        wt_true_locs, blind_true_locs, wt_false_locs, blind_false_locs = data.extract_data(self.event_locs_file, escape=True, process_coords=True, get_escape_index=True, escape_col=3)

        
        
    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder, "test")
        
        if self.save_imgs:
            if self.imgs:
                files.save_report(self.imgs, self.folder)
        
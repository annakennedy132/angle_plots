import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import files, utils
from processing import plots, data, angles

class NewPlots:
    def __init__(self, folder, mouse_type, save_figs=True):
        self.fps = 30
        self.length = 10
        self.t_minus = 5
        self.t_plus = 5

        self.folder = folder

        self.figs = []
        self.save_figs = save_figs

        self.parent_folder = os.path.dirname(self.folder)
        self.global_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("global_data")), None)
        
        self.global_angles_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("angles.csv")), None)
        self.global_locs_file = next((os.path.join(self.global_file, file) for file in os.listdir(self.global_file) if file.endswith("locs.csv")), None)
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
        self.avg_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_angles.csv")), None)
        self.avg_dist_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_distances.csv")), None)
        self.avg_speeds_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_speeds.csv")), None)

        self.mouse_type = mouse_type

    def plot_behavior(self):
        wt_event_angles, blind_event_angles = data.extract_data(self.event_angles_file, data_start=154)
        wt_event_locs, blind_event_locs = data.extract_data(self.event_locs_file, data_start=154, process_coords=True)
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, data_start=4)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, process_coords=True)
        angle_sets = [wt_baseline_angles, wt_event_angles, blind_baseline_angles, blind_event_angles]
        locs_sets = [wt_baseline_locs, wt_event_locs, blind_baseline_locs, blind_event_locs]

        wt_global_behavior = {}
        wt_event_behavior = {}
        blind_global_behavior = {}
        blind_event_behavior = {}

        for i, (angle_set, locs_set) in enumerate(zip(angle_sets, locs_sets)):
            behavior_percentages_list = []
            for angles, locs in zip(angle_set, locs_set):
                behavior_percentages = utils.analyze_behavior(angles, locs, fps=30)
                behavior_percentages_list.append(behavior_percentages)

            mean_behavior_percentages = {
                behavior: np.mean([behavior_data[behavior] for behavior_data in behavior_percentages_list])
                for behavior in behavior_percentages_list[0]
            }

            if i == 0:
                wt_global_behavior = mean_behavior_percentages
            elif i == 1:
                wt_event_behavior = mean_behavior_percentages
            elif i == 2:
                blind_global_behavior = mean_behavior_percentages
            elif i == 3:
                blind_event_behavior = mean_behavior_percentages

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Behavioral Analysis')
        self.figs.append(fig)

        axes[0, 0].pie(wt_global_behavior.values(), labels=wt_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 0].set_title('WT Baseline')

        axes[0, 1].pie(wt_event_behavior.values(), labels=wt_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 1].set_title('WT Event')

        axes[1, 0].pie(blind_global_behavior.values(), labels=blind_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 0].set_title(f'{self.mouse_type} Baseline')

        axes[1, 1].pie(blind_event_behavior.values(), labels=blind_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 1].set_title(f'{self.mouse_type} Event')

    def plot_speed_data(self):
        # Extract data
        wt_speeds, blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454)
        wt_true_speeds, wt_false_speeds, blind_true_speeds, blind_false_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454, escape=True, escape_col=3)
        wt_avg_speeds, blind_avg_speeds = data.extract_data(self.avg_speeds_file, data_start=3, escape=False)

        # Calculate average speeds
        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_avg_speeds]) for i in range(max(map(len, wt_avg_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_avg_speeds]) for i in range(max(map(len, blind_avg_speeds)))]
        ]

        # Determine the min and max speeds for consistent colormap scaling
        vmin = min(np.min(wt_speeds), np.min(blind_speeds))
        vmax = max(np.max(wt_speeds), np.max(blind_speeds))
        num_frames = len(wt_speeds[1])  # Get number of frames from speed data
        frame_time = (1. / self.fps)
        x_ticks = np.linspace(0, num_frames, 4).astype(int)  # Adjust number of ticks as needed
        x_labels = (x_ticks * frame_time) - 5

        # Extract average speeds data
        wt_avg_speeds = avg_speeds_data[0]
        blind_avg_speeds = avg_speeds_data[1]
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 3]}, sharex='col')
        self.figs.append(fig)

        # Plot average speeds above the heatmaps
        axes[0, 0].plot(wt_avg_speeds, color='red')
        axes[0, 0].set_title('Average WT Speed')
        axes[0, 0].set_ylabel('Speed (pps)')
        axes[0, 0].set_ylim(0, 250)
        axes[0, 0].spines['left'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['bottom'].set_visible(False)
        axes[0, 0].get_xaxis().set_visible(False)

        axes[0, 1].plot(blind_avg_speeds, color='red')
        axes[0, 1].set_title(f'Average {self.mouse_type} Speed')
        axes[0, 1].set_ylim(0, 250)
        axes[0, 1].spines['left'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['bottom'].set_visible(False)
        axes[0, 1].get_xaxis().set_visible(False)

        # Plot the heatmaps below the average speeds
        sns.heatmap(wt_speeds, ax=axes[1, 0], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('WT Mice Speed (pps)')
        axes[1, 0].set_ylabel('Trial')
        axes[1, 0].axvline(150, color='black', linewidth=2)  # Stimulus
        axes[1, 0].set_yticks([])
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_xticklabels(x_labels)

        sns.heatmap(blind_speeds, ax=axes[1, 1], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'{self.mouse_type} Speed (pps)')
        axes[1, 1].set_ylabel('Trial')
        axes[1, 1].axvline(150, color='black', linewidth=2)  # Stimulus
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xticks(x_ticks)
        axes[1, 1].set_xticklabels(x_labels)

        # Add colorbar for the heatmaps
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.outline.set_visible(False)
        cbar_ax.set_position([0.93, 0.11, 0.02, 0.525])
        #plt.show()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,8))
        self.figs.append(fig)

        # Plot WT Mice Speed Heatmap
        sns.heatmap(wt_true_speeds, ax=ax1, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
        ax1.set_title('WT Speed - Escape (pps)')
        ax1.set_ylabel('Trial')
        ax1.axvline(150, color='black', linewidth=2)
        ax1.set_yticks([])
        ax1.set_xticks(x_ticks)

        sns.heatmap(wt_false_speeds, ax=ax2, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
        ax2.set_title('WT Speed - No Escape (pps)')
        ax2.set_xlabel('Time (s)')
        ax2.axvline(150, color='black', linewidth=2)
        ax2.set_yticks([])
        ax2.set_xticks(x_ticks)

        sns.heatmap(blind_true_speeds, ax=ax3, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
        ax3.set_title(f'{self.mouse_type} Speed - Escape (pps)')
        ax3.set_ylabel('Trial')
        ax3.axvline(150, color='black', linewidth=2)
        ax3.set_yticks([])
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_labels)

        sns.heatmap(blind_false_speeds, ax=ax4, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax)
        ax4.set_title(f'{self.mouse_type} Speed - No Escape (pps)')
        ax4.set_xlabel('Time (s)')
        ax4.axvline(150, color='black', linewidth=2)
        ax4.set_yticks([])
        ax4.set_xticks(x_ticks)
        ax4.set_xticklabels(x_labels)

        plt.tight_layout()

        wt_max_speeds = []
        blind_max_speeds = []
        wt_avg_speeds = []
        blind_avg_speeds = []

        for speeds in wt_speeds:
            wt_max_speeds.append(max(speeds))
            wt_avg_speeds.append(np.nanmean(speeds))

        for speeds in blind_speeds:
            blind_max_speeds.append(max(speeds))
            blind_avg_speeds.append(np.nanmean(speeds))
        
        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_max_speeds, blind_max_speeds, x_label="Mouse Type", y_label="Max Speed (pps)", bar1_label="WT", bar2_label=f"{self.mouse_type}",
                        color1='blue', color2='green', ylim=None, bar_width=0.2, points=True, 
                        log_y=False, error_bars=False, show=False, close=True, title=None, show_axes='both')

        fig, ax = plt.subplots(figsize=(4,5))
        self.figs.append(fig)
        plots.plot_bar_two_groups(fig, ax, wt_avg_speeds, blind_avg_speeds, x_label="Mouse Type", y_label="Avg Speed (pps)", bar1_label="WT", bar2_label=f"{self.mouse_type}",
                        color1='blue', color2='green', ylim=None, bar_width=0.2, points=True, 
                        log_y=False, error_bars=False, show=False, close=True, title=None, show_axes='both')

    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder, "locs")
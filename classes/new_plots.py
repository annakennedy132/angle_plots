import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import files, utils
from processing import plots, data, stats

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
        self.avg_angles_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_angles.csv")), None)
        self.avg_dist_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_distances.csv")), None)
        self.avg_speeds_file = next((os.path.join(self.folder, file) for file in os.listdir(self.folder) if file.endswith("avg_speeds.csv")), None)

        self.mouse_type = mouse_type

    def plot_behavior(self):
        wt_event_angles, blind_event_angles = data.extract_data(self.event_angles_file, data_start=154)
        wt_event_locs, blind_event_locs = data.extract_data(self.event_locs_file, data_start=154, process_coords=True)
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, data_start=4, data_end=5400)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, data_end=5400, process_coords=True)
        angle_sets = [wt_baseline_angles, wt_event_angles, blind_baseline_angles, blind_event_angles]
        locs_sets = [wt_baseline_locs, wt_event_locs, blind_baseline_locs, blind_event_locs]

        wt_global_behavior = {}
        wt_event_behavior = {}
        blind_global_behavior = {}
        blind_event_behavior = {}

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

        plt.show()

    def plot_behavior_esc(self):
        wt_event_angles, wt_false_angles, blind_event_angles, blind_false_angles = data.extract_data(self.event_angles_file, data_start=154, escape=True, escape_col=3)
        wt_event_locs, wt_false_locs, blind_event_locs, blind_false_locs = data.extract_data(self.event_locs_file, data_start=154, escape=True, escape_col=3, process_coords=True)
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, data_start=4, data_end=5400)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, data_end=5400, process_coords=True)
        angle_sets = [wt_baseline_angles, wt_event_angles, blind_baseline_angles, blind_event_angles]
        locs_sets = [wt_baseline_locs, wt_event_locs, blind_baseline_locs, blind_event_locs]

        wt_global_behavior = {}
        wt_event_behavior = {}
        blind_global_behavior = {}
        blind_event_behavior = {}

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

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Behavioral Analysis - Mice That Escape')
        self.figs.append(fig)

        axes[0, 0].pie(wt_global_behavior.values(), labels=wt_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 0].set_title('WT Baseline')

        axes[0, 1].pie(wt_event_behavior.values(), labels=wt_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 1].set_title('WT Event')

        axes[1, 0].pie(blind_global_behavior.values(), labels=blind_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 0].set_title(f'{self.mouse_type} Baseline')

        axes[1, 1].pie(blind_event_behavior.values(), labels=blind_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 1].set_title(f'{self.mouse_type} Event')

        plt.show()

    def plot_behavior_no_esc(self):
        wt_true_angles, wt_event_angles, blind_true_angles, blind_event_angles = data.extract_data(self.event_angles_file, data_start=154, escape=True, escape_col=3)
        wt_true_locs, wt_event_locs, blind_true_locs, blind_event_locs = data.extract_data(self.event_locs_file, data_start=154, escape=True, escape_col=3, process_coords=True)
        wt_baseline_angles, blind_baseline_angles = data.extract_data(self.global_angles_file, data_start=4, data_end=5400)
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, data_end=5400, process_coords=True)
        angle_sets = [wt_baseline_angles, wt_event_angles, blind_baseline_angles, blind_event_angles]
        locs_sets = [wt_baseline_locs, wt_event_locs, blind_baseline_locs, blind_event_locs]

        wt_global_behavior = {}
        wt_event_behavior = {}
        blind_global_behavior = {}
        blind_event_behavior = {}

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

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Behavioral Analysis - Mice That do not Escape')
        self.figs.append(fig)

        axes[0, 0].pie(wt_global_behavior.values(), labels=wt_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 0].set_title('WT Baseline')

        axes[0, 1].pie(wt_event_behavior.values(), labels=wt_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[0, 1].set_title('WT Event')

        axes[1, 0].pie(blind_global_behavior.values(), labels=blind_global_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 0].set_title(f'{self.mouse_type} Baseline')

        axes[1, 1].pie(blind_event_behavior.values(), labels=blind_event_behavior.keys(), autopct='%1.1f%%', colors=['lightblue', 'orange', 'green', 'yellow'])
        axes[1, 1].set_title(f'{self.mouse_type} Event')

        plt.show()

    def plot_speed_data_bars(self):
        global_wt_speeds, global_blind_speeds = data.extract_data(self.global_speeds_file, data_start=4)
        wt_speeds, blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454)

        wt_max_speeds = []
        blind_max_speeds = []
        wt_avg_speeds = []
        blind_avg_speeds = []

        global_wt_max_speeds = []
        global_blind_max_speeds = []
        global_wt_avg_speeds = []
        global_blind_avg_speeds = []

        for speeds in wt_speeds:
            wt_max_speeds.append(max(speeds))
            wt_avg_speeds.append(np.nanmean(speeds))

        for speeds in blind_speeds:
            blind_max_speeds.append(max(speeds))
            blind_avg_speeds.append(np.nanmean(speeds))

        for speeds in global_wt_speeds:
            global_wt_max_speeds.append(max(speeds))
            global_wt_avg_speeds.append(np.nanmean(speeds))

        for speeds in global_blind_speeds:
            global_blind_max_speeds.append(max(speeds))
            global_blind_avg_speeds.append(np.nanmean(speeds))

        bar_width = 0.3
        x = np.array([0, 1])  # Two main groups (0 for WT, 1 for Blind)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        self.figs.append(fig)

        # Calculate means and standard deviations for error bars
        global_wt_avg_mean = np.mean(global_wt_avg_speeds)
        wt_avg_mean = np.mean(wt_avg_speeds)
        global_blind_avg_mean = np.mean(global_blind_avg_speeds)
        blind_avg_mean = np.mean(blind_avg_speeds)

        global_wt_max_mean = np.mean(global_wt_max_speeds)
        wt_max_mean = np.mean(wt_max_speeds)
        global_blind_max_mean = np.mean(global_blind_max_speeds)
        blind_max_mean = np.mean(blind_max_speeds)

        global_wt_avg_std = np.std(global_wt_avg_speeds)
        wt_avg_std = np.std(wt_avg_speeds)
        global_blind_avg_std = np.std(global_blind_avg_speeds)
        blind_avg_std = np.std(blind_avg_speeds)

        global_wt_max_std = np.std(global_wt_max_speeds)
        wt_max_std = np.std(wt_max_speeds)
        global_blind_max_std = np.std(global_blind_max_speeds)
        blind_max_std = np.std(blind_max_speeds)

        #calculate ratios
        wt_avg_ratio = wt_avg_mean / global_wt_avg_mean
        wt_max_ratio = wt_max_mean / global_wt_max_mean
        blind_avg_ratio = blind_avg_mean / global_blind_avg_mean
        blind_max_ratio = blind_max_mean / global_blind_max_mean

        # Average Speeds Bar Plot with error bars
        axs[0].bar(x[0] - bar_width/2, global_wt_avg_mean, width=bar_width, color='blue', yerr=global_wt_avg_std, capsize=5, label="baseline")
        axs[0].bar(x[0] + bar_width/2, wt_avg_mean, width=bar_width, color='blue', alpha=0.3, yerr=wt_avg_std, capsize=5, label="events")
        axs[0].bar(x[1] - bar_width/2, global_blind_avg_mean, width=bar_width, color='green', yerr=global_blind_avg_std, capsize=5, label="baseline")
        axs[0].bar(x[1] + bar_width/2, blind_avg_mean, width=bar_width, color='green', alpha=0.3, yerr=blind_avg_std, capsize=5, label="events")
        axs[0].legend()

        # Maximum Speeds Bar Plot with error bars
        axs[1].bar(x[0] - bar_width/2, global_wt_max_mean, width=bar_width, color='blue', yerr=global_wt_max_std, capsize=5, label="baseline")
        axs[1].bar(x[0] + bar_width/2, wt_max_mean, width=bar_width, color='blue', alpha=0.3, yerr=wt_max_std, capsize=5, label="events")
        axs[1].bar(x[1] - bar_width/2, global_blind_max_mean, width=bar_width, color='green', yerr=global_blind_max_std, capsize=5, label="baseline")
        axs[1].bar(x[1] + bar_width/2, blind_max_mean, width=bar_width, color='green', alpha=0.3, yerr=blind_max_std, capsize=5, label="events")
        axs[1].legend()

        axs[2].bar(x[0] - bar_width/2, wt_avg_ratio, width=bar_width, color='black', capsize=5, label="avg")
        axs[2].bar(x[0] + bar_width/2, wt_max_ratio, width=bar_width, color='black', alpha=0.3, capsize=5, label="max")
        axs[2].bar(x[1] - bar_width/2, blind_avg_ratio, width=bar_width, color='black', capsize=5, label="avg")
        axs[2].bar(x[1] + bar_width/2, blind_max_ratio, width=bar_width, color='black', alpha=0.3, capsize=5, label="max")
        axs[2].legend()

        axs[2].set_title('Ratio - Baseline vs Event')

        # Set x-ticks and labels
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[0].set_ylabel("Average Speed (pps)")
        axs[1].set_ylabel("Maximum Speed (pps)")

        # Remove right and top spines
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Update legend settings for better positioning and smaller font size
        for ax in axs:
            ax.legend(loc='upper left', fontsize='small', frameon=False, bbox_to_anchor=(1.05, 1))
        # Adjust layout to minimize overlap
        plt.tight_layout()
        plt.show()

    def plot_speed_data_heatmap(self):
        wt_speeds, blind_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454)
        wt_ages, blind_ages = data.extract_data_rows(self.event_speeds_file, data_row=2, escape=False)
        wt_true_speeds, wt_false_speeds, blind_true_speeds, blind_false_speeds = data.extract_data(self.event_speeds_file, data_start=4, data_end=454, escape=True, escape_col=3)
        wt_true_ages, wt_false_ages, blind_true_ages, blind_false_ages = data.extract_data_rows(self.event_speeds_file, data_row=2, escape=True)
        #wt_avg_speeds, blind_avg_speeds = data.extract_data(self.avg_speeds_file, data_start=3, escape=False)

        # Calculate average speeds
        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_speeds]) for i in range(max(map(len, wt_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_speeds]) for i in range(max(map(len, blind_speeds)))]
        ]

        # Determine the min and max speeds for consistent colormap scaling
        vmin = 100
        vmax = 600
        num_frames = len(wt_speeds[1])  # Get number of frames from speed data
        frame_time = (1. / self.fps)
        x_ticks = np.linspace(0, num_frames, 4).astype(int)  # Adjust number of ticks as needed
        x_labels = (x_ticks * frame_time) - 5

        # Extract average speeds data
        wt_avg_speeds = avg_speeds_data[0]
        blind_avg_speeds = avg_speeds_data[1]
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 3]}, sharex='col')
        #self.figs.append(fig)

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

        wt_age_speed_pairs = sorted(zip(wt_ages, wt_speeds), key=lambda x: x[0])
        blind_age_speed_pairs = sorted(zip(blind_ages, blind_speeds), key=lambda x: x[0])

        # Separate sorted ages and speeds after sorting
        wt_sorted_ages, wt_sorted_speeds = zip(*wt_age_speed_pairs)
        blind_sorted_ages, blind_sorted_speeds = zip(*blind_age_speed_pairs)

        wt_sorted_speeds = list(wt_sorted_speeds)
        blind_sorted_speeds = list(blind_sorted_speeds)

        cmap = "viridis"

        # Plot the heatmaps below the average speeds
        sns.heatmap(wt_sorted_speeds, ax=axes[1, 0], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('WT Mice Speed (pps)')
        axes[1, 0].set_ylabel('Trial')
        axes[1, 0].axvline(150, color='black', linewidth=2)  # Stimulus
        axes[1, 0].set_yticks([])
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_xticklabels(x_labels)

        sns.heatmap(blind_sorted_speeds, ax=axes[1, 1], cmap=cmap, cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title(f'{self.mouse_type} Speed (pps)')
        axes[1, 1].set_ylabel('Trial')
        axes[1, 1].axvline(150, color='black', linewidth=2)  # Stimulus
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xticks(x_ticks)
        axes[1, 0].set_xticklabels(x_labels)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_ax = fig.add_axes([0.93, 0.11, 0.015, 0.53])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Speed (pps)', rotation=270, labelpad=10)
        cbar.outline.set_visible(False)

        #plt.show()
        
        wt_true_age_speed_pairs = sorted(zip(wt_true_ages, wt_true_speeds), key=lambda x: x[0])
        wt_false_age_speed_pairs = sorted(zip(wt_false_ages, wt_false_speeds), key=lambda x: x[0])
        blind_true_age_speed_pairs = sorted(zip(blind_true_ages, blind_true_speeds), key=lambda x: x[0])
        blind_false_age_speed_pairs = sorted(zip(blind_false_ages, blind_false_speeds), key=lambda x: x[0])

        # Separate sorted ages and speeds after sorting
        wt_true_sorted_ages, wt_true_sorted_speeds = zip(*wt_true_age_speed_pairs)
        wt_false_sorted_ages, wt_false_sorted_speeds = zip(*wt_false_age_speed_pairs)
        blind_true_sorted_ages, blind_true_sorted_speeds = zip(*blind_true_age_speed_pairs)
        blind_false_sorted_ages, blind_false_sorted_speeds = zip(*blind_false_age_speed_pairs)

        # Convert back to list if needed
        wt_true_sorted_speeds = list(wt_true_sorted_speeds)
        wt_false_sorted_speeds = list(wt_false_sorted_speeds)
        blind_true_sorted_speeds = list(blind_true_sorted_speeds)
        blind_false_sorted_speeds = list(blind_false_sorted_speeds)

        avg_speeds_data = [
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_true_speeds]) for i in range(max(map(len, wt_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_true_speeds]) for i in range(max(map(len, blind_true_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in wt_false_speeds]) for i in range(max(map(len, wt_false_speeds)))],
            [np.nanmean([lst[i] if i < len(lst) else np.nan for lst in blind_false_speeds]) for i in range(max(map(len, blind_false_speeds)))]
        ]

        wt_true_avg_speeds = avg_speeds_data[0]
        blind_true_avg_speeds = avg_speeds_data[1]
        wt_false_avg_speeds = avg_speeds_data[2]
        blind_false_avg_speeds = avg_speeds_data[2]

        # Create figure and axes for 4 line plots and 4 heatmaps
        fig, axes = plt.subplots(4, 2, figsize=(10,8), gridspec_kw={'height_ratios': [1, 2, 1, 2]}, sharex='col')

        # Plot average speeds above the heatmaps
        axes[0, 0].plot(wt_true_avg_speeds, color='red')
        axes[0, 0].set_title('Average WT Speed (escape)')
        axes[0, 0].set_ylabel('Speed (pps)')
        axes[0, 0].set_ylim(0, 250)
        axes[0, 0].spines['left'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['bottom'].set_visible(False)
        axes[0, 0].get_xaxis().set_visible(False)

        axes[0, 1].plot(wt_false_avg_speeds, color='red')
        axes[0, 1].set_title(f'Average WT Speed (no escape)')
        axes[0, 1].set_ylim(0, 250)
        axes[0, 1].spines['left'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['bottom'].set_visible(False)
        axes[0, 1].get_xaxis().set_visible(False)

        axes[2, 0].plot(blind_true_avg_speeds, color='red')
        axes[2, 0].set_title(f'Average {self.mouse_type} Speed (escape)')
        axes[2, 0].set_ylabel('Speed (pps)')
        axes[2, 0].set_ylim(0, 250)
        axes[2, 0].spines['left'].set_visible(False)
        axes[2, 0].spines['right'].set_visible(False)
        axes[2, 0].spines['top'].set_visible(False)
        axes[2, 0].spines['bottom'].set_visible(False)
        axes[2, 0].get_xaxis().set_visible(False)

        axes[2, 1].plot(blind_false_avg_speeds, color='red')
        axes[2, 1].set_title(f'Average {self.mouse_type} Speed (no escape)')
        axes[2, 1].set_ylim(0, 250)
        axes[2, 1].spines['left'].set_visible(False)
        axes[2, 1].spines['right'].set_visible(False)
        axes[2, 1].spines['top'].set_visible(False)
        axes[2, 1].spines['bottom'].set_visible(False)
        axes[2, 1].get_xaxis().set_visible(False)

        # Plot the heatmaps below each line plot
        sns.heatmap(wt_true_sorted_speeds, ax=axes[1, 0], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('WT Speed (pps)')
        axes[1, 0].set_ylabel('Trial')
        axes[1, 0].axvline(150, color='black', linewidth=2)
        axes[1, 0].set_yticks([])
        axes[1, 0].set_xticks(x_ticks)
        axes[1, 0].set_xticklabels(x_labels)

        sns.heatmap(wt_false_sorted_speeds, ax=axes[1, 1], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[1, 1].set_title('WT Speed (pps)')
        axes[1, 1].set_ylabel('Trial')
        axes[1, 1].axvline(150, color='black', linewidth=2)
        axes[1, 1].set_yticks([])
        axes[1, 1].set_xticks(x_ticks)
        axes[1, 1].set_xticklabels(x_labels)

        sns.heatmap(blind_true_sorted_speeds, ax=axes[3, 0], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[3, 0].set_title(f'{self.mouse_type} Speed (pps)')
        axes[3, 0].set_ylabel('Trial')
        axes[3, 0].axvline(150, color='black', linewidth=2)
        axes[3, 0].set_yticks([])
        axes[3, 0].set_xticks(x_ticks)
        axes[3, 0].set_xticklabels(x_labels)

        sns.heatmap(blind_false_sorted_speeds, ax=axes[3, 1], cmap='viridis', cbar=False, vmin=vmin, vmax=vmax)
        axes[3, 1].set_title(f'{self.mouse_type} Speed (pps)')
        axes[3, 1].set_ylabel('Trial')
        axes[3, 1].axvline(150, color='black', linewidth=2)
        axes[3, 1].set_yticks([])
        axes[3, 1].set_xticks(x_ticks)
        axes[3, 1].set_xticklabels(x_labels)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar_ax = fig.add_axes([0.92, 0.11, 0.015, 0.22])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Speed (pps)', rotation=270, labelpad=10)
        cbar.outline.set_visible(False)
        plt.show()

    def plot_coverage_data(self):
        event_wt_locs, event_blind_locs = data.extract_data(self.event_locs_file, nested=True, data_start=154, data_end=None, escape=False, process_coords=True, get_escape_index=False, escape_col=3)
        wt_true_locs, wt_false_locs, blind_true_locs, blind_false_locs = data.extract_data(self.event_locs_file, data_start=154, escape=True, process_coords=True, get_escape_index=True, escape_col=3)
        all_wt_locs, all_blind_locs = data.extract_data(self.global_locs_file, nested=True, data_start=3, process_coords=True, escape_col=None)

        wt_coverage = utils.calculate_arena_coverage(all_wt_locs)
        blind_coverage = utils.calculate_arena_coverage(all_blind_locs)

        wt_dist =[]
        blind_dist =[]
        locs_sets = [all_wt_locs, all_blind_locs]
        distance_lists = [wt_dist, blind_dist]

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

        fig, axs = plt.subplots(1, 2, figsize=(8,5))
        self.figs.append(fig)
        bar_width = 0.4
        x = np.array([0, 1])

        conversion_factor = 46.5 / 645
        wt_dist = [d * conversion_factor for d in wt_dist]
        blind_dist = [d * conversion_factor for d in blind_dist]

        mean1 = np.nanmean(wt_coverage)
        mean2 = np.nanmean(blind_coverage)
        std1 = np.nanstd(wt_coverage)
        std2 = np.nanstd(blind_coverage)

        mean3 = np.nanmean(wt_dist)
        mean4 = np.nanmean(blind_dist)
        std3 = np.nanstd(wt_dist)
        std4 = np.nanstd(blind_dist)

        fig.suptitle('Baseline')

        axs[0].bar(x[0], mean1, yerr=std1, label="WT", color="blue", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[0].bar(x[1], mean2, yerr=std2, label=f"{self.mouse_type}", color="green", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[1].bar(x[0], mean3, yerr=std3, label="WT", color="blue", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[1].bar(x[1], mean4, yerr=std4, label=f"{self.mouse_type}", color="green", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(['WT', f'{self.mouse_type}'])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        axs[0].set_ylabel("Arena Covered (%)")
        axs[1].set_ylabel("Total Distance Covered (cm)")

        wt_coverage = utils.calculate_arena_coverage(event_wt_locs)
        blind_coverage = utils.calculate_arena_coverage(event_blind_locs)

        wt_dist =[]
        blind_dist =[]
        locs_sets = [event_wt_locs, event_blind_locs]
        distance_lists = [wt_dist, blind_dist]

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

        fig, axs = plt.subplots(1, 2, figsize=(8,5))
        self.figs.append(fig)
        bar_width = 0.4
        x = np.array([0, 1])

        conversion_factor = 46.5 / 645
        wt_dist = [d * conversion_factor for d in wt_dist]
        blind_dist = [d * conversion_factor for d in blind_dist]

        mean1 = np.nanmean(wt_coverage)
        mean2 = np.nanmean(blind_coverage)
        std1 = np.nanstd(wt_coverage)
        std2 = np.nanstd(blind_coverage)

        mean3 = np.nanmean(wt_dist)
        mean4 = np.nanmean(blind_dist)
        std3 = np.nanstd(wt_dist)
        std4 = np.nanstd(blind_dist)

        fig.suptitle('Events')

        axs[0].bar(x[0], mean1, yerr=std1, label="WT", color="blue", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[0].bar(x[1], mean2, yerr=std2, label=f"{self.mouse_type}", color="green", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[1].bar(x[0], mean3, yerr=std3, label="WT", color="blue", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        axs[1].bar(x[1], mean4, yerr=std4, label=f"{self.mouse_type}", color="green", alpha=0.6, width=bar_width, zorder=2, capsize=5)
        
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(['WT', f'{self.mouse_type}'])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        axs[0].set_ylabel("Arena Covered (%)")
        axs[1].set_ylabel("Total Distance Covered (cm)")

        wt_true_coverage = utils.calculate_arena_coverage(wt_true_locs)
        wt_false_coverage = utils.calculate_arena_coverage(wt_false_locs)
        blind_true_coverage = utils.calculate_arena_coverage(blind_true_locs)
        blind_false_coverage = utils.calculate_arena_coverage(blind_false_locs)

        fig, ax = plt.subplots(figsize=(8,5))
        self.figs.append(fig)
        plots.plot_grouped_bar_chart(fig, ax, 
                                    wt_true_coverage,
                                    wt_false_coverage, 
                                    blind_true_coverage,
                                    blind_false_coverage,
                                    ["WT - escape", "WT - no escape", f"{self.mouse_type} - escape", f"{self.mouse_type} - no escape"],
                                    "Mouse Type", 
                                    "% Arena Covered After Stimulus",
                                    colors=['tab:blue', 'mediumblue', 'green', 'mediumseagreen'], 
                                    bar_width=0.35,
                                    log_y=False, 
                                    show=False, 
                                    close=True)

    def plot_location_data(self):
        # Define the region of interest
        centre_roi = [200, 570, 670, 170]
        
        # Extract data for WT and blind mice, both event and baseline
        event_wt_locs, event_blind_locs = data.extract_data(self.event_locs_file, nested=True, data_start=154, data_end=None,escape=False, process_coords=True, get_escape_index=False, escape_col=3)
        all_wt_locs, all_blind_locs = data.extract_data(self.global_locs_file, nested=True, data_start=3, process_coords=True, escape_col=None)

        # Analyse locations for each group
        wt_event_centre_mean, wt_event_centre_std, wt_event_edge_mean, wt_event_edge_std = utils.analyse_locs(event_wt_locs, 30, centre_roi)
        blind_event_centre_mean, blind_event_centre_std, blind_event_edge_mean, blind_event_edge_std = utils.analyse_locs(event_blind_locs, 30, centre_roi)
        wt_baseline_centre_mean, wt_baseline_centre_std, wt_baseline_edge_mean, wt_baseline_edge_std = utils.analyse_locs(all_wt_locs, 30, centre_roi)
        blind_baseline_centre_mean, blind_baseline_centre_std, blind_baseline_edge_mean, blind_baseline_edge_std = utils.analyse_locs(all_blind_locs, 30, centre_roi)

        bar_width = 0.3
        x = np.array([0, 1])  

        # Create subplots
        fig, axs = plt.subplots(1, 2, figsize=(8,4))
        self.figs.append(fig)

        axs[0].bar(x[0] - bar_width/2, wt_baseline_centre_mean, width=bar_width, color='blue', capsize=5, label="centre")
        axs[0].bar(x[0] + bar_width/2, wt_baseline_edge_mean, width=bar_width, color='blue', alpha=0.3, capsize=5, label="edge")
        axs[0].bar(x[1] - bar_width/2, blind_baseline_centre_mean, width=bar_width, color='green', capsize=5, label="centre")
        axs[0].bar(x[1] + bar_width/2, blind_baseline_edge_mean, width=bar_width, color='green', alpha=0.3, capsize=5, label="edge")
        axs[0].legend()

        axs[1].bar(x[0] - bar_width/2, wt_event_centre_mean, width=bar_width, color='blue', capsize=5, label="centre")
        axs[1].bar(x[0] + bar_width/2, wt_event_edge_mean, width=bar_width, color='blue', alpha=0.3, capsize=5, label="edge")
        axs[1].bar(x[1] - bar_width/2, blind_event_centre_mean, width=bar_width, color='green', capsize=5, label="centre")
        axs[1].bar(x[1] + bar_width/2, blind_event_edge_mean, width=bar_width, color='green', alpha=0.3, capsize=5, label="edge")

        axs[0].set_title('Baseline')
        axs[1].set_title('Events')
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(['WT', f'{self.mouse_type}'])
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(['WT', f'{self.mouse_type}'])

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        # Set y-labels
        for ax in axs[0:1]:
            ax.set_ylabel('Time spent (%)')
            ax.set_ylim(0,100)

        # Update legend settings for better positioning and smaller font size
        for ax in axs:
            ax.legend(loc='upper left', fontsize='small', frameon=False, bbox_to_anchor=(1.05, 1))

        # Adjust layout to minimize overlap
        plt.tight_layout()
        plt.show()

    def plot_location_esc_data(self):
        wt_event_locs, blind_event_locs = data.extract_data(self.event_locs_file, nested=True, data_start=154, data_end=None, process_coords=True)
        wt_centre, wt_edge, wt_exit, wt_nest = utils.categorise_location(wt_event_locs)
        blind_centre, blind_edge, blind_exit, blind_nest = utils.categorise_location(blind_event_locs)

        fig, ax = plt.subplots(figsize=(4,4))
        self.figs.append(fig)
        bar_width = 0.3
        x = np.array([0, 0.5, 1, 1.5])
        ax.bar(x[0], wt_centre, width=bar_width, color='blue', alpha=0.5, capsize=5, label="centre")
        ax.bar(x[1], wt_edge, width=bar_width, color='green', alpha=0.5, capsize=5, label="edge")
        ax.bar(x[2], wt_exit, width=bar_width, color='yellow', alpha=0.5, capsize=5, label="exit region")
        ax.bar(x[3], wt_nest, width=bar_width, color='orange', alpha=0.5, capsize=5, label="nest")
        ax.set_title("Location over stimulus (WT)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("% time")
        ax.set_xticks(x)
        ax.set_ylim(0,60)
        ax.set_xticklabels(['centre', 'edge', 'exit \n region', 'nest'])

        fig, ax = plt.subplots(figsize=(4,4))
        self.figs.append(fig)
        bar_width = 0.3
        x = np.array([0, 0.5, 1, 1.5])
        ax.bar(x[0], blind_centre, width=bar_width, color='blue', alpha=0.5, capsize=5, label="centre")
        ax.bar(x[1], blind_edge, width=bar_width, color='green', alpha=0.5, capsize=5, label="edge")
        ax.bar(x[2], blind_exit, width=bar_width, color='yellow', alpha=0.5, capsize=5, label="exit region")
        ax.bar(x[3], blind_nest, width=bar_width, color='orange', alpha=0.5, capsize=5, label="nest")
        ax.set_title("Location over stimulus (RD1)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("% time")
        ax.set_ylim(0,60)
        ax.set_xticks(x)
        ax.set_xticklabels(['centre', 'edge', 'exit \n region', 'nest'])

        wt_exit_roi_mice, wt_escape_mice = utils.mice_that_enter_exit_roi(wt_event_locs)
        blind_exit_roi_mice, blind_escape_mice = utils.mice_that_enter_exit_roi(blind_event_locs)

        fig, ax = plt.subplots(figsize=(5,4))
        self.figs.append(fig)
        bar_width = 0.3
        x = np.array([0, 1])
        ax.bar(x[0] - bar_width/2, wt_exit_roi_mice, width=bar_width, color='blue', alpha=0.5, capsize=5, label="")
        ax.bar(x[0] + bar_width/2, wt_escape_mice, width=bar_width, color='blue', alpha=0.3, capsize=5)
        ax.bar(x[1] - bar_width/2, blind_exit_roi_mice, width=bar_width, color='green', alpha=0.5, capsize=5)
        ax.bar(x[1] + bar_width/2, blind_escape_mice, width=bar_width, color='green', alpha=0.3, capsize=5)
        ax.set_title("% Mice that enter the nest region during events (left) \n and those that escape (right)")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylabel("% mice")
        ax.set_xticks(x)
        ax.set_xticklabels(["WT", f"{self.mouse_type}"])
        plt.show()

    def plot_time_to_find_escape(self):
        wt_baseline_locs, blind_baseline_locs = data.extract_data(self.global_locs_file, data_start=4, process_coords=True)
        
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
                        log_y=False, error_bars=False, show=False, close=True, title=None, show_axes='both')
        
        fig, ax = plt.subplots(figsize=(3,4))
        self.figs.append(fig)
        bar_width = 0.3
        x = np.array([0, 1])
        ax.bar(x[0], wt_percentage_useless, width=bar_width, color='blue', capsize=5, label="WT")
        ax.bar(x[1], blind_percentage_useless, width=bar_width, color='green', alpha=0.3, capsize=5, label="RD1")
        ax.set_xlabel("Mouse Type")
        ax.set_title("Percentage of Mice \n that Never Escape")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(['WT', f'{self.mouse_type}'])

    def save_pdfs(self):
        if self.save_figs:
            if self.figs:
                files.save_report(self.figs, self.folder, "behaviour")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from processing import event, data, angles, plots, stats
from utils import files, distance, video

class AnglePlots:

    def __init__(self, tracking_file, save_figs=False):

        # Initialise settings
        self.fps = 30
        self.pcutoff = 0.5
        self.length = 10
        self.t_minus = 5
        self.t_plus = 5
        self.thumbnail_scale = 0.6

        #read tracking data
        self.tracking_file = tracking_file
        tracking_db = files.read_tracking_file(self.tracking_file)

        #initialise time and event attributes
        self.num_frames = tracking_db.shape[0]
        frame_time = (1./self.fps)
        self.total_time = self.num_frames * frame_time
        self.time = np.arange(self.num_frames)* frame_time
        self.norm_event_time = np.arange(-self.t_minus, (self.length + self.t_plus), frame_time)

        #initisalise file name stuff
        self.event_base_paths = []
        self.event_angle_dfs = []

        self.folder, self.filename = os.path.split(self.tracking_file)
        self.base_name = self.filename.removesuffix(".h5")

        parent_folder = os.path.dirname(self.folder)
        self.processed_folder = os.path.join(parent_folder, "processed")
        self.analysis_folder = os.path.join(parent_folder, "analysis")
        try: 
            os.mkdir(self.analysis_folder)
        except:
            pass
        
        self.results_folder = files.create_folder(self.analysis_folder, "ap-output")
        self.base_path = os.path.join(self.results_folder, self.base_name)

        self.stim_file = None
        self.video_file = None
        self.stim_data = np.zeros(self.num_frames)
        self.has_stim_events = False

        #initialise figures list
        self.figs = []
        self.save_figs = save_figs
        
    def process_data(self, stim_file, video_file):

        self.stim_file = stim_file
        self.video_file = video_file

        #create csv name and path
        csv_filename = os.path.splitext(os.path.basename(self.tracking_file))[0] + "_data.csv"
        csv_path = os.path.join(self.processed_folder, csv_filename)

        #create df from tracking and stim data, and extract some of the data
        self.df = data.create_df(self.tracking_file, self.stim_file, self.pcutoff)
        self.head_x, self.head_coords, self.nose_coords, self.frames, self.stim = data.extract_data(self.df)

        #add angles and distance to exit to df using extracted coords
        conversion_factor = 46.5 / 645
        self.angles, self.exit_coords = angles.get_angles_for_plot(self.video_file, self.head_coords, self.nose_coords, thumbnail_scale=0.6)
        self.distances_exit = [
            distance.calc_distance_to_exit(row['nose_x'] if not pd.isna(row['nose_x']) else row['head_x'],
                                           row['nose_y'] if not pd.isna(row['nose_y']) else row['head_y'],
                                           self.exit_coords) * conversion_factor
            for _, row in self.df.iterrows()
        ]
        
        self.df['distance from nose to exit'] = self.distances_exit
        self.df['angle difference'] = self.angles

        self.df.to_csv(csv_path, index=False)

        self.exit_roi = video.get_exit_roi(self.exit_coords)
        
    def save_angles(self, suffix="angles"):
        df = pd.DataFrame((self.angles, self.head_coords, self.distances_exit))
        df.to_csv(os.path.join(self.results_folder, self.base_name + suffix + ".csv"))

        if self.has_stim_events:
            for path, df in zip(self.event_base_paths, self.event_angle_dfs):
                df.to_csv(path + "_" + suffix + ".csv")

    def load_stim_file(self, stim_file):
        
        self.stim_file = stim_file
        
        # read stim data
        if self.stim_file is not None:
            self.stim_data = event.read_stim_file(self.stim_file, self.num_frames)
            self.stim_event_frames = event.get_stim_events(self.stim_data)
            
            if len(self.stim_event_frames) > 0:
                self.has_stim_events = True

    def draw_global_plots(self, show=False, close=True):

        self.angles_polar = [angle if not pd.isna(angle) else None for angle in self.angles]
        self.angles_line = [-abs(angle) for angle in self.angles]

        two_plots_fig_angles, angle_ax = plt.subplots()
        self.figs.append(two_plots_fig_angles)
        plt.title('Angles and Stim Events Over Time')
        plots.two_plots(two_plots_fig_angles,
                        angle_ax,
                        self.time,
                        self.angles_line,
                        self.stim,
                        "tab:blue",
                        x_label='Time (s)', 
                        data1_label='Mouse Facing Angle', 
                        data2_label='Stim (on/off)',
                        data1_lim=(-185, 20),
                        show=show)
        
        two_plots_fig_distance, distance_ax = plt.subplots()
        self.figs.append(two_plots_fig_distance)
        plt.title('Distance and Stim Events over Eime')
        plots.two_plots(two_plots_fig_distance,
                        distance_ax,
                        self.time,
                        self.distances_exit,
                        self.stim,
                        "mediumseagreen",
                        x_label='Time (s)', 
                        data1_label='Distance from Exit (cm)', 
                        data2_label='Stim (on/off)',
                        data1_lim=(0, 55),
                        show=show)

        polar_fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        self.figs.append(polar_fig)
        plt.title('Polar Plot of All Facing Angles')
        plots.plot_polar_chart(polar_fig,
                                ax, 
                                self.angles_polar, 
                                bins=36, 
                                show=show)

        coord_fig, ax = plt.subplots()
        self.figs.append(coord_fig)
        plt.title("Heatmap of All Coords")
        plots.plot_coords(coord_fig, ax, self.head_coords, "x", "y", gridsize=50, vmin=0, vmax=50, xmin=100, xmax=800, ymin=650, ymax=100, show=show)

        if close:
            plt.close('all')

    def draw_event_plots(self, show=False, close=True):
        
        if self.has_stim_events:

            all_event_angles = []
            event_stats = []

            for i, event_t0 in enumerate(self.stim_event_frames, 1):

                start = event_t0 - (self.t_minus*self.fps)
                stim_end = event_t0 + (self.length*self.fps)
                end = stim_end + (self.t_plus*self.fps)
                prev_event = 0
                if i > 1:
                    prev_event = self.stim_event_frames[i - 2]
                
                if end < (self.num_frames - (self.t_plus*self.fps)):

                    #create event folder and csv
                    event_name = 'event-' + str(i)
                    event_folder_name = files.create_folder(self.results_folder, event_name, append_date=False)
                    event_base_path = os.path.join(event_folder_name, self.base_name + "_" + event_name)
                    self.event_base_paths.append(event_base_path)

                    event_angles_line = self.angles_line[start:end]
                    event_angles_polar = self.angles_polar[start:end]
                    event_locs = self.head_coords[start:end]
                    event_stim = self.stim[start:end]
                    event_distances = self.distances_exit[start:end]

                    #find relevant coords/angles and use to find escape stats
                    pre_stim_xcoords = self.head_x[:event_t0]
                    pre_coords = self.head_coords[:event_t0]
                    stim_xcoords = self.head_x[event_t0:stim_end]
                    stim_locs = self.head_coords[event_t0:stim_end]
                    all_stim_locs = self.head_coords[event_t0:end]
                    all_stim_xcoords = self.head_x[event_t0:end]
                    stim_angles = self.angles[event_t0:end]

                    escape_frame = stats.find_escape_frame(stim_xcoords, stim_locs, event_t0, min_escape_frames=5, exit_roi=self.exit_roi)
                    post_stim_xcoords = self.head_x[escape_frame:end]
                    return_frame = stats.find_return_frame(post_stim_xcoords, escape_frame, min_return_frames=15)

                    self.escape_time, self.prev_escape_time, self.prev_escape_frame, distance_from_exit, facing_exit_time = stats.find_escape_stats(self.df,
                                                                                                                            all_stim_xcoords,
                                                                                                                            all_stim_locs,
                                                                                                                            pre_coords,
                                                                                                                            pre_stim_xcoords, 
                                                                                                                            stim_angles, 
                                                                                                                            event_t0,
                                                                                                                            prev_event, 
                                                                                                                            self.fps, 
                                                                                                                            exit_roi=self.exit_roi,
                                                                                                                            min_escape_frames=5)

                    event_stats.append([i, event_locs[self.t_minus*self.fps],
                                        escape_frame,
                                        self.escape_time,
                                        self.prev_escape_time,
                                        distance_from_exit,
                                        facing_exit_time])
                    
                    # Plot line event plot for angles and distances side by side
                    event_two_plots_fig, (angle_ax, distance_ax) = plt.subplots(1, 2, figsize=(12, 5))
                    plt.suptitle(f'Plots of Facing Angles and Distances at Stim Event {i}')
                    self.figs.append(event_two_plots_fig)

                    plots.two_plots(event_two_plots_fig, 
                                    angle_ax,
                                    self.norm_event_time,
                                    event_angles_line, event_stim, "tab:blue",
                                    x_label='Time (s)',
                                    data1_label='Mouse Facing Angle',
                                    data2_label='Stim (on/off)',
                                    data1_lim=(-185, 20), show=show)

                    plots.two_plots(event_two_plots_fig, distance_ax, 
                                    self.norm_event_time, 
                                    event_distances, 
                                    event_stim, "mediumseagreen",
                                    x_label='Time (s)', 
                                    data1_label='Distance from Exit (cm)', 
                                    data2_label='Stim (on/off)',
                                    data1_lim=(0, 55), show=show)
                    
                    #plot polar event plot
                    before_stim_angles = self.angles_polar[start:event_t0]
                    during_stim_angles = self.angles_polar[event_t0:escape_frame]
                    after_stim_angles = self.angles_polar[return_frame:end]
                    prev_esc_locs = self.distances_exit[self.prev_escape_frame:event_t0]

                    polar_titles = ['Before Stimulus', 'During Time to Escape / Stimulus', 'After Escape / Stimulus']
                    angle_lists = [before_stim_angles, during_stim_angles, after_stim_angles]

                    event_polar_fig, axes = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={'projection': 'polar'})
                    event_polar_fig.suptitle(f'Polar Plots of Facing Angles at Stim Event {i}')

                    for ax, title, angles in zip(axes, polar_titles, angle_lists):
                        plots.plot_polar_chart(event_polar_fig, ax, angles, bins=36, show=show)
                        ax.set_title(title)
                    self.figs.append(event_polar_fig)

                    event_coord_fig, ax = plt.subplots()
                    self.figs.append(event_coord_fig)
                    plt.suptitle(f"Heatmap of Coords for Stim Event {i}")
                    plots.plot_coords(event_coord_fig, 
                                      ax, 
                                      event_locs, 
                                      "x", 
                                      "y", 
                                      gridsize=50, 
                                      vmin=0, 
                                      vmax=5, 
                                      xmin=100, 
                                      xmax=800, 
                                      ymin=650, 
                                      ymax=100, 
                                      show_coord=event_locs[self.t_minus*self.fps], 
                                      show=show)
  
                    event_angle_df = pd.DataFrame((event_angles_polar, event_locs, event_distances, during_stim_angles, after_stim_angles, prev_esc_locs, event_angles_line))
                    self.event_angle_dfs.append(event_angle_df)
                    all_event_angles.append(event_angle_df)
                    
                    if close:
                        plt.close('all')

            csv_name = self.base_path + "_escape_stats.csv"
            files.create_csv(event_stats, csv_name)

        else:
            print("Cannot analyze escapes as there were no stimulus events")

    def save_pdf_report(self):
        if len(self.figs) > 0:
            files.save_report(self.figs, self.base_path)
        else:
            print("No traces have been  made yet")


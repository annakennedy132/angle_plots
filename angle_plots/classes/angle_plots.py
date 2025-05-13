import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from angle_plots.processing import calc, display, coordinates, extract, stats, stim, video
from angle_plots.utils import files

class AnglePlots:

    def __init__(self, tracking_file, settings, save_figs=False):

        for k, v in settings["video"].items():
            setattr(self, k, v)
            
        for k, v in settings["tracking"].items():
            setattr(self, k, v)

        #read tracking data
        self.tracking_file = tracking_file
        self.tracking_db = files.read_tracking_file(self.tracking_file)
        
        scorer = self.tracking_db.columns.get_level_values(0)[0]
        self.tracking = self.tracking_db[scorer][self.target_bodypart]
        self.detections = self.tracking['likelihood'].values > self.pcutoff
        self.tracking_detected = self.tracking[self.detections]

        #initialise time and event attributes
        self.num_frames = self.tracking_db.shape[0]
        frame_time = (1./self.fps)
        self.total_time = self.num_frames * frame_time
        self.time = np.arange(self.num_frames)* frame_time
        self.schedule = (self.event["t_minus"], self.event["length"], self.event["t_plus"])
        self.norm_event_time = np.arange(-self.event["t_minus"], (self.event["length"]+self.event["t_plus"]), frame_time)
        
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
        self.df = extract.create_df(self.tracking_file, self.stim_file, self.pcutoff)
        self.head_x, self.head_coords, self.nose_coords, self.frames, self.stim = extract.extract_h5_data(self.df)

        #add angles and distance to exit to df using extracted coords
        conversion_factor = 46.5 / 645
        self.angles, self.exit_coords = coordinates.get_angles_for_plot(self.video_file, self.head_coords, self.nose_coords, thumbnail_scale=0.6)
        self.distances_exit = [
            calc.calc_distance_to_exit(row['nose_x'] if not pd.isna(row['nose_x']) else row['head_x'],
                                           row['nose_y'] if not pd.isna(row['nose_y']) else row['head_y'],
                                           self.exit_coords) * conversion_factor
            for _, row in self.df.iterrows()
        ]
        
        self.df['distance from nose to exit'] = self.distances_exit
        self.df['angle difference'] = self.angles

        self.df.to_csv(csv_path, index=False)
        
        self.exit_roi = video.get_exit_roi(self.exit_coords)

        self.speeds = stats.calculate_speeds(self.tracking, 
                                                  self.detections, 
                                                  self.num_frames, 
                                                  self.fps,
                                                  self.speed_cutoff,
                                                  self.exit_roi)

    def load_stim_file(self, stim_file):
        
        self.stim_file = stim_file
        
        # read stim data
        if self.stim_file is not None:
            self.stim_data = stim.read_stim_file(self.stim_file, self.num_frames)
            self.stim_event_frames = stim.get_stim_events(self.stim_data)
            
            if len(self.stim_event_frames) > 0:
                self.has_stim_events = True
                
    def load_background_image(self, image_file):
        
        self.image_file = image_file
        
    def increase_fig_size(self):
        
        plt.rcParams['figure.figsize'] = [14, 7]
        plt.rcParams["figure.autolayout"] = True   
        
    def save_data(self, suffix="data"):
        df = pd.DataFrame((self.angles, self.head_coords, self.distances_exit, self.speeds))
        df.to_csv(os.path.join(self.results_folder, self.base_name + suffix + ".csv"))

        if self.has_stim_events:
            for path, df in zip(self.event_base_paths, self.event_angle_dfs):
                df.to_csv(path + "_" + suffix + ".csv")

    def draw_global_traces(self, show=False):
        
        self.angles_polar = [angle if not pd.isna(angle) else None for angle in self.angles]
        self.angles_line = [-abs(angle) for angle in self.angles]
        
        # trip grid
        self.trip_fig = display.trip_grid(self.tracking_detected, show=show)
        self.figs.append(self.trip_fig)
        plt.close()
        
        # # time plot
        self.time_fig, ax = plt.subplots()
        self.figs.append(self.time_fig)
        plt.title('Activity over time')
        
        display.time_plot_on_image(ax, 
                                   self.tracking,
                                   self.fps,
                                   self.pcutoff,
                                   image_file=self.image_file,
                                   length=self.total_time,
                                   show=show)
     
        # speed plot
        self.speed_fig, speed_ax = plt.subplots()
        self.figs.append(self.speed_fig)
        plt.title('Velocity')
        
        display.two_plots(self.speed_fig, 
                          speed_ax, 
                          self.time, 
                          self.speeds, 
                          self.stim_data,
                          x_label='Time (s)', 
                          data1_label='Mouse Velocity (pix/s)', 
                          data2_label='stim (on/off)',
                          show=show)

    def draw_event_plots(self, show=False, close=True):
        
        if self.has_stim_events:

            all_event_angles = []
            event_stats = []

            for i, event_t0 in enumerate(self.stim_event_frames, 1):

                start = event_t0 - (self.event["t_minus"]*self.fps)
                stim_end = event_t0 + (self.event["length"]*self.fps)
                end = stim_end + (self.event["t_plus"]*self.fps)
                prev_event = 0
                if i > 1:
                    prev_event = self.stim_event_frames[i - 2]
                
                if end < (self.num_frames - (self.event["t_plus"]*self.fps)):

                    #create event folder and csv
                    event_name = 'event-' + str(i)
                    event_folder_name = files.create_folder(self.results_folder, event_name, append_date=False)
                    event_base_path = os.path.join(event_folder_name, self.base_name + "_" + event_name)
                    self.event_base_paths.append(event_base_path)

                    event_speeds = self.speeds[start:end]
                    event_angles_line = self.angles_line[start:end]
                    event_angles_polar = self.angles_polar[start:end]
                    event_locs = self.head_coords[start:end]
                    event_stim = self.stim[start:end]
                    event_distances = self.distances_exit[start:end]
                    event_tracking = self.tracking.loc[start:end]

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

                    event_stats.append([i, event_locs[self.event["t_minus"]*self.fps],
                                        escape_frame,
                                        self.escape_time,
                                        self.prev_escape_time,
                                        distance_from_exit,
                                        facing_exit_time])
                    
                    during_stim_angles = self.angles_polar[event_t0:escape_frame]
                    after_stim_angles = self.angles_polar[return_frame:end]
                    if self.prev_escape_frame is not None:
                        prev_esc_locs = self.head_x[self.prev_escape_frame:event_t0]
                    else:
                        prev_esc_locs = []

                    event_angle_df = pd.DataFrame((event_angles_polar, event_locs, event_distances, during_stim_angles, after_stim_angles, prev_esc_locs, event_angles_line, event_speeds))
                    self.event_angle_dfs.append(event_angle_df)
                    all_event_angles.append(event_angle_df)
                    
                    event_fig, (speed_ax, time_ax) = plt.subplots(1, 2)
                    self.figs.append(event_fig)
                    
                    plt.title('Event #' + str(i))
                    
                    display.time_plot_on_image(time_ax, 
                                event_tracking, 
                                self.fps, 
                                self.pcutoff, 
                                image_file=self.image_file,
                                schedule=self.schedule,
                                show=False,
                                close=False)
                    
                    display.two_plots(event_fig, 
                                    speed_ax, 
                                    self.norm_event_time, 
                                    event_speeds, 
                                    event_stim,
                                    x_label='Time (s)', 
                                    data1_label='Mouse Velocity (pix/s)', 
                                    data2_label='stim (on/off)',
                                    show=show)
                    
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


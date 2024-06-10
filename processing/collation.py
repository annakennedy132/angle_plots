import os
import csv
import numpy as np
from utils import files
import pandas as pd

def check_for_events(data_folder_path):
    analysis_folder = os.path.join(data_folder_path, "analysis")

    ap_output_folder = next((folder for folder in os.listdir(analysis_folder) if folder.startswith("ap-output")), None)

    if ap_output_folder:
        ap_output_path = os.path.join(analysis_folder, ap_output_folder)
        escape_stats_file = next((file for file in os.listdir(ap_output_path) if file.endswith("escape_stats.csv")), None)
        return os.path.join(ap_output_path, escape_stats_file) if escape_stats_file else None
    else:
        print("No events to collate")
        return None

def read_global_data(data_folder_path, data_folder_name, index_file):

    analysis_folder = os.path.join(data_folder_path, "analysis")
    ap_output_folder = next((folder for folder in os.listdir(analysis_folder) if folder.startswith("ap-output")), None)
    ap_output_path = os.path.join(analysis_folder, ap_output_folder)

    files_list = os.listdir(ap_output_path)

    locs_file_name = [file for file in files_list if file.endswith("angles.csv")][0]
    locs_file_path = os.path.join(ap_output_path, locs_file_name)

    angles_dict = {}
    locs_dict = {}
    distances_dict ={}

    mouse_type = files.get_index_info(data_folder_name, index_file, item=1)
    mouse_age = files.get_index_info(data_folder_name, index_file, item=2)

    with open(locs_file_path, newline="") as locs_file:
        globalreader = csv.reader(locs_file)
        
        # Skip header
        next(globalreader)
        
        angles = next(globalreader)[1:]
        angles_dict[data_folder_name] = angles

        locs = next(globalreader)[1:]
        locs_dict[data_folder_name] = locs

        distances = next(globalreader)[1:]
        distances_dict[data_folder_name] = distances
    
    # Insert age and type for all dictionaries
    data_dicts = [angles_dict, locs_dict, distances_dict]
    for data_dict in data_dicts:
        data_dict[data_folder_name].insert(0, mouse_age)
        data_dict[data_folder_name].insert(0, mouse_type)

    return angles_dict, locs_dict, distances_dict

def read_event_data(data_folder_path, data_folder_name, index_file):
    analysis_folder = os.path.join(data_folder_path, "analysis")
    ap_output_folder = next((folder for folder in os.listdir(analysis_folder) if folder.startswith("ap-output")), None)
    ap_output_path = os.path.join(analysis_folder, ap_output_folder)
    et_output_folder = next((folder for folder in os.listdir(analysis_folder) if folder.startswith("et-output")), None)
    et_output_path = os.path.join(analysis_folder, et_output_folder)

    files_list = os.listdir(ap_output_path)

    meta_file_name = [file for file in files_list if file.endswith("escape_stats.csv")][0]
    meta_file_path = os.path.join(ap_output_path, meta_file_name)

    event_folders_ap = [event_folder for event_folder in os.listdir(ap_output_path) if os.path.isdir(os.path.join(ap_output_path, event_folder))]
    event_folders_et = [event_folder for event_folder in os.listdir(et_output_path) if os.path.isdir(os.path.join(et_output_path, event_folder))]

    meta_dict = {}
    angles_dict = {}
    angles_line_dict = {}
    locs_dict = {}
    distances_dict ={}
    angles_during_dict = {}
    angles_after_dict = {}
    escape_success_dict = {}
    prev_esc_locs_dict = {}
    speeds_dict = {}
    all_event_angles = []
    all_event_locs = []
    all_event_distances = []
    all_event_speeds = []

    mouse_type = files.get_index_info(data_folder_name, index_file, item=1)
    mouse_age = files.get_index_info(data_folder_name, index_file, item=2)

    total_events = 0
    successful_escapes = 0
    escape_success = None
    escape_success_list = []

    with open(meta_file_path, newline="") as metafile:
        metareader = csv.reader(metafile)

        for row in metareader:
            event_name_meta = data_folder_name + "_event-" + row[0]
            meta_dict[event_name_meta] = row[1:]
            
            escape_time = row[3]  # Assuming escape time is in the fourth column

            # Set escape success based on escape time
            if escape_time == "15.0":
                escape_success = False
            else:
                escape_success = True

            # Add additional information to meta_dict
            meta_dict[event_name_meta].insert(0, escape_success)
            meta_dict[event_name_meta].insert(0, mouse_age)
            meta_dict[event_name_meta].insert(0, mouse_type)

            total_events += 1
            if escape_success:
                successful_escapes += 1

            escape_success_list.append(escape_success)

        escape_success_percentage = (successful_escapes / total_events) * 100 if total_events > 0 else 0

        escape_success_data = [mouse_type, mouse_age, escape_success_percentage]
        escape_success_dict[data_folder_name] = escape_success_data
    
    event_folders_ap.sort(key=lambda x: int(x.split('-')[1]))

    for i, event_folder in enumerate(event_folders_ap):
        event_name = data_folder_name + "_" + event_folder

        event_folder_path = os.path.join(ap_output_path, event_folder)
        event_file = os.listdir(event_folder_path)[0]
        event_file_path = os.path.join(event_folder_path, event_file)

        with open(event_file_path, newline="") as eventfile:
            eventreader = csv.reader(eventfile)

            # Skip header row
            next(eventreader)

            # Read angles and store in angles_dict
            angles = next(eventreader)[1:601]
            angles_dict[event_name] = angles

            locs = next(eventreader)[1:601]
            locs_dict[event_name] = locs

            distances = next(eventreader)[1:601]
            distances_dict[event_name] = distances

            during_angles = next(eventreader)[1:]
            angles_during_dict[event_name] = during_angles

            after_angles = next(eventreader)[1:]
            angles_after_dict[event_name] = after_angles

            prev_esc_locs = next(eventreader)[1:]
            prev_esc_locs_dict[event_name] = prev_esc_locs

            angles_line = next(eventreader)[1:601]
            angles_line_dict[event_name] = angles_line

            # Convert to floats and store in all_event_angles
            all_event_angles.append(np.array([float(angle) if angle else np.nan for angle in angles_line]))
            all_event_locs.append(np.array([loc for loc in locs]))
            all_event_distances.append(np.array([float(distance) if distance else np.nan for distance in distances]))

            data_dicts = [angles_dict, locs_dict, distances_dict, angles_during_dict, angles_after_dict, prev_esc_locs_dict]

            # Retrieve escape success flag for the current event
            for data_dict in data_dicts:
                data_dict[event_name].insert(0, escape_success_list[i])
                data_dict[event_name].insert(0, mouse_age)
                data_dict[event_name].insert(0, mouse_type)
    
    event_folders_et.sort(key=lambda x: int(x.split('-')[1]))

    for i, event_folder in enumerate(event_folders_et):
        event_name = data_folder_name + "_" + event_folder

        event_folder_path = os.path.join(et_output_path, event_folder)
        event_file = os.listdir(event_folder_path)[0]
        event_file_path = os.path.join(event_folder_path, event_file)

        with open(event_file_path, newline="") as eventfile:
            eventreader = csv.reader(eventfile)

            # Skip header row
            next(eventreader)

            # Read speeds and store in speeds dict
            speeds = next(eventreader)[1:601]
            speeds_dict[event_name] = speeds

            all_event_speeds.append(np.array([float(speed) if speed else np.nan for speed in speeds]))

            speeds_dict[event_name].insert(0, escape_success_list[i])
            speeds_dict[event_name].insert(0, mouse_age)
            speeds_dict[event_name].insert(0, mouse_type)

    average_angles = np.nanmean(all_event_angles, axis=0)
    average_angles = np.concatenate(([mouse_type], [mouse_age], average_angles))
    average_distances = np.nanmean(all_event_distances, axis=0)
    average_distances = np.concatenate(([mouse_type], [mouse_age], average_distances))
    average_speeds = np.nanmean(all_event_speeds, axis=0)
    average_speeds = np.concatenate(([mouse_type], [mouse_age], average_speeds))

    return (meta_dict, angles_dict, locs_dict, distances_dict, speeds_dict, angles_during_dict, angles_after_dict, prev_esc_locs_dict, escape_success_dict, average_angles, average_distances, average_speeds)
        
def write_collated_global_data(path, data):

    with open(path, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        #write the header row (event name)
        writer.writerow(data.keys())

        # Find the longest list of values
        max_length = max(len(values) for values in data.values())

        #write the data rows, transposed so it is in columns
        #ensure all data is included even if dictionaries are of different length
        for i in range(max_length):
            row = [data[key][i] if i < len(data[key]) else "" for key in data.keys()]
            writer.writerow(row)

def write_collated_event_data(path, data):

    with open(path, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        #write the header row (event name)
        writer.writerow(data.keys())

        #write the data rows, transposed so it is in columns
        writer.writerows(zip(*data.values()))
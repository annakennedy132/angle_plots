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
    '''baseline_locs_dict = {}'''

    mouse_type = files.get_index_info(data_folder_name, index_file, item=1)
    mouse_age = files.get_index_info(data_folder_name, index_file, item=2)
    '''stim_start = files.get_index_info(data_folder_name, index_file, item=3)'''

    with open(locs_file_path, newline="") as locs_file:
        globalreader = csv.reader(locs_file)
        
        # Skip header
        next(globalreader)
        
        angles = next(globalreader)[1:]
        angles_dict[data_folder_name] = angles

        locs = next(globalreader)[1:]
        locs_dict[data_folder_name] = locs
        
        '''# Read locs data for baseline_locs_dict
        baseline_locs = []
        for row in globalreader:
            time_point = int(row[0])
            if time_point <= stim_start:
                baseline_locs.append(row[1:stim_start + 1])  # Exclude data after stim_start
        baseline_locs_dict[data_folder_name] = baseline_locs'''

        distances = next(globalreader)[1:]
        distances_dict[data_folder_name] = distances
    
    # Insert age and type for all dictionaries
    data_dicts = [angles_dict, locs_dict, distances_dict, '''baseline_locs_dict''']
    for data_dict in data_dicts:
        data_dict[data_folder_name].insert(0, mouse_age)
        data_dict[data_folder_name].insert(0, mouse_type)

    return angles_dict, locs_dict, distances_dict

def read_event_data(data_folder_path, data_folder_name, index_file):

    analysis_folder = os.path.join(data_folder_path, "analysis")
    ap_output_folder = next((folder for folder in os.listdir(analysis_folder) if folder.startswith("ap-output")), None)
    ap_output_path = os.path.join(analysis_folder, ap_output_folder)

    files_list = os.listdir(ap_output_path)

    meta_file_name = [file for file in files_list if file.endswith("escape_stats.csv")][0]
    meta_file_path = os.path.join(ap_output_path, meta_file_name)

    event_folders = [event_folder for event_folder in os.listdir(ap_output_path) if os.path.isdir(os.path.join(ap_output_path, event_folder))]

    meta_dict = {}
    angles_dict = {}
    angles_line_dict = {}
    locs_dict = {}
    distances_dict ={}
    angles_during_dict = {}
    angles_after_dict = {}
    escape_success_dict = {}
    prev_esc_locs_dict = {}
    all_event_angles = []
    all_event_locs = []
    all_event_distances = []

    mouse_type = files.get_index_info(data_folder_name, index_file, item=1)
    mouse_age = files.get_index_info(data_folder_name, index_file, item=2)

    total_events = 0
    successful_escapes = 0

    with open(meta_file_path, newline="") as metafile:
        metareader = csv.reader(metafile)

        for row in metareader:
            event_name_meta = data_folder_name + "_event-" + row[0]
            meta_dict[event_name_meta] = row[1:]
            
            escape_time = row[3]  # Assuming escape time is in the fourth column
            
            if escape_time == "15.0":
                escape_success = False
            else:
                escape_success = True

            meta_dict[event_name_meta].insert(0, escape_success)
            meta_dict[event_name_meta].insert(0, mouse_age)
            meta_dict[event_name_meta].insert(0, mouse_type)

            total_events += 1
            if escape_success:
                successful_escapes += 1

        escape_success_percentage = (successful_escapes / total_events) * 100 if total_events > 0 else 0

        escape_success_data = [mouse_type, mouse_age, escape_success_percentage]
        escape_success_dict[data_folder_name] = escape_success_data

        for event_folder in event_folders:
            event_name = data_folder_name + "_" + event_folder

            event_folder_path = os.path.join(ap_output_path, event_folder)
            event_file = os.listdir(event_folder_path)[0]
            event_file_path = os.path.join(event_folder_path, event_file)

            with open(event_file_path, newline="") as eventfile:

                eventreader = csv.reader(eventfile)

                #skip header row
                next(eventreader)

                #read angles and store in angles_dict
                angles = next(eventreader)[1:]
                angles_dict[event_name] = angles

                locs = next(eventreader)[1:]
                locs_dict[event_name] = locs

                distances = next(eventreader)[1:]
                distances_dict[event_name] = distances

                during_angles = next(eventreader)[1:]
                angles_during_dict[event_name] = during_angles
            
                after_angles = next(eventreader)[1:]
                angles_after_dict[event_name] = after_angles

                prev_esc_locs = next(eventreader)[1:]
                prev_esc_locs_dict[event_name] = prev_esc_locs

                angles_line = next(eventreader)[1:]
                angles_line_dict[event_name] = angles_line
    
                #convert to floats and store in all_event_angles
                all_event_angles.append(np.array([float(angle) if angle else np.nan for angle in angles_line]))
                all_event_locs.append(np.array([loc for loc in locs]))
                all_event_distances.append(np.array([float(distance) if distance else np.nan for distance in distances]))

                if any(np.isnan([float(dist) if dist else np.nan for dist in distances[150:]])):
                    escape_success = True
                else:
                    escape_success = False

                data_dicts = [angles_dict, locs_dict, distances_dict, angles_during_dict, angles_after_dict]

                for data_dict in data_dicts:
                        data_dict[event_name].insert(0, escape_success)
                        data_dict[event_name].insert(0, mouse_age)
                        data_dict[event_name].insert(0, mouse_type)

        average_angles = np.nanmean(all_event_angles, axis=0)
        average_angles = np.concatenate(([mouse_type], [mouse_age], average_angles))
        average_distances = np.nanmean(all_event_distances, axis=0)
        average_distances = np.concatenate(([mouse_type], [mouse_age], average_distances))

        return meta_dict, angles_dict, locs_dict, distances_dict, average_angles, average_distances, angles_during_dict, angles_after_dict, prev_esc_locs_dict, escape_success_dict
        
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

def create_average_csv(csv):
    filename = os.path.splitext(os.path.basename(csv))[0]
    folder_path = os.path.dirname(csv)
    avg_file = filename.rsplit("_", 1)[-1]

    # Read the data
    df = pd.read_csv(csv, header=None)

    # Extract mouse types and data
    mouse_types = df.iloc[1].tolist()
    data = df.iloc[3:].astype(float)

    # Calculate row averages for each mouse type
    averages = {}
    for mouse_type in set(mouse_types):
        columns = [i for i, mt in enumerate(mouse_types) if mt == mouse_type]
        averages[mouse_type] = data.iloc[:, columns].mean(axis=1)

    # Create DataFrame with row averages
    averages_df = pd.DataFrame(averages)

    # Save the processed data to a new CSV file
    output_file = os.path.join(folder_path, avg_file + "_avg.csv")
    averages_df.to_csv(output_file, header=True, index=False, mode='w')  # Set mode to 'w'

def create_average_csv_escape_success(csv):

    filename = os.path.splitext(os.path.basename(csv))[0]
    folder_path = os.path.dirname(csv)
    avg_file = filename.rsplit("_", 1)[-1]

    # Read the data
    df = pd.read_csv(csv)

    # Extract mouse types, escape success, and data
    mouse_types = df.iloc[0].tolist()
    escape_success = df.iloc[2].tolist()
    data = df.iloc[3:].astype(float)

    # Initialize dictionaries to store columns for each category
    columns = {
        "wt_true": [], "wt_false": [],
        "rd1_true": [], "rd1_false": [],
    }

    # Iterate through columns
    for col, (mt, esc) in enumerate(zip(mouse_types, escape_success)):
        if mt == "rd1" and esc:
            columns["rd1_true"].append(data.iloc[:, col])
        elif mt.startswith("rd1") and not esc:
            columns["rd1_false"].append(data.iloc[:, col])
        elif mt == "wt" and esc:
            columns["wt_true"].append(data.iloc[:, col])
        elif mt == "wt" and not esc:
            columns["wt_false"].append(data.iloc[:, col])

    # Calculate row averages for each category
    row_averages = {}
    for key, cols in columns.items():
        if cols:  # Check if there are any columns in this category
            row_averages[key] = pd.concat(cols, axis=1).mean(axis=1)
        else:
            row_averages[key] = pd.Series([])  # Empty series

    # Create DataFrame with row averages
    averages_df = pd.DataFrame(row_averages)

    # Save the processed data to a new CSV file
    output_file = os.path.join(folder_path, avg_file + "_avg_escape_success.csv")
    averages_df.to_csv(output_file, header=True, index=False, mode='w')  # Set mode to 'w'


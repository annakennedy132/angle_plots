import os
import yaml
import pandas as pd
from datetime import datetime
import csv
from matplotlib.backends.backend_pdf import PdfPages

def load_config():
    
    with open('config.yaml') as config:
        settings = yaml.load(config.read(), Loader=yaml.Loader)
        
    return settings

def read_tracking_file(tracking_filename):
    tracking = pd.read_hdf(tracking_filename)
    return tracking

def load_stim_file(tracking_file):
    
    folder = os.path.dirname(tracking_file)

    all_paths = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    stim_file = None

    for file in all_paths:
        
        if file.endswith("_stim.csv") or file.endswith("_sound.csv"): # sound csv legacy
            stim_file = file
    
    if stim_file is None:
        raise NameError("No stim file in folder")
    
    return stim_file

def load_video_file(tracking_file):

    folder = os.path.dirname(tracking_file)
    
    all_paths = [os.path.join(folder, file) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    video_file = None

    for file in all_paths:
        
        if file.endswith("_arena.avi"):
            video_file = file
    
    if video_file is None:
        raise NameError("No stim file in folder")
    
    return video_file

def read_tracking_file(tracking_file):
    
    tracking = pd.read_hdf(tracking_file)
    
    return tracking
    
def keep_indexed_folders(data_folders, index_file):
    with open(index_file) as file:
        indices = [index.strip().split(",") for index in file]

    # Extract folder names from indices
    index_folders = [index[0] for index in indices]

    # Filter data_folders based on index_folders
    filtered_data_folders = {key: value for key, value in data_folders.items() if key in index_folders}

    # Print loading message for each remaining folder
    for key in filtered_data_folders.keys():
        print(f"Loading {key} via index file")

    return filtered_data_folders

def get_index_info(folder_name, index_file, item):
    with open(index_file) as file:
        for line in file:
            parts = line.strip().split(",")
            if int(parts[0]) == int(folder_name):  # Check if number matches folder_name
                value = parts[item].strip()  # Get the mouse type from the second item
                return value
    return None

def get_data_folders(parent_folder):
    
    data_folders = {folder_name: os.path.join(parent_folder, folder_name) for folder_name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, folder_name))}
    
    return data_folders

def create_folder(base_folder, name, append_date=True):
    
    if append_date:
        now = datetime.now()
        time = now.strftime("_%Y-%m-%d_%H-%M-%S")
        name = name + time
    
    new_folder = os.path.join(base_folder, name)
        
    os.mkdir(new_folder)
    return new_folder

def create_csv(list, filename):
    
    with open(filename, 'w', newline='') as file:
        
        writer = csv.writer(file)
        
        for row in list:
            if row[1] is None:
                row = [row[0], "None", "None"]
            writer.writerow(row)

def save_report(figs, base_path):
    
    with PdfPages(base_path + '_report.pdf') as pdf:

        for fig in figs:
            pdf.savefig(fig)
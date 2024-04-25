import os
import yaml
import pandas as pd
from datetime import datetime
import csv
from matplotlib.backends.backend_pdf import PdfPages
from utils import parse

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

def h5_file(arg):
    
    if arg.endswith(".h5"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be an .h5 file")

def read_tracking_file(tracking_file):
    
    tracking = pd.read_hdf(tracking_file)
    
    return tracking

def get_filenames(parent_folder):
    
    all_paths = [os.path.join(parent_folder, file) for file in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, file))]
    tracking_file = stim_file = None

    for file in all_paths:
        
        if file.endswith(".h5"):
            tracking_file = h5_file(file)
            
        if file.endswith("_stim.csv") or file.endswith("_sound.csv"): # sound csv legacy
            stim_file = parse.csv_file(file)
    
    if tracking_file is not None:
        return tracking_file, stim_file
    else:
        raise NameError("Tracking file required")
    
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
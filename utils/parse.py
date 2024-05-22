import os
import numpy as np

def threshold(arg):
    
    try:
        
        threshold = int(arg)
        
        return threshold
        
    except TypeError as e:
        
        print(e)

def frame_ignore(arg):
    
    try:
        
        frame_ignore = int(arg)
        
        return frame_ignore
        
    except TypeError as e:
        
        print(e)

def text_file(arg):
    
    if arg.endswith(".txt"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("The file should be a .txt file")

def video_file(arg):
    
    if arg.endswith(".avi"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be an .avi file")
    
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
    
def csv_file(arg):
    
    if arg.endswith(".csv"):
        
        try:
            file = open(arg)
            file.close()
            return arg
        except FileNotFoundError as err:
            print(err)
            
    else:
        
        raise NameError("Needs to be a .csv file")
    
def folder(arg):
    
    if os.path.isdir(arg):
        
        return arg
            
    else:
        
        raise NameError("Needs to be directory path")

    
def get_filenames(parent_folder):
    
    all_paths = [os.path.join(parent_folder, file) for file in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, file))]
    tracking_file = stim_file = video_file = None

    for file in all_paths:
        
        if file.endswith(".h5"):
            tracking_file = h5_file(file)
            
        if file.endswith("_stim.csv") or file.endswith("_sound.csv"): # sound csv legacy
            stim_file = csv_file(file)
        
        if file.endswith("_arena.avi"):
            video_file = video_file(file)
    
    if tracking_file is not None:
        return tracking_file, stim_file, video_file
    else:
        raise NameError("Tracking file required")

def parse_coord(coord_str):
    # If it's already a tuple, return it
    if isinstance(coord_str, tuple):
        return coord_str
    # If it's a string, parse it
    elif isinstance(coord_str, str):
        # Convert string to tuple of floats
        try:
            return tuple(map(float, coord_str.strip('[]').split(',')))
        except ValueError:
            return np.nan
    # If it's a float or NaN, return NaN
    elif isinstance(coord_str, (float, np.floating)) or (isinstance(coord_str, np.ndarray) and np.isnan(coord_str)):
        return np.nan
    # For other types, return NaN
    else:
        return np.nan



    

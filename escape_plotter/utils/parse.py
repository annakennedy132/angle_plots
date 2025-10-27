import os
import numpy as np
    
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

def parse_coord(coord_str):
    
    if isinstance(coord_str, tuple):
        return coord_str
    
    elif isinstance(coord_str, str):
        try:
            return tuple(map(float, coord_str.strip('[]').split(',')))
        except ValueError:
            return (np.nan, np.nan)
        
    elif isinstance(coord_str, (float, np.floating)) or (isinstance(coord_str, np.ndarray) and np.isnan(coord_str)):
        return (np.nan, np.nan)
    
    else:
        return (np.nan, np.nan)
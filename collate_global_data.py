import sys
import os
from rich.progress import track

from processing import collation
from utils import files, parse

def run():

    FOLDER, INDEX_FILE = parse_args()

    data_folders = files.get_data_folders(FOLDER)
    global_angles_data = {}
    global_locs_data = {}
    global_dist_data = {}    

    if INDEX_FILE is not None:

        data_folders = files.keep_indexed_folders(data_folders, INDEX_FILE)
        output_folder = os.path.dirname(INDEX_FILE)

    else:

        output_folder = FOLDER
        print("loading all folders...")

    for folder_name, path in track(data_folders.items(), description="Collating global data..."):

        try:

            global_angles, global_locs, global_dist = collation.read_global_data(path, folder_name, INDEX_FILE)

            global_angles_data.update(global_angles)
            global_locs_data.update(global_locs)
            global_dist_data.update(global_dist)

            print(f"Successfuly collated {folder_name}")
        
        except:
            
            print(f"No data to analyse for {folder_name}")
        
        collated_global_angles_path = os.path.join(output_folder, "collated_global_angles.csv")
        collation.write_collated_global_data(collated_global_angles_path, global_angles_data)

        collated_global_locs_path = os.path.join(output_folder, "collated_global_locs.csv")
        collation.write_collated_global_data(collated_global_locs_path, global_locs_data)
            
        collated_global_dist_path = os.path.join(output_folder, "collated_global_distances.csv")
        collation.write_collated_global_data(collated_global_dist_path, global_dist_data)


def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("Folder and index file must be specified")
    
    elif len(sys.argv) == 2:
        
        folder = parse.folder(sys.argv[1])
        index_file = None
        
    elif len(sys.argv) == 3:
        
        folder = parse.folder(sys.argv[1])
        index_file = parse.text_file(sys.argv[2])      
             
    else:
        
        raise KeyError("Too many input arguments")
    
    return folder, index_file

if __name__ == "__main__":
    run()
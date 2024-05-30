import os
import sys
import shutil
from utils import files, parse
from rich.progress import track


def run():
    FOLDER, INDEX_FILE = parse_args()
    data_folders = files.get_data_folders(FOLDER)

    if INDEX_FILE is not None:
        data_folders = files.keep_indexed_folders(data_folders, INDEX_FILE)
    else:
        print("loading all folders")
    
    for folder_name, path in track(data_folders.items()):
        try:
            processed_folder = os.path.join(path, "analysis")
            for item in os.listdir(processed_folder):
                ap_folder = os.path.join(processed_folder, item)
                if item.startswith("ap-output"):
                    shutil.rmtree(ap_folder)
                    print(f"Removed folder: {ap_folder}")
                
        except Exception as e:
            print(f"Failed to remove {ap_folder}: {e}")

def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("tracking, sound, and video files must be specified")
    
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

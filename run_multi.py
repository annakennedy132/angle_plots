import sys
import os
from rich.progress import track

from utils import files, parse
from classes.angle_plots import AnglePlots


def run():
    FOLDER, INDEX_FILE = parse_args()
    data_folders = files.get_data_folders(FOLDER)

    if INDEX_FILE is not None:
        data_folders = files.keep_indexed_folders(data_folders, INDEX_FILE)
    else:
        print("loading all folders")
    
    for folder_name, path in track(data_folders.items()):

        try:
            processed_folder = os.path.join(path, "processed")

            tracking_file, stim_file, video_file = parse.get_filenames(processed_folder)

            ap = AnglePlots(tracking_file)
            ap.process_data(stim_file, video_file)
            ap.load_stim_file(stim_file)
            ap.draw_global_plots()
            ap.draw_event_plots()
            ap.save_angles()
            ap.save_pdf_report()

            print(f"Successfully processed {folder_name}")
            
        except:
            
            print(f"Error with video {folder_name}")

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
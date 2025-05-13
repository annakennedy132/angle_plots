import sys
from utils import parse
from classes.final_plots import FinalPlots

def run():

    FOLDER = parse_args()
    #mouse_type = input("Input blind mouse type: ")
    mouse_type="rd1"

    fp = FinalPlots(FOLDER, mouse_type)
    fp.plot_coord_data()
    fp.plot_angle_data()
    fp.plot_avgs_data()
    fp.plot_stats_data()
    fp.plot_traj_data()
    fp.plot_tort_data()
    fp.plot_behavior()
    fp.plot_speed_data()
    fp.plot_arena_coverage_data()
    fp.plot_location_data()
    fp.save_pdfs()

def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("Folder must be specified")
    
    elif len(sys.argv) == 2:
        
        folder = parse.folder(sys.argv[1])
        
    else:
        
        raise KeyError("Too many input arguments")
    
    return folder

if __name__ == "__main__":
    run()


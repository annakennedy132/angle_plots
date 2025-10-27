import sys
from escape_plotter.utils import parse
from escape_plotter import FinalPlots, config

def run():
    FOLDER = parse_args()
    
    #mouse_types = ["wt-light", "wt-dark", "rd1-light", "rd1-dark"]  # Define mouse types here
    mouse_types = ["wt", "rd1"]
    #mouse_types = ["wt", "rd1", "ogc"]
    
    fp = FinalPlots(FOLDER, config, mouse_types=mouse_types)  # Pass mouse_types directly
    fp.plot_coord_data()
    #fp.plot_angle_data()
    #fp.plot_stats_data()
    #fp.plot_traj_data()
    #fp.plot_tort_data()
    #fp.plot_behaviour()
    #fp.plot_speed_data()
    #fp.plot_arena_coverage_data()
    ##fp.plot_distance_from_wall(seconds_before_escape=15)
    #fp.plot_path_similarity_and_area()
    fp.save_figs()

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
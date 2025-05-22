import sys
from angle_plots.utils import parse
from angle_plots import FinalPlots3, config

def run():

    FOLDER = parse_args()
    
    mouse_type_1="wt"
    mouse_type_2="rd1"
    mouse_type_3="ogc"
    
    fp = FinalPlots3(FOLDER, config, mouse_type_1, mouse_type_2, mouse_type_3)
    fp.plot_coord_data()
    fp.plot_angle_data()
    fp.plot_avgs_data()
    fp.plot_stats_data()
    fp.plot_traj_data()
    fp.plot_tort_data()
    fp.plot_behavior()
    #fp.plot_speed_data()
    fp.plot_arena_coverage_data()
    fp.plot_location_data()
    #fp.plot_distance_from_wall()
    #fp.plot_path_similarity()
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


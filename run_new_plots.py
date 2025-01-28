import sys
from utils import parse
from classes.new_plots import NewPlots

def run():

    FOLDER = parse_args()
    #mouse_type = input("Input blind mouse type: ")
    mouse_type="RD1"

    np = NewPlots(FOLDER, mouse_type)
    np.plot_speed_data_bars()
    np.plot_speed_data_heatmap()
    np.plot_behavior()
    np.plot_behavior_esc()
    np.plot_behavior_no_esc()
    np.plot_coverage_data()
    np.plot_location_esc_data()
    np.plot_location_data()
    np.plot_time_to_find_escape()
    np.save_pdfs()

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
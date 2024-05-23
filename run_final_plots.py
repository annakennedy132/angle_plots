import sys
from utils import parse
from classes.final_plots import FinalPlots

def run():

    FOLDER = parse_args()

    fp = FinalPlots(FOLDER)
    fp.plot_global_data()
    fp.plot_event_data()
    fp.plot_avgs_data()
    fp.plot_stats_data()
    fp.plot_tort_data()
    fp.save_pdfs()
    fp.plot_traj_data()

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

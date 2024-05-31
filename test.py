import sys
from utils import parse
from classes.poster_plots import PosterPlots

def run():

    FOLDER = parse_args()

    pp = PosterPlots(FOLDER)
    #pp.plot_global_data()
    pp.plot_event_data()
    pp.plot_stats_data()
    pp.plot_tort_data()
    pp.plot_prev_tort()
    #pp.plot_traj_data()
    pp.save_pdfs()

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


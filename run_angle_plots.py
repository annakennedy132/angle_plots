import sys
from angle_plots.utils import files, parse
from angle_plots import AnglePlots, config

def run():

    TRACKING_FILE = parse_args()
    STIM_FILE = files.load_stim_file(TRACKING_FILE)
    VIDEO_FILE = files.load_video_file(TRACKING_FILE)

    ap = AnglePlots(TRACKING_FILE, config)
    
    ap.process_data(STIM_FILE, VIDEO_FILE)
    ap.load_stim_file(STIM_FILE)
    
    ap.load_background_image(config["tracking"]["background_image"])
    ap.increase_fig_size()
    
    ap.draw_global_traces()
    ap.draw_event_plots()
    
    ap.save_data()
    ap.save_pdf_report()

def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("tracking file must be specified")
    
    elif len(sys.argv) == 2:
        
        folder = parse.h5_file(sys.argv[1])
            
    else:
        
        raise KeyError("Too many input arguments")
    
    return folder

if __name__ == "__main__":
    run()


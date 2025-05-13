import sys

from angle_plots import VideoSegmenter, config
from angle_plots.utils import parse

def run():
    
    VIDEO_FILE = parse_args()
    
    vc = VideoSegmenter(VIDEO_FILE, config)
    vc.segment()
    
def parse_args():

    if len(sys.argv) == 1:
        raise KeyError("No file specified")

    elif len(sys.argv) == 2:
        video_file = parse.video_file(sys.argv[1])
                      
    else:
        raise KeyError("Too many input arguments")
    
    return video_file

if __name__ == "__main__":
    run()

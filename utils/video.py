import os
import cv2
from rich import print as rprint

def get_frame(video_name):
    
    video = cv2.VideoCapture(video_name)
    
    for _ in range(500):
        
        ret, frame = video.read()
        
        if ret:
            
            blank_frame = frame
            
        else:
            
            break
                
    video.release()
    
    return blank_frame

def get_exit(frame, thumbnail_scale):
    
    ex, ey = get_user_loc("Place coordinate on exit centre", frame, thumbnail_scale)
    
    rprint("Exit coordinates: ", ex, " ", ey)
    
    return (ex, ey)
                
def get_user_loc(msg, frame, thumbnail_scale):
  
    # displaying the image
    small_frame = cv2.resize(frame, 
                             (int(round(frame.shape[1]*thumbnail_scale)), 
                              int(round(frame.shape[0]*thumbnail_scale)))
                             )
    
    cv2.imshow(msg, small_frame)
  
    # setting mouse handler for the image and calling the click_event() function
    global click_x, click_y
    cv2.setMouseCallback(msg, click_event)
  
    # wait for a key to be pressed to exit
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    x = int(round(click_x/thumbnail_scale))
    y = int(round(click_y/thumbnail_scale))
    
    return (x, y)

def click_event(event, x, y, flags, params):
  
    global click_x, click_y
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
  
        rprint(x, " ", y, end='\r')
        # displaying the coordinates on the Shell
        click_x, click_y = x, y

def get_exit_roi(exit_coord):
    exit_x, exit_y = exit_coord
    roi = [exit_x - 20,
           exit_y - 100,
           exit_x + 60,
           exit_y + 100]
    
    return roi

import math
from utils import video

def get_exit_coords(video_file, thumbnail_scale):
    frame = video.get_frame(video_file)
    return video.get_exit(frame, thumbnail_scale)

def get_head_nose_angle(head_coords, nose_coords):
    angles = [round(math.degrees(math.atan2(head_y - nose_y, head_x - nose_x)) % 360, 2)
              for (head_x, head_y), (nose_x, nose_y) in zip(head_coords, nose_coords)]
    return angles

def get_head_exit_angle(head_coords, exit_coords):
    angles = [round(math.degrees(math.atan2(head_y - exit_coords[1], head_x - exit_coords[0])) % 360, 2)
              for (head_x, head_y) in head_coords]
    return angles

def get_angle_difference(head_nose, head_exit):
    angle_difference = [round(head_nose - head_exit, 2) for head_nose, head_exit in zip(head_nose, head_exit)]
    
    adjusted_angle_difference = []
    for angle in angle_difference:
        if angle >= 180:
            adjusted_angle = angle - 360
        elif -360 <= angle <= -180:
            adjusted_angle = angle + 360
        else:
            adjusted_angle = angle
        adjusted_angle_difference.append(adjusted_angle)

    return adjusted_angle_difference

def get_angles_for_plot(video_file, head_coords, nose_coords, thumbnail_scale):
    exit_coords = get_exit_coords(video_file, thumbnail_scale)
    head_nose = get_head_nose_angle(head_coords, nose_coords)
    head_exit = get_head_exit_angle(head_coords, exit_coords)
    angle_difference = get_angle_difference(head_nose, head_exit)
    
    return angle_difference, exit_coords
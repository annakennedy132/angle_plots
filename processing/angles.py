import math
from utils import video
import numpy as np

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

def get_angles_for_plot(video_file, head_coords, nose_coords, thumbnail_scale=0.6):
    exit_coords = get_exit_coords(video_file, thumbnail_scale)
    head_nose = get_head_nose_angle(head_coords, nose_coords)
    head_exit = get_head_exit_angle(head_coords, exit_coords)
    angle_difference = get_angle_difference(head_nose, head_exit)
    
    return angle_difference, exit_coords

def rotate_coordinates(coordinates, exit_point, angle_degrees):
    # Convert angle from degrees to radians
    angle = np.radians(-angle_degrees)  # negative because it's clockwise
    
    # Convert coordinates and exit point to NumPy arrays
    coordinates_array = np.array(coordinates)
    exit_point_array = np.array(exit_point)
    
    # Translate coordinates so that exit_point becomes the origin
    translated_coords = coordinates_array - exit_point_array
    
    # Calculate the rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    
    # Perform the rotation
    rotated_coords = np.dot(rotation_matrix, translated_coords.T).T
    
    # Translate coordinates back
    rotated_coords = rotated_coords + exit_point_array
    
    return rotated_coords

def get_rotated_coords(exit_coord, coords):
    final_coord = coords[-1]
    y_difference = final_coord[1] - exit_coord[1]
    shifted_coords = [(x, y - y_difference) for x, y in coords]
    first_coord = shifted_coords[0]
    other_coord = (first_coord[0], exit_coord[1])

    oe_length = exit_coord[0] - other_coord[0]
    of_length = other_coord[1] - first_coord[1]

    angle_degrees = math.degrees(math.atan(of_length / oe_length))
    rotated_coords = rotate_coordinates(shifted_coords, exit_coord, angle_degrees)

    return rotated_coords


import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_distance_to_exit(x, y, exit_coords):
    exit_x, exit_y = exit_coords
    return euclidean_distance(x, y, exit_x, exit_y)


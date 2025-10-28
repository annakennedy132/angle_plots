import numpy as np
from scipy.ndimage import gaussian_filter1d

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_dist_between_points(coord_1, coord_2):
    x1, y1 = coord_1
    x2, y2 = coord_2
    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def point_to_line_distance(point, line):
    x0, y0 = point
    (x1, y1), (x2, y2) = line

    A = np.array([x1, y1])
    B = np.array([x2, y2])
    P = np.array([x0, y0])

    AB = B - A
    AP = P - A

    AB_squared = np.dot(AB, AB)
    if AB_squared == 0:
        return np.linalg.norm(AP), A

    t = np.dot(AP, AB) / AB_squared
    if t < 0:
        closest = A
    elif t > 1:
        closest = B
    else:
        closest = A + t * AB

    dist = np.linalg.norm(P - closest)
    
    return dist

def point_to_rect(point, corners, skip=None):
    # Bail out if point is None or contains NaNs
    if point is None or np.any(np.isnan(point)):
        return np.nan

    min_distance = float('inf')

    for i in range(len(corners)):
        if i == skip:
            continue
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        dist = point_to_line_distance(point, (p1, p2))

        if dist < min_distance:
            min_distance = dist

    return min_distance

def apply_smoothing(data, sigma):
    if sigma > 0:
        return gaussian_filter1d(data, sigma=sigma)
    else:
        return np.array(data)  # no smoothing

def compute_avg_trace(data_list):
    if not data_list:
        return []

    max_len = max(map(len, data_list))
    result = []
    for i in range(max_len):
        values = [(lst[i]) if i < len(lst) else np.nan for lst in data_list]
        values = [v for v in values if not np.isnan(v)]
        result.append(np.mean(values) if values else np.nan)
    return result

def compute_avg_angle_trace(data_list):
    if not data_list:
        return []

    max_len = max(map(len, data_list))
    result = []
    for i in range(max_len):
        values = [abs(lst[i]) if i < len(lst) else np.nan for lst in data_list]
        values = [v for v in values if not np.isnan(v)]
        result.append(-np.mean(values) if values else np.nan)
    return result
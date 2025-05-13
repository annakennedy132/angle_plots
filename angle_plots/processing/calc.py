import numpy as np
from angle_plots.processing import coordinates
from scipy.spatial import cKDTree

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_distance_to_exit(x, y, exit_coords):
    exit_x, exit_y = exit_coords
    return euclidean_distance(x, y, exit_x, exit_y)

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

def point_to_rect(point, corners):
    # Bail out if point is None or contains NaNs
    if point is None or np.any(np.isnan(point)):
        return np.nan

    min_distance = float('inf')

    for i in range(len(corners)):
        p1 = corners[i]
        p2 = corners[(i + 1) % len(corners)]
        dist = point_to_line_distance(point, (p1, p2))

        if dist < min_distance:
            min_distance = dist

    return min_distance

def calculate_arena_coverage(locations, grid_size=20, arena_bounds=(90, 790, 80, 670)):
        xmin, xmax, ymin, ymax = arena_bounds

        x_bins = np.arange(xmin, xmax + grid_size, grid_size)
        y_bins = np.arange(ymin, ymax + grid_size, grid_size)

        total_cells = (len(x_bins) - 1) * (len(y_bins) - 1)  # Total number of grid cells in the arena

        coverage_percentages = []

        for mouse_locs in locations:
            filtered_locs = [loc for loc in mouse_locs if isinstance(loc, (list, tuple)) and len(loc) == 2 and not (np.isnan(loc[0]) or np.isnan(loc[1]))]
            x_coords = [loc[0] for loc in filtered_locs]
            y_coords = [loc[1] for loc in filtered_locs]
            x_grid = np.digitize(x_coords, x_bins)
            y_grid = np.digitize(y_coords, y_bins)
            visited_cells = set(zip(x_grid, y_grid))
            percentage_covered = (len(visited_cells) / total_cells) * 100
            coverage_percentages.append(percentage_covered)

        return coverage_percentages

def compute_nearest_euclidean_similarity(before_locs_list, stim_locs_list):
            scores = []
            for before_locs, stim_locs in zip(before_locs_list, stim_locs_list):
                before_locs = np.array(before_locs[::-1])
                stim_locs = np.array(stim_locs)

                before_locs = before_locs[~np.isnan(before_locs).any(axis=1)]
                stim_locs = stim_locs[~np.isnan(stim_locs).any(axis=1)]

                if len(before_locs) < 2 or len(stim_locs) < 2:
                    continue

                normalized = coordinates.normalize_length([before_locs, stim_locs])
                before_locs = np.array(normalized[0])
                stim_locs = np.array(normalized[1])

                tree = cKDTree(stim_locs)
                distances, _ = tree.query(before_locs, k=1)
                avg_distance = np.mean(distances)
                similarity = 1 / (1 + avg_distance)
                scores.append(similarity)
            return scores
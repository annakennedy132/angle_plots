import numpy as np

def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calc_distance_to_exit(x, y, exit_coords):
    exit_x, exit_y = exit_coords
    return euclidean_distance(x, y, exit_x, exit_y)

def calc_dist_between_points(coord_1, coord_2):
    x1, y1 = coord_1
    x2, y2 = coord_2
    
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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

def categorise_behavior(total_angle_change, total_distance):
    # Thresholds for behavior categorization based on total distance and total angle change
    stationary_dist_threshold = 35
    directed_dist_threshold = 100
    stationary_angle_threshold = 500
    directed_angle_threshold = 500

    if total_distance < stationary_dist_threshold and total_angle_change < stationary_angle_threshold:
        return "stationary"
    elif total_distance > directed_dist_threshold and total_angle_change < directed_angle_threshold:
        return "directed"
    else: 
        return "exploratory"

def analyse_behavior(angles, locs, fps):
    time_frame = fps * 1  # 2 seconds chunks (assuming 30 fps, this would be 60 frames)
    num_chunks = len(angles) // time_frame

    behavior_counts = {"stationary": 0, "exploratory": 0, "directed": 0}

    for i in range(num_chunks):
        chunk_angles = angles[i * time_frame: (i + 1) * time_frame]
        chunk_distances = locs[i * time_frame: (i + 1) * time_frame]

        # Calculate total angles covered
        total_angle_change = sum(abs(chunk_angles[j] - chunk_angles[j - 1]) for j in range(1, len(chunk_angles)))

        # Filter out NaN values in chunk_distances
        valid_distances = [coord for coord in chunk_distances if not np.isnan(coord).any()]

        # Check if there are valid coordinates in valid_distances
        if len(valid_distances) < 2:
            continue

        # Calculate path length using the first and last valid coordinates
        total_distance = calc_dist_between_points(valid_distances[-1], valid_distances[0])

        # Categorize behavior based on the new angle and distance calculations
        behavior = categorise_behavior(total_angle_change, total_distance)
        if behavior in behavior_counts:
            behavior_counts[behavior] += 1

    total_count = sum(behavior_counts.values())
    if total_count == 0:
        return None

    behavior_percentages = {behavior: count / total_count * 100 for behavior, count in behavior_counts.items()}
    
    return behavior_percentages
    
def compute_mean_behavior(all_angles, all_distances):
        # Initialize counters for the mean
        mean_behavior_counts = {"stationary": 0, "exploratory": 0, "directed": 0, "in nest": 0}
        num_datasets = len(all_angles)

        # Process each list of angles and distances
        for angles, distances in zip(all_angles, all_distances):
            behavior_percentages = analyse_behavior(angles, distances)

            # Accumulate behavior percentages
            for behavior, percentage in behavior_percentages.items():
                mean_behavior_counts[behavior] += percentage

        # Compute the mean by dividing by the number of datasets
        mean_behavior_counts = {behavior: count / num_datasets for behavior, count in mean_behavior_counts.items()}

        return mean_behavior_counts

def analyse_locs(all_locs, fps, centre_roi):
    centre_times = []
    edge_times = []
    
    for locs_list in all_locs:
        centre_count = 0
        edge_count = 0
        for loc in locs_list:
            if not np.isnan(loc).any():
                if (centre_roi[0] <= loc[0] <= centre_roi[2]) and (centre_roi[1] >= loc[1] >= centre_roi[3]):
                    centre_count += 1
                else:
                    edge_count += 1
            elif np.isnan(loc).any():
                continue
                
        # Convert frame counts to time in seconds
        centre_times.append(centre_count / fps)
        edge_times.append(edge_count / fps)

    centre_mean = np.nanmean(centre_times)
    centre_std = np.nanstd(centre_times)
    edge_mean = np.nanmean(edge_times)
    edge_std = np.nanstd(edge_times)

    return centre_mean, centre_std, edge_mean, edge_std

def categorise_location(locs):
    exit_roi = [650, 500, 800, 240]
    centre_roi = [200, 570, 670, 170]

    centre_times = []
    edge_times = []
    exit_times = []
    nest_times = []
    
    for locs_list in locs:

        centre_count = 0
        edge_count = 0
        exit_count = 0
        nest_count = 0

        for loc in locs_list:
            if not np.isnan(loc).any():
                if (centre_roi[0] <= loc[0] <= centre_roi[2]) and (centre_roi[1] >= loc[1] >= centre_roi[3]):
                    centre_count += 1
                elif (exit_roi[0] <= loc[0] <= exit_roi[2]) and (exit_roi[1] >= loc[1] >= exit_roi[3]):
                    exit_count += 1
                else:
                    edge_count += 1
            else:
                nest_count += 1
    
        centre_count = centre_count / len(locs_list) * 100
        edge_count = edge_count / len(locs_list) * 100
        exit_count = exit_count / len(locs_list) * 100
        nest_count = nest_count / len(locs_list) * 100

        centre_times.append(centre_count)
        edge_times.append(edge_count)
        exit_times.append(exit_count)
        nest_times.append(nest_count)

    centre_mean = np.nanmean(centre_times)
    edge_mean = np.nanmean(edge_times)
    exit_mean = np.nanmean(exit_times)
    nest_mean = np.nanmean(nest_times)

    return centre_mean, edge_mean, exit_mean, nest_mean

def mice_that_enter_exit_roi(locs_list, exit_roi = [650, 240, 800, 500], min_escape_frames=5):
    exit_roi_mice = 0
    escape_mice = 0

    for locs in locs_list:
        for i in range(0, len(locs)):
            if exit_roi is not None:
                coord_in_roi = (exit_roi[0] <= locs[i][0] <= exit_roi[2]) and \
                                                (exit_roi[1] <= locs[i][1] <= exit_roi[3])
                if coord_in_roi:
                    exit_roi_mice += 1
                    break

    for locs in locs_list:
            for i in range(0, len(locs) - min_escape_frames + 1):
                if all(np.isnan(locs[j][0]) and np.isnan(locs[j][1]) for j in range(i, i + min_escape_frames)):
                    last_coord_index = i - 1
                    if exit_roi is not None:
                        if last_coord_index >= 0:
                            last_coord_in_roi = (exit_roi[0] <= locs[last_coord_index][0] <= exit_roi[2]) and \
                                                (exit_roi[1] <= locs[last_coord_index][1] <= exit_roi[3])
                            if last_coord_in_roi:
                                escape_mice += 1
                                break
    
    percentage_exit_roi_mice = exit_roi_mice / len(locs_list) * 100
    percentage_escape_mice = escape_mice / len(locs_list) * 100
                     
    return percentage_exit_roi_mice, percentage_escape_mice
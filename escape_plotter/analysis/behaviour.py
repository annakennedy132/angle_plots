import numpy as np
from escape_plotter.processing import calc

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
        total_distance = calc.calc_dist_between_points(valid_distances[-1], valid_distances[0])

        # Categorize behavior based on the new angle and distance calculations
        behavior = categorise_behavior(total_angle_change, total_distance)
        if behavior in behavior_counts:
            behavior_counts[behavior] += 1

    total_count = sum(behavior_counts.values())
    if total_count == 0:
        return None

    behavior_percentages = {behavior: count / total_count * 100 for behavior, count in behavior_counts.items()}
    
    return behavior_percentages
    
def compute_mean_behaviour(angles, locs, fps=30):
    """Return list of dicts (one per mouse) with mean behaviour percentages."""
    means = []
    for angles_list, locs_list in zip(angles, locs):
        per_trial = [
            analyse_behavior(a, l, fps=fps)
            for a, l in zip(angles_list, locs_list)
        ]
        per_trial = [d for d in per_trial if d]  # drop None
        if not per_trial:
            means.append({})
            continue
        keys = per_trial[0].keys()
        means.append({
            k: np.nanmean([d.get(k, np.nan) for d in per_trial]) for k in keys
        })
    return means
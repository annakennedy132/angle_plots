import numpy as np

def find_escape_stats(df, all_stim_coords, stim_locs, pre_coords, pre_xcoords, post_angles, start_frame, prev_end, fps, exit_roi=None, min_escape_frames=5):

    escape_index = len(all_stim_coords)
    for i in range(0, len(all_stim_coords) - min_escape_frames + 1):
        if all(np.isnan(all_stim_coords[i: i + min_escape_frames])):
            last_coord_index = i - 1
            if exit_roi is not None:
                if last_coord_index > 0:  # Ensure index is within bounds
                    last_coord_in_roi = (exit_roi[0] <= stim_locs[last_coord_index][0] <= exit_roi[2]) and \
                                        (exit_roi[1] <= stim_locs[last_coord_index][1] <= exit_roi[3])
                    if last_coord_in_roi:
                        escape_index = i
                        break
            else:
                escape_index = None

    escape_time = round(float(escape_index) / fps, 2)

    prev_escape_time = None
    prev_escape_index = None
    for i in range(len(pre_xcoords) - min_escape_frames, prev_end, -1):
        if np.isnan(pre_xcoords[i]) and all(np.isnan(pre_xcoords[i - min_escape_frames + 1: i + 1])):
            if exit_roi is not None:
                previous_coord_index = i + 1
                if previous_coord_index > 0:
                    previous_coord_in_roi = (exit_roi[0] <= pre_coords[previous_coord_index][0] <= exit_roi[2]) and \
                                            (exit_roi[1] <= pre_coords[previous_coord_index][1] <= exit_roi[3])
                    if previous_coord_in_roi:
                        prev_escape_index = i
                        break
                else:
                    prev_escape_index = None

    if prev_escape_index is not None:
        prev_escape_time = round(float(start_frame - prev_escape_index) / fps, 2)

    distance_from_exit = df["distance from nose to exit"].iloc[start_frame]

    facing_exit_index = len(post_angles)
    facing_exit_time = None
    for i, angle in enumerate(post_angles):
        if -10 <= angle <= 10:
            facing_exit_index = i
            break
    if facing_exit_index < len(post_angles):
        facing_exit_time = round(float(facing_exit_index) / fps , 2)

    return escape_time, prev_escape_time, prev_escape_index, distance_from_exit, facing_exit_time

def find_escape_frame(stim_xcoords, stim_locs, start_frame, min_escape_frames=5, exit_roi=None):
    escape_index = len(stim_xcoords)
    
    for i in range(0, len(stim_xcoords) - min_escape_frames + 1):
        if all(np.isnan(stim_xcoords[i: i + min_escape_frames])):
            last_coord_index = i - 1
            if exit_roi is not None:
                if last_coord_index >= 0:
                    last_coord_in_roi = (exit_roi[0] <= stim_locs[last_coord_index][0] <= exit_roi[2]) and \
                                        (exit_roi[1] <= stim_locs[last_coord_index][1] <= exit_roi[3])
                    if last_coord_in_roi:
                        escape_index = i
                        break
            else:
                escape_index = i
    
    escape_frame = start_frame + escape_index

    return escape_frame

def find_return_frame(post_stim_coords, escape_frame, min_return_frames=15):

    return_index = len(post_stim_coords)
    for i in range(0, len(post_stim_coords) - min_return_frames +1):
          if all(~np.isnan(post_stim_coords[i:i + min_return_frames])):
            return_index = i
            break
    return_frame = escape_frame + return_index

    return return_frame

def find_global_esc_frame(coords, min_escape_frames=5, exit_roi=None, fps=30):

    # Ensure coords is a list of tuples or (nan, nan)
    cleaned_coords = []
    for item in coords:
        if isinstance(item, tuple) and len(item) == 2:
            cleaned_coords.append(item)
        elif np.isnan(item):  # If it's a float (nan), replace it with (nan, nan)
            cleaned_coords.append((np.nan, np.nan))
        else:
            raise ValueError(f"Invalid coordinate format: {item}")

    # Find the first valid coordinate (non-(nan, nan))
    enter_index = None
    for i in range(len(cleaned_coords)):
        x, y = cleaned_coords[i]
        if not np.isnan(x) and not np.isnan(y):  # Check if both x and y are valid
            enter_index = i
            break

    for escape_index in range(enter_index, len(cleaned_coords) - min_escape_frames + 1):
        if all(np.isnan(cleaned_coords[j][0]) and np.isnan(cleaned_coords[j][1]) for j in range(escape_index, escape_index + min_escape_frames)):
            last_coord_index = escape_index - 1
            # Check if the last valid coordinate is within the exit ROI
            if exit_roi is not None and last_coord_index >= 0:
                last_coord = cleaned_coords[last_coord_index]
                x, y = last_coord
                if not (exit_roi[0] <= x <= exit_roi[2] and exit_roi[1] <= y <= exit_roi[3]):
                    continue
            
            # Calculate the time to escape in seconds
            time_to_escape = escape_index / fps
            return time_to_escape

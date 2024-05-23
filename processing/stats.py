import numpy as np

def find_escape_stats(df, all_stim_coords, pre_coords, post_angles, start_frame, prev_end, fps, min_escape_frames=5):

    escape_index = len(all_stim_coords)
    for i in range(0, len(all_stim_coords) - min_escape_frames + 1):
        if all(np.isnan(all_stim_coords[i: i + min_escape_frames])):
            escape_index = i
            break
    escape_time = round(float(escape_index) / fps , 2)

    prev_escape_time = None
    prev_escape_index = None
    for i in range(len(pre_coords) - min_escape_frames, prev_end, -1):
        if np.isnan(pre_coords[i]) and all(np.isnan(pre_coords[i - min_escape_frames + 1: i + 1])):
            prev_escape_index = i
            break
    if prev_escape_index is not None:
        prev_escape_time = round(float(start_frame - prev_escape_index) / fps , 2)

    distance_from_exit = df["distance from nose to exit"].iloc[start_frame]

    facing_exit_index = len(post_angles)
    facing_exit_time = None
    for i, angle in enumerate(post_angles):
        if -10 <= angle <= 10:
            facing_exit_index = i
            break
    if facing_exit_index < len(post_angles):
        facing_exit_time = round(float(facing_exit_index) / fps , 2)

    return escape_time, prev_escape_time, distance_from_exit, facing_exit_time

def find_escape_frame(stim_xcoords, stim_locs, start_frame, min_escape_frames=5, exit_roi=None):
    escape_index = len(stim_xcoords)
    last_coord_in_roi = False  # Initialize to False
    for i in range(0, len(stim_xcoords) - min_escape_frames + 1):
        if all(np.isnan(stim_xcoords[i: i + min_escape_frames])):
            last_coord_index = i - 1
            if exit_roi is not None:
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

        
        




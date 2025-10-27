import numpy as np

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

def find_escape_index(column_data, min_escape_frames=5, exit_roi=(700, 300, 800, 500)):
    """
    Find the first index i such that from i to i+min_escape_frames-1 the data are all NaN,
    AND the last non-NaN sample before that block satisfies:
      - COORDINATES (Nx2): last point lies inside exit_roi = (x1,y1,x2,y2)

    Returns:
        escape_index (int): start index of the first qualifying all-NaN block,
                            or len(column_data) if not found.
    """
    cleaned = [np.nan if val == 0.0 else val for val in column_data]
    arr = np.asarray(cleaned, dtype=float)

    # Determine whether this is angle series (1D) or coordinate series (2D with last dim 2)
    if arr.ndim == 1:
        mode = "angle"
    elif arr.ndim == 2 and arr.shape[1] == 2:
        mode = "coord"
    else:
        # Try to coerce: if it's 2D but not 2 columns, treat as angle-like
        mode = "angle"

    # Step 1: first index with min_escape_frames consecutive non-NaN values
    start_index = None
    for i in range(len(arr) - min_escape_frames + 1):
        window = arr[i:i + min_escape_frames]
        if np.all(~np.isnan(window)):
            start_index = i
            break

    if start_index is None:
        return len(arr)  # never had enough non-NaNs to start counting

    # Step 2: from that point, find first block of min_escape_frames all NaN
    escape_index = len(arr)
    for i in range(start_index, len(arr) - min_escape_frames + 1):
        window = arr[i:i + min_escape_frames]
        if np.all(np.isnan(window)):
            last_idx = i - 1
            if last_idx >= 0:
                if mode == "coord":
                    # Require last coordinate to be inside ROI
                    if exit_roi is None:
                        continue  # no ROI to check; skip
                    x1, y1, x2, y2 = exit_roi
                    last_xy = arr[last_idx]
                    if not np.any(np.isnan(last_xy)):
                        x, y = last_xy
                        in_roi = (x1 <= x <= x2) and (y1 <= y <= y2)
                        if in_roi:
                            escape_index = i
                            break
                else:  # mode == "angle"
                    escape_index = i
                    break

    return escape_index

def is_nest_frame(p, roi):
    x1, y1, x2, y2 = roi
    try:
        # Try treating as (x,y)
        x, y = p
        if np.isnan(x) or np.isnan(y):
            return True
        return (x1 <= x <= x2) and (y1 <= y <= y2)
    except Exception:
        # Scalar-like
        return bool(np.isnan(p))
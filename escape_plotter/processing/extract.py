import pandas as pd
import numpy as np
from escape_plotter.utils import parse
from escape_plotter.processing.escape import find_escape_index

def extract_data_rows(file, data_row=None, escape=False, process_coords=False, mouse_type=None):
    all_data = []
    false_data = []
    true_data = []

    df = pd.read_csv(file, header=None, low_memory=False)
    mouse_type = mouse_type.lower()

    for col in df.columns:
        mouse_type_csv = df.iloc[2, col]
        escape_success_col = df.iloc[4, col]
        column_data = df.iloc[data_row, col]
        
        

        if isinstance(column_data, str):
            try:
                column_data = float(column_data)
            except ValueError:
                pass
        
        if process_coords:
            column_data = parse.parse_coord(column_data)
        else:
            pass

        if escape:
            if escape_success_col in ['True', 'TRUE']:
                if mouse_type_csv == mouse_type:
                    true_data.append(column_data)
                else:
                    continue
            if escape_success_col in ["FALSE", "False"]:
                if mouse_type_csv == mouse_type:
                    false_data.append(column_data)
                else:
                    continue

        else:
            if mouse_type_csv == mouse_type:
                all_data.append(column_data)
            else:
                continue
    
    if escape:
        return true_data, false_data
    else:
        return all_data
    

def extract_data_col(
    file,
    nested=True,
    data_start=None,
    data_end=None,
    escape=False,
    process_coords=False,
    get_escape_index=False,
    escape_col=None,
    mouse_type=None
):
    """
    Read per-trial data from a CSV where each column is a trial.
    - Preserves interior NaNs inside trials.
    - Trims only trailing padding (empty/NaN/"nan"/"none") per column within [data_start:data_end].
    - Optionally cuts each trial at an escape index.
    - Optionally classifies trials into TRUE/FALSE (by escape_col) when escape=True.
    - Filters by mouse_type using row 2 (0-based) of each column.

    Returns:
      - if escape=True: (true_data, false_data)
      - else: all_data

    Notes:
      - `parse.parse_coord` and `find_escape_index` must exist in your environment.
    """
    df = pd.read_csv(file, header=None, low_memory=False)
    mouse_type = mouse_type.lower() if mouse_type else None

    # ----- helpers -----
    def is_filled(x):
        """Cell counts as filled if not NaN/empty/''/'nan'/'none' (case-insensitive)."""
        if pd.isna(x):
            return False
        if isinstance(x, str):
            s = x.strip().lower()
            return s not in ("", "nan", "none")
        return True

    def cell(r, c):
        try:
            return df.iat[r, c]
        except Exception:
            return np.nan

    def cast_float(x):
        if is_filled(x):
            try:
                return float(x)
            except Exception:
                return np.nan
        return np.nan

    # ----- outputs (append-only) -----
    if nested:
        all_data, true_data, false_data = [], [], []
    else:
        all_data, true_data, false_data = [], [], []

    # ----- row window -----
    data_range = slice(data_start, data_end)

    # ----- iterate trials (columns) -----
    for col in df.columns:
        # filter by mouse type (row index 2 holds the type)
        mt_csv = str(cell(2, col)).lower()
        if mouse_type and mt_csv != mouse_type:
            continue

        # take the window for this column
        col_series = df.iloc[data_range, col]

        # trim ONLY trailing empties (keep interior NaNs)
        mask = col_series.map(is_filled).to_numpy()
        if mask.any():
            last_idx = np.flatnonzero(mask)[-1]
            col_series = col_series.iloc[:last_idx + 1]
        else:
            # nothing real in this window -> empty trial
            col_series = col_series.iloc[0:0]

        # parse/cast after trimming
        if process_coords:
            # keep NaNs where cells are not filled
            column_data = [parse.parse_coord(v) if is_filled(v) else np.nan
                           for v in col_series.tolist()]
        else:
            column_data = [cast_float(v) for v in col_series.tolist()]

        # optionally cut at escape index (regardless of escape classification)
        if get_escape_index:
            # pass the series exactly as your find_escape_index expects
            escape_index = find_escape_index(column_data)
            column_data = column_data[:escape_index]

        # classify into TRUE/FALSE groups if requested
        if escape and escape_col is not None:
            esc_flag = str(cell(escape_col, col)).upper()
            if esc_flag == "TRUE":
                if nested: true_data.append(column_data)
                else:      true_data.extend(column_data)
            elif esc_flag == "FALSE":
                if nested: false_data.append(column_data)
                else:      false_data.extend(column_data)
            else:
                # skip columns without a clear TRUE/FALSE flag
                pass
        else:
            # just collect all trials
            if nested: all_data.append(column_data)
            else:      all_data.extend(column_data)

    # return per the original contract
    if escape:
        return true_data, false_data
    else:
        return all_data


def extract_angle_segments(file, data_start=None, data_end=None, escape_col=3, mouse_type=None):
    df = pd.read_csv(file, header=None, low_memory=False)
    mouse_type = mouse_type.lower()

    true_data_before = []
    true_data_middle = []
    true_data_after = []

    data_range = slice(data_start, data_end)

    for col in df.columns:
        mouse_type_csv = str(df.iloc[2, col]).lower()
        if mouse_type_csv != mouse_type:
            continue

        escape_success_col = str(df.iloc[escape_col, col]).strip().lower()
        if escape_success_col != 'true':
            continue

        # Get and sanitize column data
        column_data = df.iloc[data_range, col].tolist()
        column_data = [float(x) if pd.notna(x) and str(x).lower() != 'nan' else np.nan for x in column_data]

        # Find escape index in post-stimulus frames (after frame 150)
        escape_index_relative = find_escape_index(column_data[150:])
        escape_index = 150 + escape_index_relative

        before = column_data[:150]

        # If fewer than 60 frames between stimulus and escape, skip 'middle', shrink 'after'
        if escape_index - 150 < 60:
            middle = []  # No middle section
            after = column_data[150:escape_index]  # From stimulus to escape
        else:
            middle = column_data[150:escape_index - 60]
            after = column_data[escape_index - 60:escape_index]

        true_data_before.append(before)
        true_data_middle.append(middle)
        true_data_after.append(after)

    return true_data_before, true_data_middle, true_data_after

def build_dicts(ids, values):
            out, count = {}, {}
            for k, v in zip(ids, values):
                try:
                    k = str((k))
                    v = float(v)
                except (ValueError, TypeError):
                    continue
                out[k] = out.get(k, 0) + v
                count[k] = count.get(k, 0) + 1
            return {k: out[k] / count[k] for k in out}
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
    
def extract_data_col(file, nested=True, data_start=None, data_end=None, escape=False, process_coords=False, get_escape_index=False, escape_col=None, mouse_type=None):
    df = pd.read_csv(file, header=None, low_memory=False)
    mouse_type=mouse_type.lower()

    if nested:
        all_data = [[] for _ in range(len(df.columns))]
        true_data = [[] for _ in range(len(df.columns))]
        false_data = [[] for _ in range(len(df.columns))]
    else:
        all_data = []
        true_data, false_data = [], []

    data_range = slice(data_start, data_end)

    for col in df.columns:
        mouse_type_csv = str(df.iloc[2, col])
        column_data = df.iloc[data_range, col].tolist()

        if process_coords:
            column_data = [parse.parse_coord(coord) for coord in column_data]
        else:
            column_data = [float(x) if pd.notna(x) and x != 'nan' else np.nan for x in column_data]

        if get_escape_index and escape:
            escape_success_col = df.iloc[escape_col, col]
            if escape_success_col in ['True', 'TRUE']:
                if process_coords:
                    column_data_x = [coord for coord in column_data]
                    escape_index = find_escape_index(column_data_x)
                    column_data = column_data[:escape_index]
                else:
                    escape_index = find_escape_index(column_data)
                    column_data = column_data[:escape_index]
            else:
                escape_index = len(column_data)
                column_data = column_data[:escape_index]
                
        if get_escape_index and not escape:
            if process_coords:
                    column_data_x = [coord for coord in column_data]
                    escape_index = find_escape_index(column_data_x)
                    column_data = column_data[:escape_index]
            else:
                escape_index = len(column_data)
                column_data = column_data[:escape_index]

        if escape and escape_col is not None:
            escape_success_col = df.iloc[escape_col, col]
            if escape_success_col in ['True', 'TRUE']:
                if mouse_type_csv == mouse_type:
                    if nested:
                        true_data[col] = column_data
                    else:
                        true_data.extend(column_data)
                else:
                    pass
                
            elif escape_success_col in ['False', 'FALSE']:
                if mouse_type_csv == mouse_type:
                    if nested:
                        false_data[col] = column_data
                    else:
                        false_data.extend(column_data)
                else:
                    pass
        
        else:
            if mouse_type_csv == mouse_type:
                if nested:
                    all_data[col] = column_data
                else:
                    all_data.extend(column_data)
            else:
                pass

    if nested:
        all_data = [data for data in all_data if data]
        true_data = [data for data in true_data if data]
        false_data = [data for data in false_data if data]
    else:
        all_data = list(filter(None, all_data))
        true_data = list(filter(None, true_data))
        false_data = list(filter(None, false_data))

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
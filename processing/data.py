import pandas as pd
import numpy as np
from utils import files, parse

def bodypart_coords(tracking_db):
    return list(zip(tracking_db['x'], tracking_db['y'], tracking_db['likelihood']))

def filter_likelihood(df, pcutoff):
    df.loc[df['head_likelihood'] < pcutoff, ['head_x', 'head_y', 'head_likelihood']] = np.nan
    df.loc[df['nose_likelihood'] < pcutoff, ['nose_x', 'nose_y', 'nose_likelihood']] = np.nan

def create_df(tracking_file, stim_file, pcutoff=0.5):
    # Read tracking file
    tracking_db = files.read_tracking_file(tracking_file)
    scorer = tracking_db.columns.get_level_values(0)[0]
    stim_data = pd.read_csv(stim_file, header=None, names=['stim'], encoding='ISO-8859-1')

    # Extract head and nose coordinates and merge with stimulus data
    head_coords = pd.DataFrame(bodypart_coords(tracking_db[scorer]['head base']), columns=['head_x', 'head_y', 'head_likelihood'])
    nose_coords = pd.DataFrame(bodypart_coords(tracking_db[scorer]['nose']), columns=['nose_x', 'nose_y', 'nose_likelihood'])
    df = pd.concat([head_coords, nose_coords, stim_data], axis=1)

    # Filter out rows with likelihood < 0.5 for head base and nose coordinates
    filter_likelihood(df, pcutoff)
    df.insert(0, 'frame', range(0, len(df)))

    return df

def extract_data(df):
    head_x = df[["head_x"]].values.tolist()
    head_coords = df[['head_x', 'head_y']].values.tolist()
    nose_coords = df[['nose_x', 'nose_y']].values.tolist()
    frames = df["frame"].tolist()
    stim = df["stim"].tolist()

    return head_x, head_coords, nose_coords, frames, stim

def extract_data_rows(file, data_row=None, escape=False):
    wt_data = []
    rd1_data = []
    wt_false_data = []
    rd1_false_data = []
    wt_true_data = []
    rd1_true_data = []

    df = pd.read_csv(file, header=None, low_memory=False)

    for col in df.columns:
        mouse_type = df.iloc[1, col]
        escape_success_col = df.iloc[3, col]
        column_data = df.iloc[data_row, col]

        if isinstance(column_data, str):
            try:
                column_data = float(column_data)
            except ValueError:
                pass

        if escape:
            if escape_success_col in ['True', 'TRUE']:
                if mouse_type =="wt":
                    wt_true_data.append(column_data)
                elif mouse_type =="rd1":
                    rd1_true_data.append(column_data)
            if escape_success_col in ["FALSE", "False"]:
                if mouse_type =="wt":
                    wt_false_data.append(column_data)
                elif mouse_type =="rd1":
                    rd1_false_data.append(column_data)

        else:
            if mouse_type == 'wt':
                wt_data.append(column_data)
            elif mouse_type == "rd1":
                rd1_data.append(column_data)

    if escape:
        return wt_true_data, wt_false_data, rd1_true_data, rd1_false_data
    else:
        return wt_data, rd1_data

def extract_data_columns(file, data_start, data_end=None, escape=False):
    wt_data = []
    rd1_data = []
    wt_true = []
    wt_false = []
    rd1_true = []
    rd1_false = []

    df = pd.read_csv(file, header=None, low_memory=False)

    for col in df.columns:
        mouse_type = df.iloc[1, col]
        column_data = df.iloc[data_start:data_end, col].tolist()
            
        if escape:
            escape_success_col = df.iloc[3, col]
                
            if mouse_type == 'wt':
                if escape_success_col == 'True' or escape_success_col == "TRUE":
                        wt_true.extend(column_data)
                else:
                        wt_false.extend(column_data)
            elif mouse_type == 'rd1':
                if escape_success_col == 'True' or escape_success_col == "TRUE":
                        rd1_true.extend(column_data)
                else:
                        rd1_false.extend(column_data)
        else:
            if mouse_type == 'wt':
                    wt_data.extend(column_data)
            elif mouse_type == 'rd1':
                    rd1_data.extend(column_data)

    if escape:
        return wt_true, wt_false, rd1_true, rd1_false
    else:
        return wt_data, rd1_data

def extract_escape_data(file):
    df = pd.read_csv(file, header=None, low_memory=False)
    min_escape_frames = 5

    wt_data = [[] for _ in range(len(df.columns))]
    rd1_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        #154 so that only after stimulus is represented
        column_data = df.iloc[154:, col]
        escape_success_col = df.iloc[3, col]

        column_data = [float(x) if x != 'nan' else np.nan for x in column_data]

        escape_index = len(column_data)
        for i in range(0, len(column_data) - min_escape_frames + 1):
            if all(np.isnan(column_data[i: i + min_escape_frames])):
                escape_index = i
                break

        if escape_success_col == 'TRUE' or escape_success_col == "True":  # Only include columns with "True" in escape_success_col
            if mouse_type == 'wt':
                wt_data[col] = column_data[:escape_index]
            elif mouse_type.startswith('rd1'):
                rd1_data[col] = column_data[:escape_index]

    wt_data = [data for data in wt_data if data]
    rd1_data = [data for data in rd1_data if data]

    return wt_data, rd1_data

def extract_escape_locs(file, escape=False):
    df = pd.read_csv(file, header=None, low_memory=False)
    min_escape_frames = 5

    wt_data = [[] for _ in range(len(df.columns))]
    rd1_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        #154 so that only after stimulus is represented
        column_data = df.iloc[154:, col]
        escape_success_col = df.iloc[3, col]
        column_data = [parse.parse_coord(coord) for coord in column_data]
        column_data_x = [coord[0] for coord in column_data]
        
        escape_index = len(column_data_x)
        for i in range(0, len(column_data_x) - min_escape_frames + 1):
            if all(np.isnan(column_data_x[i: i + min_escape_frames])):
                escape_index = i
                break

        if escape:
            if escape_success_col == 'True' or escape_success_col == "TRUE":
                if mouse_type == 'wt':
                    wt_data[col] = column_data[:escape_index]
                elif mouse_type.startswith('rd1'):
                    rd1_data[col] = column_data[:escape_index]
        else:
            if escape_success_col == 'False' or escape_success_col == "FALSE":
                if mouse_type == 'wt':
                    wt_data[col] = column_data[:escape_index]
                elif mouse_type.startswith('rd1'):
                    rd1_data[col] = column_data[:escape_index]

    # Remove empty lists (due to skipped columns)
    wt_data = [data for data in wt_data if data]
    rd1_data = [data for data in rd1_data if data]

    return wt_data, rd1_data

def extract_tort_data(file, start_row=154, end_row=None):
    df = pd.read_csv(file, header=None, low_memory=False)

    wt_true_data = [[] for _ in range(len(df.columns))]
    wt_false_data = [[] for _ in range(len(df.columns))]
    rd1_true_data = [[] for _ in range(len(df.columns))]
    rd1_false_data = [[] for _ in range(len(df.columns))]

    # Create the slice for data_range from start_row to end_row
    if end_row is None:
        data_range = slice(start_row, None)
    else:
        data_range = slice(start_row, end_row)

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        column_data = df.iloc[data_range, col]
        escape_success_col = df.iloc[3, col]

        column_data = [float(x) for x in column_data if pd.notna(x) and x != 'nan']

        if escape_success_col in ['True', 'TRUE']:
            if mouse_type == 'wt':
                wt_true_data[col] = column_data
            elif mouse_type.startswith('rd1'):
                rd1_true_data[col] = column_data
        elif escape_success_col in ['False', 'FALSE']:
            if mouse_type == 'wt':
                wt_false_data[col] = column_data
            elif mouse_type.startswith('rd1'):
                rd1_false_data[col] = column_data

    wt_true_data = [data for data in wt_true_data if data]
    wt_false_data = [data for data in wt_false_data if data]
    rd1_true_data = [data for data in rd1_true_data if data]
    rd1_false_data = [data for data in rd1_false_data if data]

    return wt_true_data, wt_false_data, rd1_true_data, rd1_false_data

def extract_avg_data(file, escape=False):
    df = pd.read_csv(file, header=None, low_memory=False)

    wt_data = [[] for _ in range(len(df.columns))]
    rd1_data = [[] for _ in range(len(df.columns))]

    wt_true_data = [[] for _ in range(len(df.columns))]
    wt_false_data = [[] for _ in range(len(df.columns))]
    rd1_true_data = [[] for _ in range(len(df.columns))]
    rd1_false_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        if escape:
            column_data = df.iloc[4:, col]
            escape_success_col = df.iloc[3, col]
        else:
            column_data = df.iloc[3:, col]
        
        # Convert column data to float and filter out null values
        column_data = [float(x) if pd.notnull(x) else float('nan') for x in column_data]

        if escape:
            if str(escape_success_col).strip().lower() == 'true':
                if mouse_type == 'wt':
                    wt_true_data[col] = column_data
                elif mouse_type.startswith('rd1'):
                    rd1_true_data[col] = column_data
            elif str(escape_success_col).strip().lower() == 'false':
                if mouse_type == 'wt':
                    wt_false_data[col] = column_data
                elif mouse_type.startswith('rd1'):
                    rd1_false_data[col] = column_data
        
        else:
            if mouse_type == 'wt':
                wt_data[col] = column_data
            elif mouse_type.startswith('rd1'):
                rd1_data[col] = column_data

    # Remove empty lists
    wt_data = [data for data in wt_data if data]
    rd1_data = [data for data in rd1_data if data]

    wt_true_data = [data for data in wt_true_data if data]
    wt_false_data = [data for data in wt_false_data if data]
    rd1_true_data = [data for data in rd1_true_data if data]
    rd1_false_data = [data for data in rd1_false_data if data]

    if escape:
        return wt_true_data, wt_false_data, rd1_true_data, rd1_false_data
    else:
        return wt_data, rd1_data


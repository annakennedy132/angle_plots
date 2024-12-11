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

def extract_h5_data(df):
    head_x = df[["head_x"]].values.tolist()
    head_coords = df[['head_x', 'head_y']].values.tolist()
    nose_coords = df[['nose_x', 'nose_y']].values.tolist()
    frames = df["frame"].tolist()
    stim = df["stim"].tolist()

    return head_x, head_coords, nose_coords, frames, stim

def extract_data_rows(file, data_row=None, escape=False, process_coords=False):
    wt_data = []
    blind_data = []
    wt_false_data = []
    blind_false_data = []
    wt_true_data = []
    blind_true_data = []

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
        
        if process_coords:
            column_data = parse.parse_coord(column_data)
        else:
            pass

        if escape:
            if escape_success_col in ['True', 'TRUE']:
                if mouse_type =="wt":
                    wt_true_data.append(column_data)
                elif mouse_type != "wt":
                    blind_true_data.append(column_data)
            if escape_success_col in ["FALSE", "False"]:
                if mouse_type =="wt":
                    wt_false_data.append(column_data)
                elif mouse_type != "wt":
                    blind_false_data.append(column_data)

        else:
            if mouse_type == 'wt':
                wt_data.append(column_data)
            elif mouse_type != "wt":
                blind_data.append(column_data)

    if escape:
        return wt_true_data, wt_false_data, blind_true_data, blind_false_data
    else:
        return wt_data, blind_data

def find_escape_index(column_data, min_escape_frames=5):
    escape_index = len(column_data)
    for i in range(0, len(column_data) - min_escape_frames + 1):
        if all(np.isnan(column_data[i: i + min_escape_frames])):
            escape_index = i
            break
    return escape_index

def extract_data(file, nested=True, data_start=154, data_end=None, escape=False, process_coords=False, get_escape_index=False, escape_col=None):
    df = pd.read_csv(file, header=None, low_memory=False)

    if nested:
        wt_data = [[] for _ in range(len(df.columns))]
        blind_data = [[] for _ in range(len(df.columns))]
        wt_true_data = [[] for _ in range(len(df.columns))]
        wt_false_data = [[] for _ in range(len(df.columns))]
        blind_true_data = [[] for _ in range(len(df.columns))]
        blind_false_data = [[] for _ in range(len(df.columns))]
    else:
        wt_data, blind_data = [], []
        wt_true_data, wt_false_data = [], []
        blind_true_data, blind_false_data = [], []

    data_range = slice(data_start, data_end)

    for col in df.columns:
        mouse_type = df.iloc[1, col]
        column_data = df.iloc[data_range, col].tolist()

        if process_coords:
            column_data = [parse.parse_coord(coord) for coord in column_data]
        else:
            column_data = [float(x) if pd.notna(x) and x != 'nan' else np.nan for x in column_data]

        if get_escape_index:
            escape_success_col = df.iloc[escape_col, col]
            if escape_success_col in ['True', 'TRUE']:
                if process_coords:
                    column_data_x = [coord[0] for coord in column_data]
                    escape_index = find_escape_index(column_data_x)
                    column_data = column_data[:escape_index]
                else:
                    escape_index = find_escape_index(column_data)
                    column_data = column_data[:escape_index]
            else:
                escape_index = len(column_data)
                column_data = column_data[:escape_index]

        if escape and escape_col is not None:
            escape_success_col = df.iloc[escape_col, col]
            if escape_success_col in ['True', 'TRUE']:
                if mouse_type == 'wt':
                    if nested:
                        wt_true_data[col] = column_data
                    else:
                        wt_true_data.extend(column_data)
                elif mouse_type != "wt":
                    if nested:
                        blind_true_data[col] = column_data
                    else:
                        blind_true_data.extend(column_data)
            elif escape_success_col in ['False', 'FALSE']:
                if mouse_type == 'wt':
                    if nested:
                        wt_false_data[col] = column_data
                    else:
                        wt_false_data.extend(column_data)
                elif mouse_type != "wt":
                    if nested:
                        blind_false_data[col] = column_data
                    else:
                        blind_false_data.extend(column_data)
        else:
            if mouse_type == 'wt':
                if nested:
                    wt_data[col] = column_data
                else:
                    wt_data.extend(column_data)
            elif mouse_type != "wt":
                if nested:
                    blind_data[col] = column_data
                else:
                    blind_data.extend(column_data)

    if nested:
        wt_data = [data for data in wt_data if data]
        blind_data = [data for data in blind_data if data]
        wt_true_data = [data for data in wt_true_data if data]
        wt_false_data = [data for data in wt_false_data if data]
        blind_true_data = [data for data in blind_true_data if data]
        blind_false_data = [data for data in blind_false_data if data]
    else:
        wt_data = list(filter(None, wt_data))
        blind_data = list(filter(None, blind_data))
        wt_true_data = list(filter(None, wt_true_data))
        wt_false_data = list(filter(None, wt_false_data))
        blind_true_data = list(filter(None, blind_true_data))
        blind_false_data = list(filter(None, blind_false_data))

    if escape:
        return wt_true_data, wt_false_data, blind_true_data, blind_false_data
    else:
        return wt_data, blind_data
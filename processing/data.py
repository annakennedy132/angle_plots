import pandas as pd
import numpy as np
from utils import files, parse
from scipy.interpolate import interp1d

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
                if escape_success_col == 'True':
                        wt_true.extend(column_data)
                else:
                        wt_false.extend(column_data)
            elif mouse_type == 'rd1':
                if escape_success_col == 'True':
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
        
def extract_data_rows(file, row, bool_row):

    wt_data = []
    rd1_data = []
    df = pd.read_csv(file, header=None, low_memory=False)

    for col in df.columns:
        mouse_type = df.iloc[1, col]
        column_data = df.iloc[row, col]

        # Convert strings to floats for numeric rows
        if row != bool_row:
            if isinstance(column_data, str):
                try:
                    column_data = float(column_data)
                except ValueError:
                    pass

            else:
                if np.isnan(column_data):
                    pass
                elif column_data.strip().lower() == 'true':
                    column_data = True
                elif column_data.strip().lower() == 'false':
                        column_data = False
                

        if mouse_type == "wt":
                wt_data.append(column_data)
        else:
                rd1_data.append(column_data)
        
    return wt_data, rd1_data
    
def extract_data_rows_esc(file, row):
        wt_false_data = []
        rd1_false_data = []
        wt_true_data = []
        rd1_true_data = []

        df = pd.read_csv(file, header=None, low_memory=False)

        for col in df.columns:
            mouse_type = df.iloc[1, col]
            esc_success = df.iloc[3, col]
            column_data = df.iloc[row, col]

            if isinstance(column_data, str):
                    try:
                        column_data = float(column_data)
                    except ValueError:
                        pass

            if mouse_type == "wt" and esc_success == "True":
                wt_true_data.append(column_data)
            elif mouse_type == "wt" and esc_success == "False":
                wt_false_data.append(column_data)
            elif mouse_type == "rd1" and esc_success == "True":
                rd1_true_data.append(column_data)
            elif mouse_type == "rd1" and esc_success == "False":
                rd1_false_data.append(column_data)
        
        return wt_true_data, wt_false_data, rd1_true_data, rd1_false_data
    
def extract_escape_data(file):
    df = pd.read_csv(file, header=None, low_memory=False)

    wt_data = [[] for _ in range(len(df.columns))]
    rd1_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        #154 so that only after stimulus is represented
        column_data = df.iloc[154:, col]
        escape_success_col = df.iloc[3, col]

        column_data = [float(x) if x != 'nan' else np.nan for x in column_data]

        # Find the index of the first NaN value
        nan_index = next((i for i, x in enumerate(column_data) if pd.isna(x)), len(column_data))

        if escape_success_col == 'TRUE' or escape_success_col == "True":  # Only include columns with "True" in escape_success_col
            if mouse_type == 'wt':
                wt_data[col] = column_data[:nan_index]
            elif mouse_type.startswith('rd1'):
                rd1_data[col] = column_data[:nan_index]

    wt_data = [data for data in wt_data if data]
    rd1_data = [data for data in rd1_data if data]

    return wt_data, rd1_data

def extract_escape_locs(file, escape=False):
    df = pd.read_csv(file, header=None, low_memory=False)

    wt_data = [[] for _ in range(len(df.columns))]
    rd1_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        #154 so that only after stimulus is represented
        column_data = df.iloc[154:, col]
        escape_success_col = df.iloc[3, col]
        column_data = [parse.parse_coord(coord) for coord in column_data]
        nan_index = next((i for i, (x, y) in enumerate(column_data) if np.isnan(x) or np.isnan(y)), len(column_data))

        if escape:
            if escape_success_col == 'True':
                if mouse_type == 'wt':
                    wt_data[col] = column_data[:nan_index]
                elif mouse_type.startswith('rd1'):
                    rd1_data[col] = column_data[:nan_index]
        else:
            if escape_success_col == 'False':
                if mouse_type == 'wt':
                    wt_data[col] = column_data[:nan_index]
                elif mouse_type.startswith('rd1'):
                    rd1_data[col] = column_data[:nan_index]

    # Remove empty lists (due to skipped columns)
    wt_data = [data for data in wt_data if data]
    rd1_data = [data for data in rd1_data if data]

    return wt_data, rd1_data

def extract_tort_data(file):
    df = pd.read_csv(file, header=None, low_memory=False)

    wt_true_data = [[] for _ in range(len(df.columns))]
    wt_false_data = [[] for _ in range(len(df.columns))]
    rd1_true_data = [[] for _ in range(len(df.columns))]
    rd1_false_data = [[] for _ in range(len(df.columns))]

    for col in range(len(df.columns)):
        mouse_type = df.iloc[1, col]
        column_data = df.iloc[5:, col]
        escape_success_col = df.iloc[3, col]

        column_data = [float(x) for x in column_data if pd.notna(x) and x != 'nan']

        if escape_success_col == 'True':  # Only include columns with "True" in escape_success_col
            if mouse_type == 'wt':
                wt_true_data[col] = column_data
            elif mouse_type.startswith('rd1'):
                rd1_true_data[col] = column_data
        if escape_success_col == 'False':  # Only include columns with "True" in escape_success_col
            if mouse_type == 'wt':
                wt_false_data[col] = column_data
            elif mouse_type.startswith('rd1'):
                rd1_false_data[col] = column_data

    wt_true_data = [data for data in wt_true_data if data]
    wt_false_data = [data for data in wt_false_data if data]
    rd1_true_data = [data for data in rd1_true_data if data]
    rd1_false_data = [data for data in rd1_false_data if data]

    return wt_true_data, wt_false_data, rd1_true_data, rd1_false_data
    
def normalize_length(coord_sets):
    max_length = max([len(coord_set) for coord_set in coord_sets])
        
    normalized_sets = []
    for coord_set in coord_sets:
            # Interpolate or resample the coordinates to match avg_length
        x_interp = interp1d(np.linspace(0, 1, len(coord_set)), [coord[0] for coord in coord_set])
        y_interp = interp1d(np.linspace(0, 1, len(coord_set)), [coord[1] for coord in coord_set])
        normalized_x = x_interp(np.linspace(0, 1, int(max_length)))
        normalized_y = y_interp(np.linspace(0, 1, int(max_length)))
        normalized_sets.append(list(zip(normalized_x, normalized_y)))
        
    return normalized_sets

def calculate_mean_coords(coords):
    coord_parsed = [parse.parse_coord(coord) for coord in coords]
    coords = [coord for coord in coord_parsed if coord is not any(np.isnan(coord))]
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    mean_x = np.nanmean(x_coords)
    mean_y = np.nanmean(y_coords)
    return mean_x, mean_y


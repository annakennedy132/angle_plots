import pandas as pd
import numpy as np
from utils import files

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

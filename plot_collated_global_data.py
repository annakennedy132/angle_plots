import sys
import os
from utils import parse
from processing import plots
import pandas as pd
import matplotlib.pyplot as plt
from utils import files

def run():

    FOLDER = parse_args()

    figs = []

    angles_file = next((os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith("angles.csv")), None)
    locs_file = next((os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith("locs.csv")), None)
    dist_file = next((os.path.join(FOLDER, file) for file in os.listdir(FOLDER) if file.endswith("distances.csv")), None)
    
    global_wt_angles, global_rd1_angles, global_rd1_3_angles = extract_data(angles_file)
    global_wt_locs, global_rd1_locs, global_rd1_3_locs = extract_data(locs_file)
    global_wt_distances, global_rd1_distances, global_rd1_3_distances = extract_data(dist_file)

    global_all_rd1_angles = global_rd1_angles + global_rd1_3_angles
    global_all_rd1_locs = global_rd1_locs + global_rd1_3_locs
    global_all_rd1_3_distances = global_rd1_distances + global_rd1_3_distances

    wt_rd1_coords_fig, (wt_ax, rd1_ax) = plt.subplots(1, 2, figsize=(12, 5))
    plt.suptitle(f'Heatmaps of Coords Comparing WT and RD1 Mice')
    figs.append(wt_rd1_coords_fig)
    wt_ax.set_title('WT Mice')
    plots.plot_string_coords(wt_rd1_coords_fig, wt_ax, global_wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=5000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
    rd1_ax.set_title('RD1 Mice')
    plots.plot_string_coords(wt_rd1_coords_fig, rd1_ax, global_all_rd1_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=5000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)

    wt_rd1_rd1_3_coords_fig, (wt_ax, rd1_ax, rd1_3_ax) = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle(f'Heatmaps of Coords Comparing WT, RD1 and RD1_3 Mice')
    figs.append(wt_rd1_rd1_3_coords_fig)
    wt_ax.set_title('WT Mice')
    plots.plot_string_coords(wt_rd1_rd1_3_coords_fig, wt_ax, global_wt_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=5000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
    rd1_ax.set_title('RD1 Mice')
    plots.plot_string_coords(wt_rd1_rd1_3_coords_fig, rd1_ax, global_rd1_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=5000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
    rd1_3_ax.set_title('RD1_3 Mice')
    plots.plot_string_coords(wt_rd1_rd1_3_coords_fig, rd1_3_ax, global_rd1_3_locs, xlabel="x", ylabel="y", gridsize=50, vmin=0, vmax=5000, xmin=90, xmax=790, ymin=670, ymax=80, show=False, close=True)
    
    wt_rd1_polar_fig, (wt_ax, rd1_ax) = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(projection='polar'))
    wt_rd1_polar_fig.suptitle(f'Polar Plot Comparing Facing Angles WT and RD1 Mice')
    figs.append(wt_rd1_polar_fig)
    wt_ax.set_title('WT Mice')
    plots.plot_polar_chart(wt_rd1_polar_fig, wt_ax, global_wt_angles, bins=36, direction=1, zero="E", show=False, close=True)
    rd1_ax.set_title('RD1 Mice')
    plots.plot_polar_chart(wt_rd1_polar_fig, rd1_ax, global_all_rd1_angles, bins=36, direction=1, zero="E", show=False, close=True)

    wt_rd1_rd1_3_polar_fig, (wt_ax, rd1_ax, rd1_3_ax) = plt.subplots(1, 3, figsize=(12, 5), subplot_kw=dict(projection='polar'))
    wt_rd1_rd1_3_polar_fig.suptitle(f'Polar Plot Comparing Facing Angles of WT, RD1 and RD1_3 Mice')
    figs.append(wt_rd1_rd1_3_polar_fig)
    wt_ax.set_title('WT Mice')
    plots.plot_polar_chart(wt_rd1_rd1_3_polar_fig, wt_ax, global_wt_angles, bins=36, direction=1, zero="E", show=False, close=True)
    rd1_ax.set_title('RD1 Mouse')
    plots.plot_polar_chart(wt_rd1_rd1_3_polar_fig, rd1_ax, global_rd1_angles, bins=36, direction=1, zero="E", show=False, close=True)
    rd1_3_ax.set_title('RD1_3 Mice')
    plots.plot_polar_chart(wt_rd1_rd1_3_polar_fig, rd1_3_ax, global_rd1_3_angles, bins=36, direction=1, zero="E", show=False, close=True)

    if len(figs) > 0:
        files.save_report(figs, FOLDER)


def extract_data(file):
    # Initialize lists to store data for each mouse type
    wt_data = []
    rd1_data = []
    rd1_3_data = []

    # Read the data file
    df = pd.read_csv(file, header=None, low_memory=False)

    # Iterate through each column
    for col in df.columns:
        # Get the mouse type from the second row of the current column
        mouse_type = df.iloc[1, col]
        
        # Extract data from the current column, skipping the first 3 rows
        column_data = df.iloc[3:, col].tolist()
        
        # Append the data to the corresponding mouse type list
        if mouse_type == 'wt':
            wt_data.extend(column_data)
        elif mouse_type == 'rd1':
            rd1_data.extend(column_data)
        elif mouse_type == 'rd1_3':
            rd1_3_data.extend(column_data)

    return wt_data, rd1_data, rd1_3_data

def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("Folder must be specified")
    
    elif len(sys.argv) == 2:
        
        folder = parse.folder(sys.argv[1])
        
    else:
        
        raise KeyError("Too many input arguments")
    
    return folder

if __name__ == "__main__":
    run()
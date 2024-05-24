import sys
import os
from rich.progress import track

from processing import collation
from utils import files, parse

def run():

    FOLDER, INDEX_FILE = parse_args()

    data_folders = files.get_data_folders(FOLDER)
    meta_data = {}
    angles_data = {}
    locs_data = {}
    dist_data = {}
    during_angles_data = {}
    after_angles_data = {}
    prev_esc_locs_data = {}
    average_angles_data = {}
    average_dist_data = {}
    escape_stats_data = {}

    if INDEX_FILE is not None:

        data_folders = files.keep_indexed_folders(data_folders, INDEX_FILE)
        output_folder = os.path.dirname(INDEX_FILE)

    else:

        output_folder = FOLDER
        print("loading all folders...")

    for folder_name, path in track(data_folders.items(), description="Collating events data..."):

        if collation.check_for_events(path):

            try:

                meta, angles, locs, distances, average_angles, average_distances, during_angles, after_angles, prev_esc_locs, escape_stats = collation.read_event_data(path, folder_name, INDEX_FILE)

                meta_data.update(meta)
                angles_data.update(angles)
                locs_data.update(locs)
                dist_data.update(distances)
                during_angles_data.update(during_angles)
                after_angles_data.update(after_angles)
                prev_esc_locs_data.update(prev_esc_locs)
                escape_stats_data.update(escape_stats)
                average_angles_data[folder_name] = average_angles
                average_dist_data[folder_name] = average_distances

                print(f"Successfuly collated {folder_name}")

            except:

                print(f"Error with {folder_name}")

        else:
            print(f"No events for {folder_name}")
        
        collated_meta_path = os.path.join(output_folder, "collated_escape-stats.csv")
        collation.write_collated_event_data(collated_meta_path, meta_data)

        collated_angles_path = os.path.join(output_folder, "collated_event_angles.csv")
        collation.write_collated_event_data(collated_angles_path, angles_data)

        collated_locs_path = os.path.join(output_folder, "collated_event_locs.csv")
        collation.write_collated_event_data(collated_locs_path, locs_data)

        collated_dist_path = os.path.join(output_folder, "collated_event_distances.csv")
        collation.write_collated_event_data(collated_dist_path, dist_data)

        collated_avg_angles_path = os.path.join(output_folder, "collated_avg_angles.csv")
        collation.write_collated_event_data(collated_avg_angles_path, average_angles_data)

        collated_avg_dist_path = os.path.join(output_folder, "collated_avg_distances.csv")
        collation.write_collated_event_data(collated_avg_dist_path, average_dist_data)
        
        collated_during_path = os.path.join(output_folder, "collated_during_angles.csv")
        collation.write_collated_event_data(collated_during_path, during_angles_data)

        collated_after_path = os.path.join(output_folder, "collated_after_angles.csv")
        collation.write_collated_event_data(collated_after_path, after_angles_data)

        collated_after_path = os.path.join(output_folder, "collated_prev_esc_locs.csv")
        collation.write_collated_event_data(collated_after_path, prev_esc_locs_data)

        collated_escape_path = os.path.join(output_folder, "collated_escape_success.csv")
        collation.write_collated_event_data(collated_escape_path, escape_stats_data)

    collation.create_average_csv(collated_avg_angles_path)
    collation.create_average_csv_escape_success(collated_angles_path)
    collation.create_average_csv(collated_avg_dist_path)
    collation.create_average_csv_escape_success(collated_dist_path)

def parse_args():

    if len(sys.argv) == 1:
        
        raise KeyError("Folder and index file must be specified")
    
    elif len(sys.argv) == 2:
        
        folder = parse.folder(sys.argv[1])
        index_file = None
        
    elif len(sys.argv) == 3:
        
        folder = parse.folder(sys.argv[1])
        index_file = parse.text_file(sys.argv[2])      
             
    else:
        
        raise KeyError("Too many input arguments")
    
    return folder, index_file

if __name__ == "__main__":
    run()


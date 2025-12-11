import os
import yaml
import pandas as pd
from datetime import datetime
import csv
from matplotlib.backends.backend_pdf import PdfPages

def load_config():
    
    with open('config.yaml') as config:
        settings = yaml.load(config.read(), Loader=yaml.Loader)
        
    return settings
def create_folder(base_folder, name, append_date=True):
    
    if append_date:
        now = datetime.now()
        time = now.strftime("_%Y-%m-%d_%H-%M-%S")
        name = name + time
    
    new_folder = os.path.join(base_folder, name)
        
    os.mkdir(new_folder)
    return new_folder

def create_csv(data, filepath, columns=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    max_len = max(map(len, data))
    rows = zip(*[list(col) + [None]*(max_len - len(col)) for col in data])

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if columns: writer.writerow(columns)
        writer.writerows([[v for v in row] for row in rows])

def create_trial_csv(trials_by_type, event_ids_by_type, mouse_types, filepath):

    cols = []       # each element is a full column (mouse_type, event_id, trial frames...)
    col_names = []  # names for each column

    for mt, trials, ids in zip(mouse_types, trials_by_type, event_ids_by_type):
        for trial, eid in zip(trials, ids):
            col = [mt, eid] + list(trial)
            cols.append(col)

    create_csv(cols, filepath, columns=col_names)
        
def save_report(figs, base_path, title=None):
    if title:
        report_path = f"{base_path}_{title}_report.pdf"
    else:
        report_path = f"{base_path}_report.pdf"

    with PdfPages(report_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)

def save_images(imgs, base_path, title=None):

    folder = os.path.join(base_path + "_imgs")
    os.makedirs(folder, exist_ok=True)

    for i, fig in enumerate(imgs):
        if title:
            img_path = os.path.join(folder, f"{title}_figure_{i + 1}.png")
        else:
            img_path = os.path.join(folder, f"figure_{i + 1}.png")
        
        fig.savefig(img_path, format="png")
        
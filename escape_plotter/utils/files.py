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

def create_csv(list, filename):
    
    with open(filename, 'w', newline='') as file:
        
        writer = csv.writer(file)
        
        for row in list:
            if row[1] is None:
                row = [row[0], "None", "None"]
            writer.writerow(row)

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
        
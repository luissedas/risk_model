from tkinter import DOTBOX
import pandas as pd
import pickle
import os
import json
from shutil import copy2

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = config["output_folder_path"]
output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']


def store_model_into_pickle():
    to_copy = ["latestscore.txt", "ingestedfiles.txt", "trainedmodel.pkl"]

    for f in to_copy:
        if f == "ingestedfiles.txt":
            origin = os.path.join(output_folder_path, f)
        else:
            origin = os.path.join(output_model_path, f)

        dest = os.path.join(prod_deployment_path, f)

        copy2(origin, dest)


if __name__ == '__main__':
    store_model_into_pickle()

import pandas as pd
import os
import json
from datetime import datetime


# Load config.json and get input and output paths
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


# Function for data ingestion
def merge_multiple_dataframe():
    # check for datasets, compile them together, and write to an output file
    files_dir = os.listdir(input_folder_path)

    files = [os.path.join(input_folder_path, f) for f in files_dir]

    dfs = (pd.read_csv(f) for f in files)

    df_complete = pd.concat(dfs, ignore_index=True)

    df_complete.drop_duplicates(inplace=True)

    df_complete.to_csv(os.path.join(
        output_folder_path, 'finaldata.csv'), index=False)

    # log the input files to create the previous file
    with open(os.path.join(output_folder_path, 'ingestedfiles.txt'), 'w') as f:
        for file_name in files_dir:
            f.write(file_name + "\n")


if __name__ == '__main__':
    merge_multiple_dataframe()

from time import time
import pandas as pd
import timeit
import os
import json
import pickle
import subprocess
import sys
import setup

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

prod_deployment_path = config['prod_deployment_path']
test_data_path = config['test_data_path']


def model_predictions(dataframe):

    with open(os.path.join(prod_deployment_path, 'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(dataframe)

    return y_pred.tolist()


def dataframe_summary():
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    summary = []

    for col in setup.independent_cols:
        summary.append([col, 'mean', df[col].mean()])
        summary.append([col, 'median', df[col].median()])
        summary.append([col, 'std', df[col].std()])

    return summary


def missing_data():
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    total = df.shape[0]
    missing = []
    for col in df.columns:
        counter = df[col].isna().sum()
        missing.append([col, str(int(counter/total*100))+"%"])
    return str(missing)


def execution_time():
    steps = ['ingestion.py', 'training.py']
    times = []
    for step in steps:
        start = timeit.default_timer()
        os.system("python %s" % step)
        length = timeit.default_timer() - start
        times.append([step, length])
    return str(times)


def outdated_packages_list():
    outdated_packages = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()

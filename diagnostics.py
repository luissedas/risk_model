
from time import time
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess
import sys

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

model_path = os.path.join(config['prod_deployment_path']) 
test_data_path = os.path.join(config['test_data_path']) 

##################Function to get model predictions
def model_predictions(dataset_path='testdata.csv'):
    #read the deployed model and a test dataset, calculate predictions
    df = pd.read_csv(os.path.join(test_data_path, dataset_path))
    y = df['exited'].values.reshape(-1,1)
    X = df[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)

    with open(os.path.join(model_path,'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)
    
    return y_pred, y

##################Function to get summary statistics
def dataframe_summary():

    #calculate summary statistics here
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
        ]
    summary = []
    for col in columns:
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


##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    steps = ['ingestion.py','training.py']
    times = []
    for step in steps:
        start = timeit.default_timer()
        os.system("python3 %s" % step)
        length = timeit.default_timer() - start
        times.append([step, length])
    return str(times)

##################Function to check dependencies
def outdated_packages_list():
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated']).decode(sys.stdout.encoding)
    
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    execution_time()
    outdated_packages_list()





    

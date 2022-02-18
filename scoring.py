from cgi import test
from math import prod
from ntpath import join
from sys import path
import pandas as pd
import pickle
import os
from sklearn import metrics
import json
import setup


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = config['test_data_path']
output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
# prod_deployment_path = config['prod_deployment_path']


def f1_calculation(model_path, file_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

    data = pd.read_csv(file_path)
    X, y = setup.split_features_target(data)

    y_pred = model.predict(X)
    f1 = metrics.f1_score(y, y_pred)
    return f1


# Function for model scoring
def score_model():
    f1_score = f1_calculation(
        os.path.join(output_model_path, "trainedmodel.pkl"),
        os.path.join(test_data_path, "testdata.csv")
    )

    with open(os.path.join(output_model_path, 'latestscore.txt'), 'w') as f:
        f.write(str(f1_score) + "\n")

    return f1_score


if __name__ == '__main__':
    score_model()

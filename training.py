import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json
import setup

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
output_model_path = config['output_model_path']


# Function for training the model
def train_model():
    data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    
    x, y = setup.split_features_target(data)

    # use this logistic regression for training
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='auto', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    model = logit.fit(x, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    pickle.dump(model, open(os.path.join(
        output_model_path, 'trainedmodel.pkl'), 'wb'))


if __name__ == '__main__':
    train_model()

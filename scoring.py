from ntpath import join
from sys import path
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 
    
test_data_path = os.path.join(config['test_data_path']) 
output_folder_path = os.path.join(config['output_folder_path'])
output_model_path = os.path.join(config['prod_deployment_path']) 


#################Function for model scoring
def score_model(test=True):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    if test == True:
        data = pd.read_csv(os.path.join(test_data_path,'testdata.csv'))
    else:
        data = pd.read_csv(os.path.join(output_folder_path,'finaldata.csv'))

    y = data['exited'].values.reshape(-1,1)
    X = data[['lastmonth_activity','lastyear_activity','number_of_employees']].values.reshape(-1,3)

    with open(os.path.join(output_model_path,'trainedmodel.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)
    f1_score = metrics.f1_score(y, y_pred)

    with open(os.path.join(output_model_path,'latestscore.txt'), 'w') as f:
        f.write(str(f1_score) + "\n")
    
    return f1_score


if __name__ == '__main__':
    score_model()
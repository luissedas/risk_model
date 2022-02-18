from tkinter import DOTBOX
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from shutil import copy2

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_folder_path = config["output_folder_path"]

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
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


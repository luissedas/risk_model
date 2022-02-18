from diagnostics import model_predictions
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
from diagnostics import model_predictions, dataframe_summary, missing_data, outdated_packages_list, execution_time
from scoring import score_model


######################Set up variabsles for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    path = request.json.get('dataset_path')
    y_pred, _ = model_predictions(path)
    print(str(y_pred))
    return str(y_pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    result = score_model()
    return str(result)

#######################Summary Statistics Endpoint
@app.route("/summary", methods=['GET','OPTIONS'])
def summary():        
    #check means, medians, and modes for each column
    summary = dataframe_summary()
    return str(summary)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    #check timing and percent NA values   
    exc_time = execution_time()
    miss_data = missing_data()
    outdd_dep = outdated_packages_list()     
    return str(
        "execution_time:" + exc_time + 
        "\nmissing_data;"+ miss_data + 
        "\noutdated_packages:" + outdd_dep)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
    extra_files=['config.json']

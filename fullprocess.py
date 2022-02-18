from calendar import different_locale
from doctest import OutputChecker
from math import prod
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import json
import os
from scoring import f1_calculation


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = config['prod_deployment_path']
model_path = config['output_model_path']
output_folder_path = config['output_folder_path']


# Check and read new data
# first, read ingestedfiles.txt
used_files = []
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt')) as f:
    for file in f:
        used_files.append(file)


# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_files = os.listdir(input_folder_path)
new_files = [i for i in current_files if i not in used_files]
new_files_bool = len(new_files) > 0


# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print("No new data, stopping process")
    exit(0)

# Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
ingestion.merge_multiple_dataframe()

with open(os.path.join(prod_deployment_path, 'latestscore.txt')) as f:
    f1_deployed = float(f.read())

f1_latest = f1_calculation(
    os.path.join(prod_deployment_path, "trainedmodel.pkl"),
    os.path.join(output_folder_path, "finaldata.csv"),
)

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here
if f1_latest >= f1_deployed:
    print(
        f"No model drift {round(f1_latest,2)}>{round(f1_deployed,2)}, stopping process")
    exit(0)

os.system('python training.py')
os.system('python scoring.py')

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
os.system('python deployment.py')

##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
os.system('python apicalls.py')
os.system('python reporting.py')

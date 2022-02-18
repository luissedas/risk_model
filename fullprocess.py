

from calendar import different_locale
import training
import scoring
import deployment
import diagnostics
import reporting
import ingestion
import json
import os


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path']) 


##################Check and read new data
#first, read ingestedfiles.txt
used_files = []
with open(os.path.join(prod_deployment_path, 'ingestedfiles.txt')) as f:
    for file in f:
        used_files.append(file)


#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_files = os.listdir(input_folder_path)
new_files = [i for i in current_files if i not in used_files]
new_files_bool = len(new_files)>0


##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    print("No new nada, stopping process")
    exit(0)

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
ingestion.merge_multiple_dataframe()
scoring.score_model(test=False)

with open(os.path.join(prod_deployment_path, 'latestscore.txt')) as f:
    f1_deployed = float(f.read())

with open(os.path.join(model_path, 'latestscore.txt')) as f:
    f1_latest = float(f.read())

##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if f1_latest >= f1_deployed:
    print("No model drift, stopping process")
    exit(0)

training.train_model()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()


##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model

os.system('python3 apicalls.py')
os.system('python3 reporting.py')






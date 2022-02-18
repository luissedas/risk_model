import requests
import json
import os

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

# Call each API endpoint and store the responses
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

response1 = requests.post(
    "%s/prediction" % URL, json={"dataset_path": "testdata/testdata.csv"}, headers=headers).text
response2 = requests.get(URL+'/scoring').text
response3 = requests.get(URL+'/summary').text
response4 = requests.get(URL+'/diagnostics').text

# combine all API responses
responses = response1 + '\n' + response2 + '\n' + response3 + '\n' + response4

# write the responses to your workspace
with open('config.json', 'r') as f:
    config = json.load(f)
model_path = os.path.join(config['output_model_path'])

with open(os.path.join(model_path, "apireturns.txt"), "w") as f:
    f.write(responses)

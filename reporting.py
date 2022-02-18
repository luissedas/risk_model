from cgi import test
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
from diagnostics import model_predictions
import setup

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

test_data_path = config['test_data_path']
model_path = config['output_model_path']


def score_model():

    data = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    x, y = setup.split_features_target(data)

    y_pred = model_predictions(x)

    conf_mat = metrics.confusion_matrix(y, y_pred)

    _, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j],
                    va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(model_path, "confusionmatrix.png"))


if __name__ == '__main__':
    score_model()

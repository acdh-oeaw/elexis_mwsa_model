import json
import logging
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from pandas.core.common import SettingWithCopyWarning
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, plot_confusion_matrix
import seaborn as sns

from mwsa_model.service.data_loader import DataLoader
from mwsa_model.service.model_trainer import MwsaModelTrainer


def calculate_metrics(reference_labels, predicted_labels):
    f1 = f1_score(reference_labels, predicted_labels, average='weighted')
    precision = precision_score(reference_labels, predicted_labels, average='weighted')
    recall = recall_score(reference_labels, predicted_labels, average='weighted')

    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


warnings.filterwarnings(
    action='ignore',
    category=SettingWithCopyWarning,
    module=r'.*'
)

logger = logging.getLogger('evaluate')
logger.setLevel(logging.INFO)

if len(sys.argv) != 5:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python evaluate.py language model_name test_data_filename metrics_filename prediction_filename')

model = sys.argv[1]
metrics_file = sys.argv[2]
prediction_file = sys.argv[3]
lang = sys.argv[4]

config_file = lang + '_params.yaml'
with open(config_file, 'r') as fd:
    params = yaml.safe_load(fd)

dataset = params['data']['dataset']
version = params['data']['version']

output_dir = 'mwsa_model/output/'
model_output_dir = output_dir + 'models/'
metrics_output_dir = output_dir + 'metrics/'
predictions_output_dir = output_dir + 'predictions/'
plot_dir = output_dir + 'plots/'
Path(plot_dir).mkdir(parents=True, exist_ok=True)

model_file_name = model_output_dir + model
pipeline_file_name = 'mwsa_model/output/pipeline/pipeline_' + dataset + '.pkl'
with open(pipeline_file_name, 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

with open(model_file_name, 'rb') as model_file:
    model = pickle.load(model_file)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    data_loader = DataLoader()
    testdata = data_loader.load('mwsa/test', dataset, testdata=True, version=version)

    model_trainer = MwsaModelTrainer()

    preprocessed = pipeline.transform(testdata)

    reference_labels = data_loader.load('mwsa/reference', dataset, version=version)['relation']

    predicted = model.predict(preprocessed)
    predicted_series = pd.Series(predicted)

metrics = calculate_metrics(reference_labels, predicted_series)

with open(metrics_output_dir + metrics_file, 'w+') as fd:
    fd.write(json.dumps(metrics))

#############################
#### Export Result data #####
#############################

testdata['relation'] = predicted_series
testdata_df = testdata[['word', 'pos', 'def1', 'def2', 'relation']]

testdata_df.to_csv(predictions_output_dir + prediction_file, sep='\t', index=False)

#############################
### Plot Confusion Matrix ###
#############################

confusion_matrix = confusion_matrix(reference_labels, predicted_series, labels=model.classes_)
cm_plot = plot_confusion_matrix(model.best_estimator_, preprocessed, reference_labels,
                                display_labels=model.classes_,
                                cmap=plt.cm.Blues)
plt.savefig(plot_dir+"confusion_matrix_"+lang+".png", dpi=120)
plt.close()

##########################
### Feature importance ###
##########################

best_model = model.best_estimator_
logger.info(best_model.feature_importances_)
importances = best_model.feature_importances_
labels = preprocessed.columns
logger.info(preprocessed.columns)
feature_df = pd.DataFrame(list(zip(labels, importances)), columns=["feature", "importance"])
feature_df = feature_df.sort_values(by='importance', ascending=False)

# image formatting
axis_fs = 18  # fontsize
title_fs = 22  # fontsize
sns.set(style="whitegrid")

ax = sns.barplot(x="importance", y="feature", data=feature_df)
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)  # ylabel
ax.set_title('Random forest\nfeature importance', fontsize=title_fs)

plt.tight_layout()
plt.savefig(plot_dir+"feature_importance_"+lang+".png", dpi=120)
plt.close()

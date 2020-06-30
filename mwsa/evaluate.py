import logging
import pickle
import sys

import pandas as pd
from sklearn.metrics import f1_score

from mwsa.service.data_loader import DataLoader

logger = logging.getLogger('evaluate')
logger.setLevel(logging.INFO)

if len(sys.argv) != 5:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python evaluate.py model_name test_data_filename metrics_filename prediction_filename')

model = sys.argv[1]
test_data_file = sys.argv[2]
metrics_file = sys.argv[3]
prediction_file = sys.argv[4]

output_dir = 'mwsa/output/'
model_output_dir = output_dir + 'models/'
metrics_output_dir = output_dir + 'metrics/'
predictions_output_dir = output_dir + 'predictions/'

file = 'mwsa/output/models/' + model
with open(file, 'rb') as model_file:
    model = pickle.load(model_file)

data_loader = DataLoader()
testdata = data_loader.load('data/test', test_data_file, testdata=True)
reference_labels = data_loader.load('data/reference_data', test_data_file)['relation']

predicted = model.predict(testdata)
predicted_series = pd.Series(predicted)
testdata['relation'] = predicted_series

f1 = f1_score(reference_labels, testdata['relation'], average='weighted')

testdata_df = testdata[['word', 'pos', 'def1', 'def2', 'relation']]

with open(metrics_output_dir+metrics_file, 'w+') as fd:
    fd.write('{:4f}\n'.format(f1))

testdata_df.to_csv(predictions_output_dir + prediction_file, sep='\t', index=False)

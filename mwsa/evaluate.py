import logging
import pickle
import sys

import pandas as pd
from sklearn.metrics import f1_score

from mwsa.service.data_loader import DataLoader

logger = logging.getLogger('evaluate')

file = 'mwsa/output/models/'+sys.argv[1]
with open(file, 'rb') as model_file:
    model = pickle.load(model_file)

data_loader = DataLoader()
testdata = data_loader.load('data/test', sys.argv[2],testdata=True)
reference_labels = data_loader.load('data/reference_data', sys.argv[2])['relation']

predicted = model.predict(testdata)
predicted_series= pd.Series(predicted)
testdata['relation'] = predicted_series

f1 = f1_score(reference_labels, testdata['relation'], average = 'weighted')

testdata_df = testdata[['word','pos','def1','def2','relation']]

metrics_file = sys.argv[3]
with open(metrics_file, 'w+') as fd:
    fd.write('{:4f}\n'.format(f1))

testdata_df.to_csv('en_predicted.csv',sep='\t', index = False)
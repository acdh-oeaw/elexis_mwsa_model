import logging
import sys
import pickle
from pathlib import Path

from mwsa.service.data_loader import DataLoader


def load_data(file_path, file_name):
    data_loader = DataLoader()
    data = data_loader.load(file_path, file_name)
    return data_loader.split_feature_label(data)


logger = logging.getLogger('load_data')
logger.setLevel(logging.INFO)
logger.info("Loading data")

if len(sys.argv) != 3:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python data_loader.py file_path file_name')

file_path = sys.argv[1]
file_name = sys.argv[2]

features, labels = load_data(file_path, file_name)

output_dir = 'mwsa/data/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
feature_filename = output_dir + 'features_'+file_name+'.pkl'
with open(feature_filename, 'wb+') as file:
    pickle.dump(features, file)

labels_filename = output_dir + 'labels_'+file_name+'.pkl'
with open(labels_filename, 'wb+') as file:
    pickle.dump(labels, file)

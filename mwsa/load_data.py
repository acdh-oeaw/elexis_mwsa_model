import logging
import sys
import pickle

from mwsa.service.data_loader import DataLoader

logger = logging.getLogger('load_data')

if len(sys.argv) != 3:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python data_loader.py file_path file_name')

file_path = sys.argv[1]
file_name = sys.argv[2]

data_loader = DataLoader()
data = data_loader.load(file_path, file_name)
features, labels = data_loader.split_feature_label(data)

output_dir = 'mwsa/data/'
feature_filename = output_dir + 'features.pickle'
with open(feature_filename, 'wb+') as file:
    pickle.dump(features, file)

labels_filename = output_dir + 'labels.pickle'
with open(labels_filename, 'wb+') as file:
    pickle.dump(labels, file)
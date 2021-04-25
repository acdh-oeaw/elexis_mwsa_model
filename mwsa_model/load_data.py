import logging
import sys
import pickle
import warnings
from pathlib import Path

import yaml
from pandas.core.common import SettingWithCopyWarning
from mwsa_model.service.data_loader import DataLoader

warnings.filterwarnings(
    action='ignore',
    category=SettingWithCopyWarning,
    module=r'.*'
)


def load_data(file_path, file_name, version=None):
    data_loader = DataLoader()
    data = data_loader.load(file_path, file_name, version=version)
    return data_loader.split_feature_label(data)


logger = logging.getLogger('load_data')
logger.setLevel(logging.INFO)

logger.info("Loading data")

if len(sys.argv) != 3:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python data_loader.py file_path file_name')

lang = sys.argv[1]

config_file = lang + '_params.yaml'
with open(config_file, 'r') as fd:
    params = yaml.safe_load(fd)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    features, labels = load_data('mwsa/train', 'english_nuig.tsv', version=params['data']['version'])

output_dir = 'mwsa_model/data/'
Path(output_dir).mkdir(parents=True, exist_ok=True)

file_name = params['data']['dataset' \
                           '']
feature_filename = output_dir + 'features_' + file_name + '.pkl'

with open(feature_filename, 'wb+') as file:
    pickle.dump(features, file)

labels_filename = output_dir + 'labels_' + file_name + '.pkl'

with open(labels_filename, 'wb+') as file:
    pickle.dump(labels, file)

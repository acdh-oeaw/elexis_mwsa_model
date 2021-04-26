import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier

from mwsa_model.service.model_trainer import MwsaModelTrainer

dataset = sys.argv[1]
lang = sys.argv[2]
config_file = lang+'_params.yaml'

DATA_DIR = 'mwsa_model/data/'
OUTPUT_DIR = 'mwsa_model/output/'
MODEL_OUTPUT_DIR = OUTPUT_DIR + 'models/'
METRICS_OUTPUT_DIR = 'mwsa_model/output/metrics/'

FEATURE_FILE_NAME = 'preprocessed_'+dataset+'.pkl'
FEATURE_FILE = OUTPUT_DIR + FEATURE_FILE_NAME
LABELS_FILE_NAME = 'labels_'+dataset+'.pkl'
LABELS_FILE = DATA_DIR + LABELS_FILE_NAME
# TODO: move this to configfile

with open(config_file, 'r') as fd:
    params = yaml.safe_load(fd)

with open(FEATURE_FILE, 'rb') as pickle_file:
    preprocessed = pickle.load(pickle_file)

with open(LABELS_FILE, 'rb') as pickle_file:
    labels = pickle.load(pickle_file)

params_min = {
    'random_state': params['random_forest']['random_forest__random_state'],
    'bootstrap': params['random_forest']['random_forest__bootstrap'],
    'class_weight': params['random_forest']['random_forest__class_weight'],
    'max_depth': params['random_forest']['random_forest__max_depth'],
    'max_features': params['random_forest']['random_forest__max_features'],
    'min_samples_leaf': params['random_forest']['random_forest__min_samples_leaf'],
    'min_samples_split': params['random_forest']['random_forest__min_samples_split'],
    'n_estimators': params['random_forest']['random_forest__n_estimators'],
    'n_jobs': params['random_forest']['random_forest__n_jobs']
}

model_trainer = MwsaModelTrainer()

grid_search = model_trainer.configure_grid_search(RandomForestClassifier(), params_min)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = grid_search.fit(preprocessed, labels)


model_filename = MODEL_OUTPUT_DIR + lang + '.pkl'
path = Path(model_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(model_filename, 'wb+') as file:
    pickle.dump(model, file)

joblib_filename = MODEL_OUTPUT_DIR + lang + '.joblib'
path = Path(joblib_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(joblib_filename, 'wb+') as file:
    joblib.dump(model, file)

score_filename = METRICS_OUTPUT_DIR + lang + '_cv_score.txt'
path = Path(score_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(score_filename, 'w+') as fd:
    result = {
        'f1': model.best_score_
    }
    fd.write(json.dumps(result))
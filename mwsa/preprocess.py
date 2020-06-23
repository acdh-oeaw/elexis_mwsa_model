import logging
import pickle
import sys

from mwsa.service.model_trainer import MwsaModelTrainer
from mwsa.service.util import SupportedLanguages

logger = logging.getLogger('preprocess')

if len(sys.argv) != 3:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python preprcess.py features_file labels_file')

with open(sys.argv[1], 'rb') as pickle_file:
    features = pickle.load(pickle_file)

with open(sys.argv[2], 'rb') as pickle_file:
    labels = pickle.load(pickle_file)

model_trainer = MwsaModelTrainer()

pipeline = model_trainer.build_pipeline(SupportedLanguages.English)

params = {
    'preprocess__lang':[SupportedLanguages.English],
    'random_forest__bootstrap': [True],
    'random_forest__class_weight': ['balanced', 'balanced_subsample'],
    'random_forest__max_depth': [30],
    'random_forest__max_features': ['auto'],
    'random_forest__min_samples_leaf': [3, 5],
    'random_forest__min_samples_split': [2],
    'random_forest__n_estimators': [50,300],
    'random_forest__n_jobs': [5]
}

grid_search = model_trainer.configure_grid_serach(pipeline, params)

model = grid_search.fit(features, labels)

model_filename = 'mwsa/output/models/'+SupportedLanguages.English.value+'.pkl'
with open(model_filename, 'wb+') as file:
    pickle.dump(model, file)

score_filename = 'mwsa/output/metrics/'+SupportedLanguages.English.value+'_cv_score.txt'
with open(score_filename, 'w+') as fd:
    fd.write('{:4f}\n'.format(model.best_score_))
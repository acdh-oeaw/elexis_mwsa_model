import logging
import pickle
import sys

from pandas.core.common import SettingWithCopyWarning
from sklearn.externals import joblib

from mwsa.service.model_trainer import MwsaModelTrainer
from mwsa.service.util import SupportedLanguages
import warnings
from pathlib import Path

warnings.filterwarnings(
    action='ignore',
    category=SettingWithCopyWarning,
    module=r'.*'
)

logger = logging.getLogger('preprocess')
logger.setLevel(logging.INFO)

print(len(sys.argv))
for arg in sys.argv:
    print(arg)


if len(sys.argv) != 4:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python preprcess.py features_file labels_file language')

lang = sys.argv[3]

data_dir = 'data/'
with open(data_dir+sys.argv[1], 'rb') as pickle_file:
    features = pickle.load(pickle_file)

with open(data_dir+sys.argv[2], 'rb') as pickle_file:
    labels = pickle.load(pickle_file)

model_trainer = MwsaModelTrainer()

pipeline = model_trainer.build_pipeline(SupportedLanguages(lang))

params = {
    'preprocess__lang':[SupportedLanguages(lang)],
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

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = grid_search.fit(features, labels)


model_filename = 'mwsa/output/models/'+lang+'.pkl'
path = Path(model_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(model_filename, 'wb+') as file:
    pickle.dump(model, file)

joblib_filename = 'mwsa/output/models/'+lang+'.joblib'
path = Path(joblib_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(joblib_filename, 'wb+') as file:
    joblib.dump(model, file)

score_filename = 'mwsa/output/metrics/'+lang+'_cv_score.txt'
path = Path(score_filename)
path.parent.mkdir(parents=True, exist_ok=True)

with open(score_filename, 'w+') as fd:
    fd.write('{:4f}\n'.format(model.best_score_))
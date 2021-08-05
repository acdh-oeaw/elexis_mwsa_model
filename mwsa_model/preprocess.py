import logging
from pathlib import Path

import dill as pickle
import sys
import warnings

import stanza
from pandas.core.common import SettingWithCopyWarning
from sklearn.pipeline import Pipeline

from mwsa_model.service.model_trainer import MwsaModelTrainer
from mwsa_model.service.util import SupportedLanguages

OUTPUT_DIR = 'mwsa_model/output/'
PIPELINE_DIR = OUTPUT_DIR+'pipeline/'
DATA_DIR = 'mwsa_model/data/'

Path(PIPELINE_DIR).mkdir(parents=True, exist_ok=True)

warnings.filterwarnings(
    action='ignore',
    category=SettingWithCopyWarning,
    module=r'.*'
)

logger = logging.getLogger('preprocess')
logger.setLevel(logging.INFO)

for arg in sys.argv:
    logger.debug(arg)


if len(sys.argv) != 3:
    logger.error('Arguments error. Usage \n')
    logger.error('\t python preprcess.py features_file labels_file language')


logger.info("Changed")
dataset = sys.argv[1]
lang = sys.argv[2]

stanza.download(lang)

with open(DATA_DIR+'features_'+dataset+'.pkl', 'rb') as pickle_file:
    features = pickle.load(pickle_file)

model_trainer = MwsaModelTrainer()

pipeline: Pipeline = model_trainer.build_pipeline(SupportedLanguages(lang))
pipeline.set_params(preprocess__lang=SupportedLanguages(lang))
pipeline.fit(features)
preprocessed = pipeline.transform(features)

with open(PIPELINE_DIR+'pipeline_'+dataset+'.pkl', 'wb+') as file:
    pickle.dump(pipeline, file)
with open(OUTPUT_DIR+'preprocessed_'+dataset+'.pkl', 'wb+') as file:
    pickle.dump(preprocessed, file)




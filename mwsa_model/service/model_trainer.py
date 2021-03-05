from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mwsa_model import features
from mwsa_model.service.util import SupportedLanguages
from mwsa_model.transformers.lang_pipeline import english_pipeline, german_pipeline, russian_pipeline, italian_pipeline, \
    portuguese_pipeline, danish_pipeline, dutch_pipeline, serbian_pipeline, bulgarian_pipeline, slovene_pipeline, \
    estonian_pipeline, basque_pipeline, irish_pipeline, hungarian_pipeline


class MwsaModelTrainer(object):
    def __init__(self):
        self.pipelines = {
            SupportedLanguages.English: english_pipeline,
            SupportedLanguages.German: german_pipeline,
            SupportedLanguages.Russian: russian_pipeline,
            SupportedLanguages.Italian: italian_pipeline,
            SupportedLanguages.Portuguese: portuguese_pipeline,
            SupportedLanguages.Danish: danish_pipeline,
            SupportedLanguages.Dutch: dutch_pipeline,
            SupportedLanguages.Serbian: serbian_pipeline,
            SupportedLanguages.Bulgarian: bulgarian_pipeline,
            SupportedLanguages.Slovene: slovene_pipeline,
            SupportedLanguages.Estonian: estonian_pipeline,
            SupportedLanguages.Basque: basque_pipeline,
            SupportedLanguages.Irish: irish_pipeline,
            SupportedLanguages.Hungarian: hungarian_pipeline
        }

    def train(self, features, labels, grid_search):
        return grid_search.fit(features, labels)

    def build_pipeline(self, lang):
        return self.pipelines[lang]

    def configure_grid_search(self, pipeline, params, score='f1', cv=5, verbose=1):
        return GridSearchCV(pipeline, param_grid=params,
                            scoring='%s_weighted' % 'f1', cv=cv, verbose=verbose, n_jobs=-1)

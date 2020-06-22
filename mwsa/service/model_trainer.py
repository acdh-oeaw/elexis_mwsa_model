from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mwsa.service.util import SupportedLanguages
from mwsa.transformers.pipeline import SpacyProcessor, FirstWordSameProcessor, SimilarityProcessor, FeatureSelector, \
    DiffPosCountTransformer, OneHotPosTransformer


class MwsaModelTrainer(object):
    def __init__(self):
        english_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor()),
                                           ('diff_pos_count', DiffPosCountTransformer()),
                                           ('one_hot_pos', OneHotPosTransformer()),
                                           ('first_word_same', FirstWordSameProcessor()),
                                           ('similarity', SimilarityProcessor()),
                                           ('feature_selector', FeatureSelector()),
                                           ('random_forest', RandomForestClassifier())])
        self.pipelines = {
            SupportedLanguages.English: english_pipeline
        }

    def train(self, features, labels, grid_search):
        return grid_search.fit(features, labels)

    def build_pipeline(self, lang):
        return self.pipelines[lang]

    def configure_grid_serach(self, pipeline, params, score='f1', cv=5, n_jobs=-1, verbose=1):
        return GridSearchCV(pipeline, param_grid=params,
                            scoring='%s_weighted' % 'f1', cv=cv,
                            n_jobs=n_jobs, verbose=verbose)

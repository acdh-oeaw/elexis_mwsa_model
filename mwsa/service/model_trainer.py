from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mwsa import features
from mwsa.service.util import SupportedLanguages
from mwsa.transformers.pipeline import SpacyProcessor, FirstWordSameProcessor, SimilarityProcessor, FeatureSelector, \
    DiffPosCountTransformer, OneHotPosTransformer, MatchingLemmaTransformer, CountEachPosTransformer, \
    AvgSynsetCountTransformer, DifferenceInLengthTransformer, ToTargetSimilarityDiffTransformer, \
    MaxDependencyTreeDepthTransformer, TargetWordSynsetCountTransformer, SemicolonCountTransformer, TfidfTransformer


class MwsaModelTrainer(object):
    def __init__(self):
        english_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(with_wordnet=True)),
                                           ('diff_pos_count', DiffPosCountTransformer()),
                                           ('one_hot_pos', OneHotPosTransformer()),
                                           ('first_word_same', FirstWordSameProcessor()),
                                           ('similarity', SimilarityProcessor()),
                                           ('matching_lemma', MatchingLemmaTransformer()),
                                           #('pos_count', CountEachPosTransformer()),
                                           ('avg_synset_count', AvgSynsetCountTransformer()),
                                           ('diff_in_length', DifferenceInLengthTransformer()),
                                           ('max_depth_tree', MaxDependencyTreeDepthTransformer()),
                                           ('target_word_synset_count', TargetWordSynsetCountTransformer()),
                                           ('target_similarity_diff', ToTargetSimilarityDiffTransformer()),
                                           ('semicolon_diff', SemicolonCountTransformer()),
                                           ('feature_selector', FeatureSelector()),
                                           ('random_forest', RandomForestClassifier())])
        german_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor()),
                                          (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                          (features.SIMILARITY, SimilarityProcessor()),
                                          (features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                          (features.ONE_HOT_POS, OneHotPosTransformer()),
                                          (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                          (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                          (features.TFIDF_COS, TfidfTransformer()),
                                          ('feature_selector', FeatureSelector()),
                                          ('random_forest', RandomForestClassifier())])
        self.pipelines = {
            SupportedLanguages.English: english_pipeline,
            SupportedLanguages.German: german_pipeline
        }

    def train(self, features, labels, grid_search):
        return grid_search.fit(features, labels)

    def build_pipeline(self, lang):
        return self.pipelines[lang]

    def configure_grid_serach(self, pipeline, params, score='f1', cv=5, n_jobs=-1, verbose=1):
        return GridSearchCV(pipeline, param_grid=params,
                            scoring='%s_weighted' % 'f1', cv=cv,
                            n_jobs=n_jobs, verbose=verbose)

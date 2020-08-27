from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mwsa_model import features
from mwsa_model.service.util import SupportedLanguages
from mwsa_model.transformers.pipeline import SpacyProcessor, FirstWordSameProcessor, SimilarityProcessor, FeatureSelector, \
    DiffPosCountTransformer, OneHotPosTransformer, MatchingLemmaTransformer, CountEachPosTransformer, \
    AvgSynsetCountTransformer, DifferenceInLengthTransformer, ToTargetSimilarityDiffTransformer, \
    MaxDependencyTreeDepthTransformer, TargetWordSynsetCountTransformer, SemicolonCountTransformer, TfidfTransformer, \
    CosineTransformer, JaccardTransformer


class MwsaModelTrainer(object):
    def __init__(self):
        english_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(with_wordnet=True)),
                                           (features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                           #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                           (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                           (features.SIMILARITY, SimilarityProcessor()),
                                           (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                           #('pos_count', CountEachPosTransformer()),
                                           #(features.SYNSET_COUNT_DIFF, AvgSynsetCountTransformer()),
                                           (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                           (features.MAX_DEPTH_TREE_DIFF, MaxDependencyTreeDepthTransformer()),
                                           (features.TARGET_WORD_SYNSET_COUNT, TargetWordSynsetCountTransformer()),
                                           (features.SIMILARITY_DIFF_TO_TARGET, ToTargetSimilarityDiffTransformer()),
                                           #(features.SEMICOLON_DIFF, SemicolonCountTransformer()),
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
                                          #(features.JACCARD, JaccardTransformer()),
                                          #(features.COSINE, CosineTransformer()),
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

    def configure_grid_serach(self, pipeline, params, score='f1', cv=5, verbose=1):
        return GridSearchCV(pipeline, param_grid=params,
                            scoring='%s_weighted' % 'f1', cv=cv, verbose=verbose, n_jobs=-1)

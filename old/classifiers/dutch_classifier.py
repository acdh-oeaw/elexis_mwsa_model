import logging
import os

# warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from old.classifier_config import ClassifierConfig
from old.feature_extractor import FeatureExtractor
from old.model_trainer import ModelTrainer
from old.wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == '__main__':
    configure()

    german_config = ClassifierConfig('nl_core_news_sm', "dutch", 'data/train', balancing_strategy="none",testset_ratio=0.2, logger = 'dutch_all_features_nonebalance')

    feature_extractor = FeatureExtractor() \
        .first_word() \
        .similarity() \
        .diff_pos_count() \
        .tfidf() \
        .one_hot_pos() \
        .matching_lemma() \
        .count_each_pos() \
        .cosine() \
        .jaccard() \
        .difference_in_length()

    rf = {
        'estimator': RandomForestClassifier(),
        'parameters': {
            'bootstrap': [True],
            'class_weight': ['balanced', 'balanced_subsample','None'],
            'max_depth': [30],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [2, 5],
            'min_samples_split': [5, 8],
            'n_estimators': [500, 1000],
            'n_jobs':[8]
        }
    }
    # rf = {
    #     'estimator': RandomForestClassifier(),
    #     'parameters': {
    #         'bootstrap': [True],
    #         'max_depth': [30, 50],
    #         'max_features': [None],
    #         'min_samples_leaf': [3, 5],
    #         'min_samples_split': [2, 5, 8],
    #         'n_estimators': [500, 600]
    #     }
    # }
    dt = {'estimator': DecisionTreeClassifier(), 'parameters': {}}

    model_trainer = ModelTrainer(german_config.testset_ratio, german_config.logger)
    model_trainer.add_estimators([rf])
    german_classifier = WordSenseAlignmentClassifier(german_config, feature_extractor, model_trainer)
    german_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .select_features(['similarities', 'cos_tfidf',
                          'ADP', 'DET', 'pos_diff', 'len_diff',
                          'PRON', 'CONJ','X', 'PROPN', 'NOUN', 'cos', 'ADJ', 'VERB', 'jaccard', 'PUNCT', 'noun', 'ADV', 'adjective'])\
        .train(with_testset=True)

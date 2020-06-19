import logging
import os
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from classifier_config import ClassifierConfig
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == '__main__':
    configure()

    english_config = ClassifierConfig('en_core_web_lg', "english", 'data/train', balancing_strategy="none",
                                      testset_ratio=0.0, with_wordnet= True, dataset='english_nuig', logger = 'en_nuig_split_biggest')

    feature_extractor = FeatureExtractor() \
        .diff_pos_count() \
        .ont_hot_pos() \
        .matching_lemma() \
        .count_each_pos() \
        .avg_count_synsets() \
        .difference_in_length()\
        .similarity_diff_to_target()\
        .max_dependency_tree_depth() \
        .target_word_synset_count()\
        .token_count_norm_diff()\
        .semicol_count()\
        .elmo_similarity()

    rf = {
        'estimator': RandomForestClassifier(),
        'parameters': {
            'bootstrap': [True],
            'class_weight': ['balanced', 'balanced_subsample','None'],
            'max_depth': [5, 10, 30, 50, 80],
            'max_features': [2, 10, 15, 'auto', 'sqrt', 'log2'],
            'min_samples_leaf': [2, 5, 10],
            'min_samples_split': [2, 5, 10, 20],
            'n_estimators': [500, 800, 1000, 1500],
            'n_jobs':[8]
        }
    }

    model_trainer = ModelTrainer(english_config.testset_ratio, english_config.logger)
    model_trainer.add_estimators([rf])
    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor, model_trainer)
    english_classifier.load_data() \
        .extract_features(['len_diff', 'pos_diff']) \
        .train()

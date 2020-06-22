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
        .one_hot_pos() \
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
            'class_weight': ['balanced', 'balanced_subsample', ],
            'max_depth': [10, 20],
            'max_features': ['auto', 'sqrt', 'log2', None],
            'min_samples_leaf': [2],
            'min_samples_split': [5, 10],
            'n_estimators': [300, 800],
            'n_jobs':[8]
        }
    }

    model_trainer = ModelTrainer(english_config.testset_ratio, english_config.logger)
    model_trainer.add_estimators([rf])
    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor, model_trainer)
    english_classifier.load_data() \
        .extract_features(['len_diff', 'pos_diff']) \
        .select_features(['target_word_synset_count',
                          'elmo_sim',
                          'simdiff_to_target',
                          'synsets_count_diff',
                          'lemma_match_normalized',
                          'token_count_norm_diff',
                          'len_diff',
                          'NOUN',
                          'VERB',
                          'PUNCT',
                          'pos_diff',
                          'CCONJ',
                          'semicol_count2_norm',
                          'ADP',
                          'ADJ',
                          'semicol_diff',
                          'max_depth_deptree_2',
                          'max_depth_deptree_1'])\
        .train()

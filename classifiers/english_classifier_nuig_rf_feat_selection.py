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
                                      testset_ratio=0.0, with_wordnet= True, dataset='english_nuig', logger = 'en_nuig_feat_selection')

    feature_extractor = FeatureExtractor() \
        .first_word() \
        .similarity() \
        .tfidf() \
        .ont_hot_pos() \
        .matching_lemma() \
        .count_each_pos() \
        .jaccard() \
        .avg_count_synsets() \
        .difference_in_length()\
        .max_dependency_tree_depth() \
        .target_word_synset_count()\

    rf = {
        'estimator': RandomForestClassifier(),
        'parameters': {
            'class_weight': ['balanced_subsample', 'balanced'],
            'max_depth': [5,10, 15],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [2],
            'min_samples_split': [5, 10],
            'n_estimators': [300, 1000],
            'n_jobs':[8]
        }
    }

    model_trainer = ModelTrainer(english_config.testset_ratio, english_config.logger)
    model_trainer.add_estimators([rf])
    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor, model_trainer)
    english_classifier.load_data() \
        .extract_features(['similarities', 'len_diff']) \
        .select_features(['cos_tfidf','jaccard','similarities','first_word_same','PART','noun','adjective','verb','target_word_synset_count','adverb','len_diff'])\
        .train()

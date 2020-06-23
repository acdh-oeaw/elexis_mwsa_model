import logging
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from old.classifier_config import ClassifierConfig
from old.feature_extractor import FeatureExtractor
from old.model_trainer import ModelTrainer
from old.wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == '__main__':
    configure()

    english_config = ClassifierConfig('en_core_web_lg', "english", 'data/train', balancing_strategy="none",
                                      testset_ratio=0.0, with_wordnet= True, dataset='english_nuig', logger = 'en_nuig_split_biggest')

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
        .avg_count_synsets() \
        .difference_in_length()\
        .similarity_diff_to_target()\
        .max_dependency_tree_depth() \
        .target_word_synset_count()\
        .token_count_norm_diff()\
        .semicol_count()

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
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train()

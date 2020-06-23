import logging
import os

import pandas as pd
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

    english_config = ClassifierConfig('en_core_web_lg', "english", 'data/train', balancing_strategy="none",
                                      testset_ratio=0.2, with_wordnet= True, dataset='english_nuig', logger = 'en_nuig')

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
        .difference_in_length()

    dt = {'estimator': DecisionTreeClassifier(), 'parameters': {}}

    model_trainer = ModelTrainer(english_config.testset_ratio, english_config.logger)
    model_trainer.add_estimators([dt])
    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor, model_trainer)
    english_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

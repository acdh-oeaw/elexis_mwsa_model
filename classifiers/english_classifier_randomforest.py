# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination
import logging
import os

# warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifier_config import ClassifierConfig
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == '__main__':
    configure()

    english_config = ClassifierConfig('en_core_web_lg', "english", '../data/train', balancing_strategy="none",
                                      testset_ratio=0.2, with_wordnet=True)

    feature_extractor = FeatureExtractor() \
        .first_word() \
        .similarity() \
        .diff_pos_count() \
        .tfidf() \
        .ont_hot_pos() \
        .matching_lemma() \
        .count_each_pos() \
        .cosine() \
        .jaccard() \
        .avg_count_synsets() \
        .difference_in_length()

    rf = {
        'estimator': RandomForestClassifier(),
        'parameters': {
            'bootstrap': [True],
            'class_weight': ['balanced', 'balanced_subsample', 'None'],
            'max_depth': [30, 50, 80],
            'max_features': [2, 10, 15, 'auto', 'sqrt', 'log2', None],
            'min_samples_leaf': [3, 5],
            'min_samples_split': [2, 5, 8],
            'n_estimators': [500, 800],
            'n_jobs': [-1]
        }
    }

    model_trainer = ModelTrainer(english_config.testset_ratio, english_config.logger)
    model_trainer.add_estimators([rf])
    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor, model_trainer)
    english_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

import logging
import os

# warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from classifier_config import ClassifierConfig
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer, BaseClassifier
from wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


if __name__ == '__main__':
    configure()

    german_config = ClassifierConfig('de_core_news_md', "german", 'data/train', balancing_strategy="none",testset_ratio=0.2, logger = 'de_all_features_nonebalance')

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
        .difference_in_length()

    model_trainer = ModelTrainer(german_config.testset_ratio, german_config.logger)
    model_trainer.add_estimators([BaseClassifier()])
    german_classifier = WordSenseAlignmentClassifier(german_config, feature_extractor, model_trainer)
    german_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

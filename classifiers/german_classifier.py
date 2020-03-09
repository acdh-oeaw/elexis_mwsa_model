# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination
import logging
import os
import warnings

# warnings.filterwarnings('ignore')
import pandas as pd
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

    german_config = ClassifierConfig('de_core_news_md', "german", 'data/train', balancing_strategy="none",testset_ratio=0.2)

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
    german_classifier = WordSenseAlignmentClassifier(german_config, feature_extractor, model_trainer)
    german_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

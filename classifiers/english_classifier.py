# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination


import warnings

# warnings.filterwarnings('ignore')
import pandas as pd
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from classifier_config import ClassifierConfig
from feature_extractor import FeatureExtractor
from wsa_classifier import WordSenseAlignmentClassifier


def configure():
    pd.set_option('display.max_colwidth', -1)


if __name__ == '__main__':
    configure()

    english_config = ClassifierConfig('en_core_web_lg', "english", '../data/train', balancing_strategy="none", testset_ratio=0.2)

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
        .avg_count_synsets()\
        .difference_in_length()

    english_classifier = WordSenseAlignmentClassifier(english_config, feature_extractor)
    english_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

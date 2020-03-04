# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination


import warnings

# warnings.filterwarnings('ignore')
import pandas as pd
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from classifier_config import ClassifierConfig
from feature_extractor import FeatureExtractor
from wsa_classifier import DataLoader


def convert_to_text(token_array):
    seperator = ' '
    return seperator.join(token_array)


def get_baseline_df(y_test):
    tp = 0
    for index in y_test.index:
        if y_test[index] == 'none':
            tp += 1

    return float(tp / len(y_test))


def is_not_none(df):
    return df['relation'] != 'none'


def configure():
    pd.set_option('display.max_colwidth', -1)


# def count_relation_and_sort():
#    return str(balanced_en_data.groupby('relation').count().word.sort_values(ascending=False)) + "\n"


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

    german_classifier = DataLoader(german_config, feature_extractor)
    german_classifier.load_data() \
        .extract_features(['similarities', 'len_diff', 'pos_diff']) \
        .train(with_testset=True)

# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination


import warnings
from copy import deepcopy

# warnings.filterwarnings('ignore')
import os
import spacy
import pandas as pd
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer

folder = 'data/train'


def add_column_names(df):
    column_names = ['word', 'pos', 'def1', 'def2', 'relation']
    df.columns = column_names


def load_data(file_path):
    loaded_data = pd.read_csv(file_path, sep='\t', header=None)
    add_column_names(loaded_data)

    return loaded_data


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


def has_label(df, label):
    return df['relation'] == label


def configure():
    pd.set_option('display.max_colwidth', -1)


def load_training_data():
    combined_set = {}

    for filename in os.listdir(folder):
        if filename.endswith(".tsv"):
            combined_set[filename.split('.')[0]] = load_data(folder + '/' + filename)

    return combined_set


def undersample_dataset(imbalanced_set):
    none = imbalanced_set[has_label(imbalanced_set, 'none') == True]
    second_biggest = imbalanced_set.groupby('relation').count().word.sort_values(ascending=False)[1]
    result = imbalanced_set.drop(none.index[second_biggest:])

    return result.sample(frac=1, random_state=7)


def lemmatizer(doc, spacy_model):
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    return spacy_model.make_doc(u' '.join(doc))


def remove_stopwords(doc, output='text'):
    # TODO: ADD 'etc' to stopwords list
    if output == 'token':
        return [token for token in doc if token.is_stop != True and token.is_punct != True]

    return [token.text for token in doc if token.is_stop != True and token.is_punct != True]


def filter_small_length(dataset, threshold):
    return len(dataset.index) > threshold


def sort(lst):
    return sorted(lst, key=len)


def upsample_from_bigger_set(smallest_by_label, bigger_by_label):
    biggest_label, biggest_label_size = find_biggest_label_and_size(smallest_by_label)

    return upsample_by_diff(bigger_by_label, biggest_label, biggest_label_size, smallest_by_label)


def upsample_by_diff(bigger_by_label, biggest_label, biggest_label_size, smallest_by_label):
    for key in smallest_by_label:
        if key != biggest_label:
            diff = biggest_label_size - len(smallest_by_label[key].index)
            if diff > 0:
                new_data = bigger_by_label[key].sample(n=diff, random_state=7, replace=True)
                smallest_by_label[key] = smallest_by_label[key].append(new_data)

    return smallest_by_label


def find_biggest_label_and_size(dataset_by_label):
    biggest_label = None
    biggest_label_size = 0

    for key in dataset_by_label:
        if len(dataset_by_label[key].index) > biggest_label_size:
            biggest_label_size = len(dataset_by_label[key].index)
            biggest_label = key

    return biggest_label, biggest_label_size


def combine_labels(dict_by_label):
    df = pd.DataFrame()

    for key in dict_by_label:
        df = df.append(dict_by_label[key])

    return df.sample(frac=1, random_state=7)


def sort_dataset(all_data, dataset_lang):
    lang_data = []
    for key in all_data.keys():
        if dataset_lang in key:
            lang_data.append(all_data[key])
    sorted_sets = list(filter(lambda elem: filter_small_length(elem, 100), sort(lang_data)))
    return sorted_sets


def categorize_by_label(df):
    relation_labels = df['relation'].unique()
    smallest_by_label = {}
    for relation in relation_labels:
        smallest_by_label[relation] = df[has_label(df, relation)]

    return smallest_by_label


def switch_broader_and_narrower(dataset_by_label):
    biggest_label, biggest_label_size = find_biggest_label_and_size(dataset_by_label)
    original_df = deepcopy(dataset_by_label)

    for key in ['narrower', 'broader', 'related', 'exact']:
        if biggest_label_size - len(original_df[key].index) > 0:
            opposite_data = swap_columns(key, original_df)
            dataset_by_label[key] = dataset_by_label[key].append(opposite_data)

    return dataset_by_label


def swap_columns(key, original_df):
    opposite_relation = {'narrower': 'broader', 'broader': 'narrower', 'related': 'related', 'exact': 'exact'}
    opposite_data = original_df[opposite_relation[key]].copy(deep=True)

    temp = opposite_data['def1'].copy(deep=True)
    opposite_data['def1'] = opposite_data['def2']
    opposite_data['def2'] = temp
    opposite_data['relation'] = key

    return opposite_data


def balance_dataset(sorted_sets, balancing):
    if balancing == 'undersampling':
        result = undersample_dataset(sorted_sets[0])

    elif balancing == 'oversampling':
        smallest = sorted_sets[0]
        smallest_by_label = categorize_by_label(smallest)
        smallest_by_label = switch_broader_and_narrower(smallest_by_label)
        if len(sorted_sets) > 1:
            bigger = sorted_sets[1]
        else:
            bigger = deepcopy(sorted_sets[0])

        bigger_by_label = categorize_by_label(bigger)
        result = combine_labels(upsample_from_bigger_set(smallest_by_label, bigger_by_label))

    else:
        return sorted_sets[0]

    return result.reset_index()


def count_relation_and_sort():
    return str(balanced_en_data.groupby('relation').count().word.sort_values(ascending=False)) + "\n"


class DataLoader:
    def __init__(self, spacy_pipeline, language, balancing='oversampling'):
        self._language = language
        self._nlp = spacy_pipeline
        self._balancing = balancing

    def __load_and_balance(self):
        all_data = load_training_data()
        sorted_sets = sort_dataset(all_data, self._language)

        data = balance_dataset(sorted_sets, self._balancing)

        self.__preprocess(data)

        return data

    def __preprocess(self, data):
        data['processed_1'] = data['def1'].map(self._nlp)
        data['processed_2'] = data['def2'].map(self._nlp)
        data['lemmatized_1'] = data['processed_1'].map(lambda doc: lemmatizer(doc, self._nlp))
        data['stopwords_removed_1'] = data['lemmatized_1'].map(remove_stopwords)
        print(data['lemmatized_1'])
        print(data['stopwords_removed_1'])
        data['lemmatized_2'] = data['processed_2'].map(lambda doc: lemmatizer(doc, self._nlp))
        print(data['lemmatized_2'])
        data['stopwords_removed_2'] = data['lemmatized_2'].map(remove_stopwords)


    def load_data(self):
        data = self.__load_and_balance()
        self.__preprocess(data)
        return data


if __name__ == '__main__':
    configure()
    nlp = spacy.load('de_core_news_md')
    ##nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

    #balanced_en_data = loa\d_preprocess('german', nlp)

    classifier = DataLoader(nlp, "german", "none")
    balanced_en_data = classifier.load_data()
    # report_file.write(count_relation_and_sort())

#['similarities', 'len_diff', 'pos_diff','synset_count_1','synset_count_2']
#        .avg_count_synsets()\
    #['similarities', 'len_diff', 'pos_diff']
    features = FeatureExtractor() \
        .first_word()\
        .similarity()\
        .diff_pos_count()\
        .tfidf()\
        .ont_hot_pos()\
        .matching_lemma()\
        .count_each_pos()\
        .cosine()\
        .jaccard()\
        .difference_in_length()\
        .extract(balanced_en_data, ['similarities', 'len_diff', 'pos_diff'])
    # .avg_count_synsets()\

    models = ModelTrainer(features, balanced_en_data['relation'], 0.2)\
        .train(with_testset=True)

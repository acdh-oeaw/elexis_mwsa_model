# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination
# TODO: Ask papers/results
# TODO: Ask Tanja

import warnings
from copy import deepcopy

# warnings.filterwarnings('ignore')
import os
import spacy
import pandas as pd

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


def remove_stopwords(doc):
    # TODO: ADD 'etc' to stopwords list
    doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
    return doc


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


def load_and_preprocess(dataset_lang, spacy_model, balancing='oversampling'):
    all_data = load_training_data()

    sorted_sets = sort_dataset(all_data, dataset_lang)

    balanced = balance_dataset(sorted_sets, balancing)

    balanced['processed_1'] = balanced['def1'].map(spacy_model)
    balanced['processed_2'] = balanced['def2'].map(spacy_model)

    balanced['lemmatized_1'] = balanced['processed_1'].map(lambda doc: lemmatizer(doc, spacy_model))
    balanced['stopwords_removed_1'] = balanced['lemmatized_1'].map(remove_stopwords)
    print(balanced['lemmatized_1'])
    print(balanced['stopwords_removed_1'])

    balanced['lemmatized_2'] = balanced['processed_2'].map(lambda doc: lemmatizer(doc, spacy_model))
    print(balanced['lemmatized_2'])
    balanced['stopwords_removed_2'] = balanced['lemmatized_2'].map(remove_stopwords)

    return balanced


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

    else:
        smallest = sorted_sets[0]
        bigger = sorted_sets[1]

        smallest_by_label = categorize_by_label(smallest)
        bigger_by_label = categorize_by_label(bigger)
        smallest_by_label = switch_broader_and_narrower(smallest_by_label)
        result = combine_labels(upsample_from_bigger_set(smallest_by_label, bigger_by_label))

    return result.reset_index()


def count_relation_and_sort():
    return str(balanced_en_data.groupby('relation').count().word.sort_values(ascending=False)) + "\n"



if __name__ == '__main__':
    configure()
    nlp = spacy.load('en_core_web_lg')


    balanced_en_data = load_and_preprocess('english', nlp)

    #report_file.write(count_relation_and_sort())

    features = FeatureExtractor(feats_to_scale=['similarities', 'len_diff', 'pos_diff']).extract_features(
        balanced_en_data)

    model_trainer = ModelTrainer(features, balanced_en_data['relation'])
    models = model_trainer.train()
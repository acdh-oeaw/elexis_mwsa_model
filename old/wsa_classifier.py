import logging
import os
from copy import deepcopy
from datetime import datetime
import numpy as np
import pandas as pd
import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from old.classifier_config import ClassifierConfig
from old.feature_extractor import FeatureExtractor
from old.model_trainer import ModelTrainer


def has_label(df, label):
    return df['relation'] == label


def lemmatizer(doc, spacy_model):
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    return spacy_model.make_doc(u' '.join(doc))


def remove_stopwords(doc, output='text'):
    # TODO: ADD 'etc' to stopwords list
    if output == 'token':
        return [token for token in doc if token.is_stop is not True and token.is_punct is not True]

    return [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]


class WordSenseAlignmentClassifier:
    def __init__(self, config, feature_extractor, model_trainer):
        assert isinstance(feature_extractor, FeatureExtractor)
        assert isinstance(model_trainer, ModelTrainer)
        assert isinstance(config, ClassifierConfig)

        self._language = config.language
        self._dataset_name = config.dataset
        self._balancing = config.balancing_strategy
        self._nlp = spacy.load(config.language_model)
        self._folder = config.folder
        self.__configure_logger(config)
        if config.with_wordnet is True:
            self._nlp.add_pipe(WordnetAnnotator(self._nlp.lang), after='tagger')
        self._model_trainer = model_trainer
        # ModelTrainer(config.testset_ratio, self._logger.name)
        self._feature_extractor = feature_extractor
        self._data = None
        self._is_testdata = config.is_testdata

    def __configure_logger(self, config):
        self._LOG_FILENAME = 'reports/' + '_'.join(
            [config.language, config.balancing_strategy, str(config.with_testset),
             datetime.now().strftime("%Y%m%d-%H%M%S"), '.log'])
        self._logger = logging.getLogger(config.logger)
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(self._LOG_FILENAME)
        self._logger.addHandler(handler)

    def _add_column_names(self, df):
        column_names = ['word', 'pos', 'def1', 'def2', 'relation']
        if self._is_testdata is True:
            column_names.remove('relation')

        df.columns = column_names

    def __load_data(self, file_path):
        loaded_data = pd.read_csv(file_path, sep='\t', header=None)
        self._add_column_names(loaded_data)

        return loaded_data

    def __filter_small_length(self, dataset, threshold):
        return len(dataset.index) > threshold

    def __sort(self, lst):
        return sorted(lst, key=len)

    def __sort_dataset(self, all_data, dataset_lang):
        lang_data = []
        for key in all_data.keys():
            if dataset_lang in key:
                lang_data.append(all_data[key])
        sorted_sets = list(filter(lambda elem: self.__filter_small_length(elem, 100), self.__sort(lang_data)))
        return sorted_sets

    def __load_training_data(self):
        combined_set = {}

        for filename in os.listdir(self._folder):
            if filename.endswith(".tsv"):
                combined_set[filename.split('.')[0]] = self.__load_data(self._folder + '/' + filename)

        return combined_set

    def __undersample_dataset(self, imbalanced_set):
        none, second_biggest = self.__extract_two_top_groups(imbalanced_set)
        result = imbalanced_set.drop(none.index[second_biggest:])

        return result.sample(frac=1, random_state=7)

    def __extract_two_top_groups(self, imbalanced_set):
        none = imbalanced_set[has_label(imbalanced_set, 'none') == True]
        second_biggest = imbalanced_set.groupby('relation').count().word.sort_values(ascending=False)[1]
        return none, second_biggest

    def __categorize_by_label(self, df):
        relation_labels = df['relation'].unique()
        smallest_by_label = {}
        for relation in relation_labels:
            smallest_by_label[relation] = df[has_label(df, relation)]

        return smallest_by_label

    def __find_biggest_label_and_size(self, dataset_by_label):
        biggest_label = None
        biggest_label_size = 0

        for key in dataset_by_label:
            if len(dataset_by_label[key].index) > biggest_label_size:
                biggest_label_size = len(dataset_by_label[key].index)
                biggest_label = key

        return biggest_label, biggest_label_size

    def __swap_columns(self, key, original_df):
        opposite_relation = {'narrower': 'broader', 'broader': 'narrower', 'related': 'related', 'exact': 'exact'}
        opposite_data = original_df[opposite_relation[key]].copy(deep=True)

        temp = opposite_data['def1'].copy(deep=True)
        opposite_data['def1'] = opposite_data['def2']
        opposite_data['def2'] = temp
        opposite_data['relation'] = key

        return opposite_data

    def __switch_broader_and_narrower(self, dataset_by_label):
        biggest_label, biggest_label_size = self.__find_biggest_label_and_size(dataset_by_label)
        original_df = deepcopy(dataset_by_label)

        for key in ['narrower', 'broader', 'related', 'exact']:
            if biggest_label_size - len(original_df[key].index) > 0:
                opposite_data = self.__swap_columns(key, original_df)
                dataset_by_label[key] = dataset_by_label[key].append(opposite_data)

        return dataset_by_label

    def __combine_labels(self, dict_by_label):
        df = pd.DataFrame()

        for key in dict_by_label:
            df = df.append(dict_by_label[key])

        return df.sample(frac=1, random_state=7)

    def __upsample_from_bigger_set(self, smallest_by_label, bigger_by_label):
        biggest_label, biggest_label_size = self.__find_biggest_label_and_size(smallest_by_label)

        return self.__upsample_by_diff(bigger_by_label, biggest_label, biggest_label_size, smallest_by_label)

    def __upsample_by_diff(self, bigger_by_label, biggest_label, biggest_label_size, smallest_by_label):
        for key in smallest_by_label:
            if key != biggest_label:
                diff = biggest_label_size - len(smallest_by_label[key].index)
                if diff > 0:
                    new_data = bigger_by_label[key].sample(n=diff, random_state=7, replace=True)
                    smallest_by_label[key] = smallest_by_label[key].append(new_data)

        return smallest_by_label

    def __balance_dataset(self, sorted_sets, balancing):
        if balancing == 'undersampling':
            result = self.__undersample_dataset(sorted_sets[0])
        elif balancing == 'oversampling' or balancing == 'swap':
            smallest = sorted_sets[0]
            smallest_by_label = self.__categorize_by_label(smallest)
            smallest_by_label = self.__switch_broader_and_narrower(smallest_by_label)
            if len(sorted_sets) > 1:
                bigger = sorted_sets[1]
            else:
                bigger = deepcopy(sorted_sets[0])

            if balancing == 'oversampling':
                bigger_by_label = self.__categorize_by_label(bigger)
                result = self.__combine_labels(self.__upsample_from_bigger_set(smallest_by_label, bigger_by_label))
            else:
                result = self.__combine_labels(smallest_by_label)

        elif balancing == 'split_biggest' and not self._is_testdata:
            by_label = self.__categorize_by_label(sorted_sets[0])
            none, second_biggest = self.__extract_two_top_groups(sorted_sets[0])
            splitted = np.array_split(none, int(len(none) / second_biggest))

            self.__replace_none_by_splitted(by_label, splitted)
            result = self.__combine_labels(by_label)

        else:
            return sorted_sets[0]

        return result.reset_index()

    def __replace_none_by_splitted(self, df, splitted):
        df.pop('none', None)
        for splitted_df in splitted:
            df[splitted_df.iloc[0]['relation']] = splitted_df

    def __load_and_balance(self):
        all_data = self.__load_training_data()
        if self._dataset_name is not None:
            all_data = {self._dataset_name: all_data[self._dataset_name]}

        sorted_sets = self.__sort_dataset(all_data, self._language)
        data = self.__balance_dataset(sorted_sets, self._balancing)
        self.__preprocess(data)

        return data

    def __preprocess(self, data):
        data['processed_1'] = data['def1'].map(self._nlp)
        data['processed_2'] = data['def2'].map(self._nlp)
        data['word_processed'] = data['word'].map(self._nlp)
        data['lemmatized_1'] = data['processed_1'].map(lambda doc: lemmatizer(doc, self._nlp))
        data['stopwords_removed_1'] = data['lemmatized_1'].map(remove_stopwords)
        data['lemmatized_2'] = data['processed_2'].map(lambda doc: lemmatizer(doc, self._nlp))
        data['stopwords_removed_2'] = data['lemmatized_2'].map(remove_stopwords)
        self._data = data

    def load_data(self):
        data = self.__load_and_balance()
        self.__preprocess(data)
        return self

    def extract_features(self, feats_to_scale):
        self._feature_extractor.extract(self._data, feats_to_scale)
        return self

    def select_features(self, feats_to_include):
        self._feature_extractor.keep_feats(feats_to_include)
        return self

    def train(self, with_testset=False):
        models = self._model_trainer.train(self._feature_extractor.feats, self._data['relation'], with_testset)

    def get_preprocessed_data(self):
        return self._data

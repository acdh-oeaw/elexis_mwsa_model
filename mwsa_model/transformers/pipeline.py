import pandas as pd

pd.options.mode.chained_assignment = None
import spacy
import time
import logging

from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from spacy_wordnet.wordnet_annotator import WordnetAnnotator

from mwsa_model.service.util import SupportedLanguages
from mwsa_model import features
from nltk.corpus import wordnet as wn
from spacy_stanza import StanzaLanguage
import stanza
import warnings
from collections import Counter

import gensim
import gensim.downloader as api

from scipy.spatial import distance

import numpy as np
from tqdm import tqdm
from ufal.udpipe import Model, Pipeline

import fasttext
import fasttext.util

from navec import Navec

warnings.filterwarnings("ignore", category=UserWarning)

from spacy.vocab import Vocab

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

spacy_models = {
    SupportedLanguages.English: 'en_core_web_md',
    SupportedLanguages.German: 'de_core_news_md',
    SupportedLanguages.Danish: 'da_core_news_md',
    SupportedLanguages.Dutch: 'nl_core_news_md',
    SupportedLanguages.Italian: 'it_core_news_md',
    SupportedLanguages.Portuguese: 'pt_core_news_md'

}

models = {SupportedLanguages.Russian: StanzaLanguage(stanza.Pipeline(lang="ru")),
          SupportedLanguages.Slovene: StanzaLanguage(stanza.Pipeline(lang="sl")),
          SupportedLanguages.Serbian: StanzaLanguage(stanza.Pipeline(lang="sr")),
          SupportedLanguages.Bulgarian: StanzaLanguage(stanza.Pipeline(lang="bg")),
          SupportedLanguages.Hungarian: StanzaLanguage(stanza.Pipeline(lang="hu")),
          SupportedLanguages.Basque: StanzaLanguage(stanza.Pipeline(lang = "eu")),
          SupportedLanguages.Irish: StanzaLanguage(stanza.Pipeline(lang="ga")),
          SupportedLanguages.Estonian: StanzaLanguage(stanza.Pipeline(lang="et")),
          SupportedLanguages.Italian: spacy.load(spacy_models[SupportedLanguages.Italian]),
          SupportedLanguages.English: spacy.load(spacy_models[SupportedLanguages.English]),
          SupportedLanguages.German: spacy.load(spacy_models[SupportedLanguages.German]),
          SupportedLanguages.Danish: spacy.load(spacy_models[SupportedLanguages.Danish]),
          SupportedLanguages.Dutch: spacy.load(spacy_models[SupportedLanguages.Dutch]),
          SupportedLanguages.Portuguese: spacy.load(spacy_models[SupportedLanguages.Portuguese])
          }


def lemmatizer(doc, spacy_model):
    if type(doc) == float:
        return spacy_model.make_doc("")

    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    return spacy_model.make_doc(u' '.join(doc))


def remove_stopwords(doc, output='text'):
    # TODO: ADD 'etc' to stopwords list
    if output == 'token':
        return [token for token in doc if token.is_stop is not True and token.is_punct is not True]

    return [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]

def remove_special_characters(input_string):
    output_string = input_string.strip("0123456789|")
    return output_string


class Error(Exception):
    pass


class UnsupportedSpacyModelError(Error):

    def __init__(self, message):
        self.message = message


class SpacyProcessor(BaseEstimator, TransformerMixin):

    def __init__(self, lang=None, with_wordnet=False):
        self.logger = logging.getLogger(__name__)
        self.lang = lang
        self.with_wordnet = with_wordnet

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        nlp = models[self.lang]
        if self.lang == SupportedLanguages.English and self.with_wordnet:
            nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

        logger.debug('transforming with spacy...')

        logger.debug(len(X['def1']), len(X['def2']))

        X.loc[:, 'def1'] = X['def1'].map(lambda features: remove_special_characters(features))
        logger.debug('-------------special characters removed 1  ------------')

        X.loc[:, 'def2'] = X['def2'].map(lambda features: remove_special_characters(features))
        logger.debug('-------------special characters removed 2  ------------')

        X.loc[:, 'processed_1'] = pd.Series(list(nlp.pipe(iter(X['def1']), batch_size=1000)))
        logger.debug('----------processed 1 ------------')

        X.loc[:, 'processed_2'] = pd.Series(list(nlp.pipe(iter(X['def2']), batch_size=1000)))
        logger.debug('----------processed 2 ------------')

        X.loc[:, 'word_processed'] = pd.Series(list(nlp.pipe(iter(X['word']), batch_size=1000)))
        logger.debug('------------word processed ------------')

        X.loc[:, 'lemmatized_1'] = X['processed_1'].map(lambda doc: lemmatizer(doc, nlp))
        X.loc[:, 'stopwords_removed_1'] = X['lemmatized_1'].map(remove_stopwords)
        logger.debug('-------------lemma and sw removed 1  ------------')

        X.loc[:, 'lemmatized_2'] = X['processed_2'].map(lambda doc: lemmatizer(doc, nlp))
        X.loc[:, 'stopwords_removed_2'] = X['lemmatized_2'].map(remove_stopwords)
        logger.debug('-------------lemma and sw removed 2  ------------')

        logger.debug('SpacyProcessor.transform() took %.3f seconds' % (time.time() - t0))

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        #X.to_pickle("/home/varya/data/dataframe_with_features/data.pkl")

        for col in X.columns:
            if col in ['word', 'pos', 'def1', 'def2',
                       'processed_1', 'processed_2', 'word_processed',
                       'lemmatized_1', 'stopwords_removed_1', 'lemmatized_2',
                       'stopwords_removed_2', 'relation', 'filtered_def1', 'filtered_def2']: 
                X = X.drop(col, axis=1)

        logger.debug('FeatureSelector.transform() took %.3f seconds' % (time.time() - t0))

        return X


class FirstWordSameProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, features.FIRST_WORD_SAME] = X.apply(
            lambda row: self.__first_word_same(row['def1'], row['def2']), axis=1)
        logger.debug('FirstWordSameProcessor.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __first_word_same(col1, col2):
        return col1.split(' ')[0].lower() == col2.split(' ')[0].lower()




class SimilarityProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, features.SIMILARITY] = X.apply(
            lambda row: row['processed_1'].similarity(row['processed_2'])
            if type(row['processed_1']) != float else 0, axis=1)  # how often is type(row['processed_1'])==float ?

        logger.debug('SimilarityProcessor.transform() took %.3f seconds' % (time.time() - t0))

        return X



class OneHotPosTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.encoder = None

    def fit(self, X, y=None):
        pos_numpy = X['pos'].to_numpy().reshape(-1, 1)
        self.encoder = OneHotEncoder(handle_unknown='ignore') \
            .fit(pos_numpy)

        return self

    def transform(self, X, y=None):
        t0 = time.time()
        pos_numpy = X['pos'].to_numpy().reshape(-1, 1)
        encoded_array = self.encoder.transform(pos_numpy).toarray()

        encoded_dataframe = pd.DataFrame(data=encoded_array[0:, 0:],
                                         index=X.index,
                                         columns=self.encoder.categories_[0])
        logger.debug('OneHotPosTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return pd.concat([X, encoded_dataframe], axis=1)


class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf_vectorizer = None

    def fit(self, X, y=None):
        col1 = X['stopwords_removed_1']
        col2 = X['stopwords_removed_2']

        values = self.__join_definitions(col1, col2)
        self.tfidf_vectorizer = TfidfVectorizer().fit(values)

        return self

    def transform(self, X, y=None):
        t0 = time.time()
        col1 = X['stopwords_removed_1']
        col2 = X['stopwords_removed_2']

        tfidf_holder = pd.DataFrame()
        tfidf_holder['col1'] = col1
        tfidf_holder['col2'] = col2

        values = self.__join_definitions(col1, col2)

        tfidf_matrix = self.tfidf_vectorizer.transform(values)
        split_index = int(tfidf_matrix.get_shape()[0] / 2)
        tfidf_array = tfidf_matrix.todense()

        tfidf_holder['tfidf_1'] = [row.tolist()[0] for row in tfidf_array[0:split_index]]
        tfidf_holder['tfidf_2'] = [row.tolist()[0] for row in tfidf_array[split_index:]]

        X[features.TFIDF_COS] = tfidf_holder.apply(
            lambda row: cosine_similarity([row['tfidf_1'], row['tfidf_2']])[0, 1], axis=1)
        logger.debug('TfidfTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __join_definitions(col1, col2):
        joined_definitions = pd.concat([col1, col2])
        return joined_definitions.apply(lambda tokens: ' '.join(tokens)).values.T


class MatchingLemmaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X[features.LEMMA_MATCH] = X.apply(
            lambda row: self.__matching_lemma_normalized(row['lemmatized_1'], row['lemmatized_2']), axis=1)

        logger.debug('MatchingLemmaTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def __matching_lemma_normalized(self, doc1, doc2):
        lemma_1_list = [token.text for token in doc1 if token.is_stop is not True and token.is_punct is not True]
        lemma_2_list = [token.text for token in doc2 if token.is_stop is not True and token.is_punct is not True]

        combined_length = (len(lemma_1_list) + len(lemma_2_list))

        if combined_length == 0:
            return 0.0

        return len(self.__intersection(lemma_1_list, lemma_2_list)) / combined_length


class ToTargetSimilarityDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        result = X.apply(lambda row: self.__calculate_max_similarity(row['word_processed'][0], row['processed_1']),
                         axis=1)
        result2 = X.apply(lambda row: self.__calculate_max_similarity(row['word_processed'][0], row['processed_2']),
                          axis=1)
        X[features.SIMILARITY_DIFF_TO_TARGET] = result - result2

        logger.debug('ToTargetSimilarityDiffTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __calculate_max_similarity(target, spacy_doc):
        similarities = []
        for token in spacy_doc:
            if token.is_stop is False:
                similarities.append(target.similarity(token))
        if len(similarities) == 0:
            return 0.0
        return max(similarities)


class DifferenceInLengthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    @staticmethod
    def __difference_in_length(col1, col2):
        return abs(len(col1.split(' ')) - len(col2.split(' ')[0]))

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, features.LEN_DIFF] = X.apply(
            lambda row: self.__difference_in_length(row['def1'], row['def2']), axis=1)

        logger.debug('DifferenceInLengthTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X


class AvgSynsetCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, 'synset_count_1'] = X['processed_1'].map(lambda doc: self.__count_avg_synset(doc))
        X.loc[:, 'synset_count_2'] = X['processed_2'].map(lambda doc: self.__count_avg_synset(doc))
        X.loc[:, features.SYNSET_COUNT_DIFF] = X['synset_count_1'] - X['synset_count_2']

        logger.debug('AvgSynsetCountTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def remove_stopwords(doc, output='text'):
        # TODO: ADD 'etc' to stopwords list
        if output == 'token':
            return [token for token in doc if token.is_stop is not True and token.is_punct is not True]

        return [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]

    def __count_avg_synset(self, doc):
        doc_sw_removed = self.remove_stopwords(doc, 'token')
        count = 0
        for token in doc_sw_removed:
            count = count + len(token._.wordnet.synsets())

        return count / len(doc)


class CountEachPosTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        result = self.__count_pos(X['processed_1'], X['processed_2'])
        for pos in set(result.columns):
            X.loc[:, pos] = result[pos]

        logger.debug('CountEachPosTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __count_pos(col1, col2):
        pos_counter = pd.DataFrame()

        for index, doc in col1.items():
            for token in doc:
                if token.pos_ in pos_counter.columns:
                    pos_counter[token.pos_][index] = pos_counter[token.pos_][index] + 1
                else:
                    pos_counter[token.pos_] = pd.Series(0, index=col1.index)
                    pos_counter[token.pos_][index] = pos_counter[token.pos_][index] + 1

        for index, doc in col2.items():
            for token in doc:
                if token.pos_ in pos_counter.columns:
                    pos_counter[token.pos_][index] = pos_counter[token.pos_][index] - 1
                else:
                    pos_counter[token.pos_] = pd.Series(0, index=col2.index)
                    pos_counter[token.pos_][index] = pos_counter[token.pos_][index] - 1

        for pos in pos_counter.columns:
            pos_counter[pos] = preprocessing.scale(pos_counter[pos])
        return pos_counter


class TargetWordSynsetCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._tag_map = {'adjective': wn.ADJ,
                         'adverb': wn.ADV,
                         'conjunction': None,
                         'interjection': None,
                         'noun': wn.NOUN,
                         'number': None,
                         'preposition': wn.ADV,
                         'pronoun': None,
                         'verb': wn.VERB}

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X.loc[:, features.TARGET_WORD_SYNSET_COUNT] = X.apply(lambda row: self.__targetword_synset_count(row), axis=1)

        logger.debug('TargetWordSynsetCountTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    def __targetword_synset_count(self, row):
        return len(wn.synsets(row['word'], self._tag_map[row['pos']]))


class SemicolonCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.semicolon_mean_1 = None
        self.semicolon_mean_2 = None

    def fit(self, X, y=None):
        self.semicolon_mean_1 = X['processed_1'].map(lambda doc: self.__count_semicolon(doc)).mean()
        self.semicolon_mean_2 = X['processed_2'].map(lambda doc: self.__count_semicolon(doc)).mean()

        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, 'semicol_count1_norm'] = 0.0 if self.semicolon_mean_1 == 0.0 else X['processed_1'].map(
            lambda doc: self.__count_semicolon(doc))
        X.loc[:, 'semicol_count2_norm'] = 0.0 if self.semicolon_mean_2 == 0.0 else X['processed_2'].map(
            lambda doc: self.__count_semicolon(doc))

        X.loc[:, features.SEMICOLON_DIFF] = X['semicol_count1_norm'] - X['semicol_count2_norm']

        logger.debug('SemicolonCountTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    def __count_semicolon(self, doc):
        return len([token for token in doc if token.text == ';'])


class TokenCountNormalizedDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.token_count_mean_1 = None
        self.token_count_mean_2 = None

    def fit(self, X, y=None):
        self.token_count_mean_1 = X['processed_1'].map(lambda doc: len(doc)).mean()
        self.token_count_mean_2 = X['processed_2'].map(lambda doc: len(doc)).mean()

        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X['token_count_1'] = X['processed_1'].map(lambda doc: len(doc))
        X['token_count_2'] = X['processed_2'].map(lambda doc: len(doc))

        X['token_count_1_norm'] = X['token_count_1'] / self.token_count_mean_1
        X['token_count_2_norm'] = X['token_count_2'] / self.token_count_mean_2
        X.loc[:, 'token_count_norm_diff'] = X['token_count_1_norm'] - X['token_count_2_norm']

        logger.debug('TokenCountNormalizedDiffTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X


class MaxDependencyTreeDepthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X.loc[:, 'max_depth_deptree_1'] = X['processed_1'].map(lambda doc: self.__max_dep_tree_depth(doc))
        X.loc[:, 'max_depth_deptree_2'] = X['processed_2'].map(lambda doc: self.__max_dep_tree_depth(doc))

        logger.debug('MaxDependencyTreeDepthTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    def __traverse_to_root(self, token, depth):
        if token.dep_ == 'ROOT':
            return depth
        else:
            return self.__traverse_to_root(token.head, depth + 1)

    def __max_dep_tree_depth(self, doc):
        max_deptree_depth = []
        for token in doc:
            max_deptree_depth.append(self.__traverse_to_root(token, 0))

        return max(max_deptree_depth)


class JaccardTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X.loc[:, features.JACCARD] = X.apply(
            lambda row: self.__get_jaccard_sim(row['def1'], row['def2']), axis=1)

        logger.debug('JaccardTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))


class CosineTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()

        X.loc[:, features.COSINE] = X.apply(
            lambda row: self.__cosine(row['def1'], row['def2']), axis=1)

        logger.debug('CosineTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    def __cosine(self, col1, col2):
        return self.__get_cosine_sim(col1, col2)[0, 1]

    def __get_cosine_sim(self, *strs):
        vectors = [t for t in self.__get_vectors(*strs)]
        return cosine_similarity(vectors)

    def __get_vectors(self, *strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

class MostDescriptiveWordsProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):

        self.vocab_dict = Counter()

        for sentence in X["def1"]:
            self.vocab_dict.update(word.strip('.,?!"\'').lower() for word in sentence.split())
        for sentence in X["def2"]:
            self.vocab_dict.update(word.strip('.,?!"\'').lower() for word in sentence.split())
            
        return self

    def transform(self, X, y=None):
        X.loc[:, "filtered_def1"] = X.apply(
        lambda row: self.get_least_n_words(row['def1'], 5), axis=1)

        X.loc[:, "filtered_def2"] = X.apply(
        lambda row: self.get_least_n_words(row['def2'], 5), axis=1)

        return X


    def get_least_n_words(self, definition, n):
        vocab_def = dict()
        for word in definition.lower().split():
            word = word.strip('.,?!"\':')
            vocab_def[word] = self.vocab_dict[word]
        vocab_def = sorted(vocab_def.items(), key=lambda item: item[1], reverse=False)
        vocab_def = [item[0] for item in vocab_def]
        return vocab_def[:n]

class MeanCosineSimilarityProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.w2v_model_path = '/home/varya/data/models/skipgram_russian_300_5_2019/model.bin'
        self.tag_model_path = '/home/varya/data/models/russian-syntagrus-ud-2.0-170801.udpipe'

        self.model_w2v = None
        self.model_tag = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.model_w2v = gensim.models.KeyedVectors.load_word2vec_format(self.w2v_model_path, binary=True)
        self.model_tag = Model.load(self.tag_model_path)

        t0 = time.time()

        X.loc[:, features.MEANCOSINE] = X.apply(
            lambda row: self.compute_similarity_based_on_mean_vectors(row['filtered_def1'], row['filtered_def2']), axis=1)

        logger.debug('MeanCosineSimilarityProcessor.transform() took %.3f seconds' % (time.time() - t0))

        self.model_w2v = None
        self.model_tag = None

        return X

    def tag(self, word):
        pipeline = Pipeline(self.model_tag, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        processed = pipeline.process(word)
        output = [l for l in processed.split('\n') if not l.startswith('#')]
        tagged = ['_'.join(w.split('\t')[2:4]) for w in output if w]
        return tagged

    # def tagged_defs(self, string_of_words):
    #     tagged_data = []
    #     #tagged_data = ""
    #     for word in string_of_words.split():
    #         word_tagged = self.tag(word.strip('123456789|\(\).,?!"\'').lower())
    #         tagged_data += word_tagged
    #     return tagged_data
    
    def tagged_defs(self, list_of_words):
        tagged_data = []
        for word in list_of_words:
            word_tagged = self.tag(word.strip('|\(\).,?!"\'').lower())
            tagged_data += word_tagged
        return tagged_data

    def compute_similarity_based_on_mean_vectors(self, tdef1, tdef2):
        word_vectors_def1 = [self.model_w2v.get_vector(word) for word in self.tagged_defs(tdef1) if word in self.model_w2v.key_to_index]
        word_vectors_def2 = [self.model_w2v.get_vector(word) for word in self.tagged_defs(tdef2) if word in self.model_w2v.key_to_index]

        if len(word_vectors_def1) == 0 or len(word_vectors_def2) == 0:
            return 0.0

        # mean_vector_1 = np.zeros(word_vectors_def1[0].shape)
        # mean_vector_2 = np.zeros(word_vectors_def1[0].shape)

        # for word_vector in word_vectors_def1:
        #     mean_vector_1 += word_vector
        # mean_vector_1 /= len(word_vectors_def1)

        # for word_vector in word_vectors_def2:
        #     mean_vector_2 += word_vector
        # mean_vector_2 /= len(word_vectors_def2)
    
        # dot_def1_def1 = mean_vector_1.dot(mean_vector_1.T)
        # dot_def2_def2 = mean_vector_2.dot(mean_vector_2.T)
        # dot_def1_def2 = mean_vector_1.dot(mean_vector_2.T)
        # return dot_def1_def2/ np.sqrt(dot_def1_def1*dot_def2_def2)

        word_vectors_def1_mean = np.mean(word_vectors_def1, axis = 0)
        word_vectors_def2_mean = np.mean(word_vectors_def2, axis = 0)

        
        return 1. - distance.cosine(word_vectors_def1_mean, word_vectors_def2_mean)

    def compute_similarity(self, def1, def2):
        list_cos_sim_def1_def2 = []
        max_cos = 0
        word_vectors_def1 = [self.model_w2v.get_vector(word) for word in self.tagged_defs(def1) if word in self.model_w2v.key_to_index]
        word_vectors_def2 = [self.model_w2v.get_vector(word) for word in self.tagged_defs(def2) if word in self.model_w2v.key_to_index]

        for word_vector_1 in word_vectors_def1:
            for word_vector_2 in word_vectors_def2:
                cos_sim = 1. - distance.cosine(word_vector_1, word_vector_2)
                print(cos_sim)
                if cos_sim > max_cos:
                    max_cos = cos_sim
                    print(max_cos)
            list_cos_sim_def1_def2.append(max_cos)
        
        if len(list_cos_sim_def1_def2) == 0:
            return 0.0

        return sum(list_cos_sim_def1_def2)/(len(list_cos_sim_def1_def2))

#it does not work 
class WordMoverSimilarityProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        #self.w2v_model_path = '/home/varya/data/models/skipgram_russian_300_5_2019/model.bin'
        self.tag_model_path = '/home/varya/data/models/russian-syntagrus-ud-2.0-170801.udpipe'

        self.model_w2v = None
        self.model_tag = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.model_w2v = api.load("word2vec-ruscorpora-300")
        self.model_tag = Model.load(self.tag_model_path)

        t0 = time.time()

        X.loc[:, features.WORDMOVERSIMILARITY] = X.apply(
            lambda row: self.compute_similarity(row['filtered_def1'], row['filtered_def2']), axis=1)

        logger.debug('WordMoverSimilarityProcessor.transform() took %.3f seconds' % (time.time() - t0))

        self.model_w2v = None
        self.model_tag = None

        return X

    def tag(self, word):
        pipeline = Pipeline(self.model_tag, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        processed = pipeline.process(word)
        output = [l for l in processed.split('\n') if not l.startswith('#')]
        tagged = ['_'.join(w.split('\t')[2:4]) for w in output if w]
        return tagged
    
    def tagged_defs(self, list_of_words):
        tagged_data = ''
        for word in list_of_words:
            word_tagged = self.tag(word.strip('|\(\).,?!"\'').lower())
            tagged_data.join(word_tagged)
        return tagged_data

    def compute_similarity(self, tdef1, tdef2):
        
        return self.model_w2v.wmdistance(self.tagged_defs(tdef1), self.tagged_defs(tdef2))

class MeanCosineSimilarityGloveProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_glove_path = "/home/varya/data/models/navec_hudlit_v1_12B_500K_300d_100q.tar"

        self.model_glove = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.model_glove = Navec.load(self.model_glove_path)

        t0 = time.time()

        X.loc[:, features.MEANCOSINEGLOVE] = X.apply(
            lambda row: self.compute_similarity_based_on_mean_vectors(row["filtered_def1"], row["filtered_def1"]), axis=1)

        logger.debug('MeanCosineSimilarityGloveProcessor.transform() took %.3f seconds' % (time.time() - t0))

        self.model_glove = None

        return X


    def compute_similarity_based_on_mean_vectors(self, tdef1, tdef2):
        word_vectors_def1 = [self.model_glove[word] for word in tdef1 if word in self.model_glove]
        word_vectors_def2 = [self.model_glove[word] for word in tdef2 if word in self.model_glove]

        if len(word_vectors_def1) == 0 or len(word_vectors_def2) == 0:
            return 0.0

        word_vectors_def1_mean = np.mean(word_vectors_def1, axis = 0)
        word_vectors_def2_mean = np.mean(word_vectors_def2, axis = 0)

        
        return 1. - distance.cosine(word_vectors_def1_mean, word_vectors_def2_mean)

class MeanCosineSimilarityFasttextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model_fasttext_path = "/home/varya/data/models/fasttext_original_ru_cc.ru.300.bin.gz/cc.ru.300.bin"

        self.model_fasttext = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.model_fasttext = fasttext.load_model(self.model_fasttext_path)

        t0 = time.time()

        X.loc[:, features.MEANCOSINEFASTTEXT] = X.apply(
            lambda row: self.compute_similarity(row['def1'], row['def2']), axis=1)

        logger.debug('MeanCosineSimilarityFasttextProcessor.transform() took %.3f seconds' % (time.time() - t0))

        self.model_fasttext = None

        return X


    def compute_similarity_based_on_mean_vectors(self, def1, def2):

        word_vectors_def1 = [self.model_fasttext.get_word_vector(word) for word in def1.split() if word in self.model_fasttext]
        word_vectors_def2 = [self.model_fasttext.get_word_vector(word) for word in def2.split() if word in self.model_fasttext]

        if len(word_vectors_def1) == 0 or len(word_vectors_def2) == 0:
            return 0.0
        word_vectors_def1_mean = np.mean(word_vectors_def1, axis = 0)
        word_vectors_def2_mean = np.mean(word_vectors_def2, axis = 0)

        
        return 1. - distance.cosine(word_vectors_def1_mean, word_vectors_def2_mean)
    
    def compute_similarity(self, def1, def2):
        list_cos_sim_def1_def2 = []
        max_cos = 0
        word_vectors_def1 = [self.model_fasttext.get_word_vector(word) for word in def1.split() if word in self.model_fasttext]
        word_vectors_def2 = [self.model_fasttext.get_word_vector(word) for word in def2.split() if word in self.model_fasttext]

        for word_vector_1 in word_vectors_def1:
            for word_vector_2 in word_vectors_def2:
                cos_sim = 1. - distance.cosine(word_vector_1, word_vector_2)
                print(cos_sim)
                if cos_sim > max_cos:
                    max_cos = cos_sim
                    print(max_cos)
            list_cos_sim_def1_def2.append(max_cos)
        
        if len(list_cos_sim_def1_def2) == 0:
            return 0.0

        return sum(list_cos_sim_def1_def2)/(len(list_cos_sim_def1_def2))
class DiffPosCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        X.loc[:, features.POS_COUNT_DIFF] = X.apply(
            lambda row: self.__diff_pos_count(row['processed_1'], row['processed_2']),
            axis=1)
        logger.debug('DiffPosCountTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    @staticmethod
    def __diff_pos_count(col1, col2):
        pos_def1 = list(set([token.pos for token in col1]))
        pos_def2 = list(set([token.pos for token in col2]))

        return len(pos_def1) - len(pos_def2)

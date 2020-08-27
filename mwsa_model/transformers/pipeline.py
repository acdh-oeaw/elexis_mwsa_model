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

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def lemmatizer(doc, spacy_model):
    doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
    return spacy_model.make_doc(u' '.join(doc))


def remove_stopwords(doc, output='text'):
    # TODO: ADD 'etc' to stopwords list
    if output == 'token':
        return [token for token in doc if token.is_stop is not True and token.is_punct is not True]

    return [token.text for token in doc if token.is_stop is not True and token.is_punct is not True]


class Error(Exception):
    pass


class UnsupportedSpacyModelError(Error):

    def __init__(self, message):
        self.message = message


class SpacyProcessor(BaseEstimator, TransformerMixin):
    spacy_models = {
        SupportedLanguages.English: 'en_core_web_md',
        SupportedLanguages.German: 'de_core_news_sm'
    }

    def __init__(self, lang=None, with_wordnet=False):
        self.logger = logging.getLogger(__name__)
        self.lang = lang
        self.with_wordnet = with_wordnet

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0 = time.time()
        try:
            nlp = spacy.load(self.spacy_models[self.lang])
            if self.with_wordnet:
                nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
        except KeyError:
            raise UnsupportedSpacyModelError("No Spacy Language model exists for language " + str(self.lang))

        X.loc[:, 'processed_1'] = X['def1'].map(nlp)
        X.loc[:, 'processed_2'] = X['def2'].map(nlp)
        X.loc[:, 'word_processed'] = X['word'].map(nlp)
        X.loc[:, 'lemmatized_1'] = X['processed_1'].map(lambda doc: lemmatizer(doc, nlp))
        X.loc[:, 'stopwords_removed_1'] = X['lemmatized_1'].map(remove_stopwords)
        X.loc[:, 'lemmatized_2'] = X['processed_2'].map(lambda doc: lemmatizer(doc, nlp))
        X.loc[:, 'stopwords_removed_2'] = X['lemmatized_2'].map(remove_stopwords)
        self.logger.debug(X)

        logger.debug('SpacyProcessor.transform() took %.3f seconds' % (time.time() - t0))

        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0=time.time()
        for col in X.columns:
            if col in ['word', 'pos', 'def1', 'def2',
                       'processed_1', 'processed_2', 'word_processed',
                       'lemmatized_1', 'stopwords_removed_1', 'lemmatized_2',
                       'stopwords_removed_2', 'relation']:
                X = X.drop(col, axis=1)
        logger.debug('FeatureSelector.transform() took %.3f seconds' % (time.time() - t0))

        return X


class FirstWordSameProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0=time.time()
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
        t0=time.time()
        X.loc[:, features.SIMILARITY] = X.apply(
            lambda row: row['processed_1'].similarity(row['processed_2']), axis=1)
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
        t0=time.time()
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
        t0=time.time()
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

        X[features.TFIDF_COS] = tfidf_holder.apply(lambda row: cosine_similarity([row['tfidf_1'], row['tfidf_2']])[0, 1], axis=1)
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
        X[features.LEMMA_MATCH] = X.apply(
            lambda row: self.__matching_lemma_normalized(row['lemmatized_1'], row['lemmatized_2']), axis=1)

        return X

    @staticmethod
    def __intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def __matching_lemma_normalized(self, doc1, doc2):
        t0=time.time()
        lemma_1_list = [token.text for token in doc1 if token.is_stop is not True and token.is_punct is not True]
        lemma_2_list = [token.text for token in doc2 if token.is_stop is not True and token.is_punct is not True]

        combined_length = (len(lemma_1_list) + len(lemma_2_list))

        if combined_length == 0:
            return 0.0

        logger.debug('MatchingLemmaTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return len(self.__intersection(lemma_1_list, lemma_2_list)) / combined_length


class ToTargetSimilarityDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0=time.time()
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
        t0=time.time()
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
        t0=time.time()
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
        t0=time.time()
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
        t0=time.time()

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
        t0=time.time()
        X.loc[:, 'semicol_count1_norm'] = 0.0 if self.semicolon_mean_1 == 0.0 else X['processed_1'].map(lambda doc: self.__count_semicolon(doc))
        X.loc[:, 'semicol_count2_norm'] = 0.0 if self.semicolon_mean_2 == 0.0 else X['processed_2'].map(lambda doc: self.__count_semicolon(doc))

        X.loc[:, features.SEMICOLON_DIFF] = X['semicol_count1_norm'] - X['semicol_count2_norm']

        logger.debug('SemicolonCountTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

    def __count_semicolon(self, doc):
        return len([token for token in doc if token.text is ';'])


class TokenCountNormalizedDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.token_count_mean_1 = None
        self.token_count_mean_2 = None

    def fit(self, X, y=None):
        self.token_count_mean_1 = X['processed_1'].map(lambda doc: len(doc)).mean()
        self.token_count_mean_2 = X['processed_2'].map(lambda doc: len(doc)).mean()

        return self

    def transform(self, X, y=None):
        t0=time.time()

        X['token_count_1'] = X['processed_1'].map(lambda doc: len(doc))
        X['token_count_2'] = X['processed_2'].map(lambda doc: len(doc))

        X['token_count_1_norm'] = X['token_count_1']/self.token_count_mean_1
        X['token_count_2_norm'] = X['token_count_2']/self.token_count_mean_2
        X.loc[:, 'token_count_norm_diff'] = X['token_count_1_norm']-X['token_count_2_norm']

        logger.debug('TokenCountNormalizedDiffTransformer.transform() took %.3f seconds' % (time.time() - t0))

        return X

class MaxDependencyTreeDepthTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0=time.time()

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
        t0=time.time()

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
        t0=time.time()

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

class DiffPosCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        t0= time.time()
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

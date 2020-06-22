import pandas as pd
import spacy
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from mwsa.service.util import SupportedLanguages
import features


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
        SupportedLanguages.English: 'en_core_web_lg'
    }

    def __init__(self, lang=None):
        self.logger = logging.getLogger(__name__)
        self.lang = lang

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            nlp = spacy.load(self.spacy_models[self.lang])
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
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            if col in ['word', 'pos', 'def1', 'def2',
                       'processed_1', 'processed_2', 'word_processed',
                       'lemmatized_1', 'stopwords_removed_1', 'lemmatized_2',
                       'stopwords_removed_2', 'relation']:
                X = X.drop(col, axis=1)

        return X


class FirstWordSameProcessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X[features.FIRST_WORD_SAME] = X.apply(
            lambda row: self.__first_word_same(row['def1'], row['def2']), axis=1)
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
        X[features.SIMILARITY] = X.apply(
            lambda row: row['processed_1'].similarity(row['processed_2']), axis=1)
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
        pos_numpy = X['pos'].to_numpy().reshape(-1, 1)
        encoded_array = self.encoder.transform(pos_numpy).toarray()

        encoded_dataframe = pd.DataFrame(data=encoded_array[0:, 0:],
                                         index=X.index,
                                         columns=self.encoder.categories_[0])

        return pd.concat([X, encoded_dataframe], axis=1)


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
        lemma_1_list = [token.text for token in doc1 if token.is_stop is not True and token.is_punct is not True]
        lemma_2_list = [token.text for token in doc2 if token.is_stop is not True and token.is_punct is not True]

        combined_length = (len(lemma_1_list) + len(lemma_2_list))

        if combined_length == 0:
            return 0.0

        return len(self.__intersection(lemma_1_list, lemma_2_list)) / combined_length


class DiffPosCountTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X.loc[:, features.POS_COUNT_DIFF] = X.apply(
            lambda row: self.__diff_pos_count(row['processed_1'], row['processed_2']),
            axis=1)

        return X

    @staticmethod
    def __diff_pos_count(col1, col2):
        pos_def1 = list(set([token.pos for token in col1]))
        pos_def2 = list(set([token.pos for token in col2]))

        return len(pos_def1) - len(pos_def2)

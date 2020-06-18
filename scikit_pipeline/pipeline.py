import spacy
import logging
from sklearn.base import BaseEstimator, TransformerMixin

from mwsa.util import SupportedLanguages
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

        X['processed_1'] = X['def1'].map(nlp)
        X['processed_2'] = X['def2'].map(nlp)
        X['word_processed'] = X['word'].map(nlp)
        X['lemmatized_1'] = X['processed_1'].map(lambda doc: lemmatizer(doc, nlp))
        X['stopwords_removed_1'] = X['lemmatized_1'].map(remove_stopwords)
        X['lemmatized_2'] = X['processed_2'].map(lambda doc: lemmatizer(doc, nlp))
        X['stopwords_removed_2'] = X['lemmatized_2'].map(remove_stopwords)

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

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

import features


class BaseFeatureExtractor:
    def extract(self, data, feats):
        pass


class FirstWordSame(BaseFeatureExtractor):
    def __init__(self):
        pass

    @staticmethod
    def __first_word_same(col1, col2):
        return col1.split(' ')[0].lower() == col2.split(' ')[0].lower()

    def extract(self, data, feats):
        feats[features.FIRST_WORD_SAME] = data.apply(
            lambda row: FirstWordSame.__first_word_same(row['def1'], row['def2']), axis=1)
        return feats


class AvgSynsetCountExtractor(BaseFeatureExtractor):
    def __init__(self):
        pass

    @staticmethod
    def remove_stopwords(doc, output='text'):
        # TODO: ADD 'etc' to stopwords list
        if output == 'token':
            return [token for token in doc if token.is_stop != True and token.is_punct != True]

        return [token.text for token in doc if token.is_stop != True and token.is_punct != True]

    def __count_avg_synset(self, doc):
        doc_sw_removed = self.remove_stopwords(doc, 'token')
        count = 0
        for token in doc_sw_removed:
            count = count + len(token._.wordnet.synsets())

        return count / len(doc)

    def extract(self, data, feats):
        feats['synset_count_1'] = data['processed_1'].map(lambda doc: self.__count_avg_synset(doc))
        feats['synset_count_2'] = data['processed_2'].map(lambda doc: self.__count_avg_synset(doc))


class SimilarityExtractor(BaseFeatureExtractor):
    def __init__(self):
        pass

    def extract(self, data, feats):
        feats[features.SIMILARITY] = data.apply(
            lambda row: row['processed_1'].similarity(row['processed_2']), axis=1)


class DifferenceInLengthExtractor(BaseFeatureExtractor):
    @staticmethod
    def __difference_in_length(col1, col2):
        return abs(len(col1.split(' ')) - len(col2.split(' ')[0]))

    def extract(self, data, feats):
        feats[features.LEN_DIFF] = data.apply(
            lambda row: self.__difference_in_length(row['def1'], row['def2']), axis=1)


class CosineExtractor(BaseFeatureExtractor):

    def __cosine(self, col1, col2):
        return self.__get_cosine_sim(col1, col2)[0, 1]

    def __get_cosine_sim(self, *strs):
        vectors = [t for t in self.__get_vectors(*strs)]
        return cosine_similarity(vectors)

    @staticmethod
    def __get_vectors(*strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    def extract(self, data, feats):
        feats[features.COSINE] = data.apply(
            lambda row: self.__cosine(row['def1'], row['def2']), axis=1)


class JaccardExtractor(BaseFeatureExtractor):
    @staticmethod
    def __get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def extract(self, data, feats):
        feats[features.JACCARD] = data.apply(
            lambda row: self.__get_jaccard_sim(row['def1'], row['def2']), axis=1)


class DiffPosCountExtractor(BaseFeatureExtractor):

    @staticmethod
    def __diff_pos_count(col1, col2):
        pos_def1 = list(set([token.pos for token in col1]))
        pos_def2 = list(set([token.pos for token in col2]))

        return len(pos_def1) - len(pos_def2)

    def extract(self, data, feats):
        feats[features.POS_COUNT_DIFF] = data.apply(
            lambda row: self.__diff_pos_count(row['processed_1'], row['processed_2']),
            axis=1)


class MatchingLemmaExtractor(BaseFeatureExtractor):

    @staticmethod
    def __intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    def __matching_lemma_normalized(self, doc1, doc2):
        lemma_1_list = [token.text for token in doc1 if token.is_stop != True and token.is_punct != True]
        lemma_2_list = [token.text for token in doc2 if token.is_stop != True and token.is_punct != True]

        combined_length = (len(lemma_1_list) + len(lemma_2_list))

        if combined_length == 0:
            return 0.0

        return len(self.__intersection(lemma_1_list, lemma_2_list)) / combined_length

    def extract(self, data, feats):
        feats[features.LEMMA_MATCH] = data.apply(
            lambda row: self.__matching_lemma_normalized(row['lemmatized_1'], row['lemmatized_2']), axis=1)


class TfIdfExtractor(BaseFeatureExtractor):

    @staticmethod
    def __join_definitions(col1, col2):
        joined_definitions = pd.concat([col1, col2])
        return joined_definitions.apply(lambda tokens: ' '.join(tokens)).values.T

    def __tfidf(self, col1, col2):
        tfidf_holder = pd.DataFrame()
        tfidf_holder['col1'] = col1
        tfidf_holder['col2'] = col2

        values = self.__join_definitions(col1, col2)
        tfidf_holder['tfidf_1'], tfidf_holder['tfidf_2'] = self.__tfidf_vectors(values)

        return tfidf_holder.apply(lambda row: cosine_similarity([row['tfidf_1'], row['tfidf_2']])[0, 1], axis=1)

    @staticmethod
    def __tfidf_vectors(values):
        tfidf_matrix = TfidfVectorizer().fit_transform(values)

        split_index = int(tfidf_matrix.get_shape()[0] / 2)
        tfidf_array = tfidf_matrix.todense()

        df_result1 = [row.tolist()[0] for row in tfidf_array[0:split_index]]
        df_result2 = [row.tolist()[0] for row in tfidf_array[split_index:]]

        return df_result1, df_result2

    def extract(self, data, feats):
        feats[features.TFIDF_COS] = self.__tfidf(data['stopwords_removed_1'],
                                                 data['stopwords_removed_2'])


class OneHotPosExtractor(BaseFeatureExtractor):

    @staticmethod
    def __one_hot_encode(dataset):
        pos_numpy = dataset['pos'].to_numpy().reshape(-1, 1)
        encoder = OneHotEncoder(handle_unknown='ignore') \
            .fit(pos_numpy)

        encoded_array = encoder.transform(pos_numpy).toarray()

        encoded_dataframe = pd.DataFrame(data=encoded_array[0:, 0:],
                                         index=dataset.index,
                                         columns=encoder.categories_[0])

        return pd.concat([dataset, encoded_dataframe], axis=1)

    def __one_hot_pos(self, data, feat):
        data = self.__one_hot_encode(data)
        for pos in set(data['pos']):
            feat[pos] = data[pos]

    def extract(self, data, feats):
        self.__one_hot_pos(data, feats)


class CountEachPosExtractor(BaseFeatureExtractor):

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

    def extract(self, data, feats):
        result = self.__count_pos(data['processed_1'], data['processed_2'])
        for pos in set(result.columns):
            feats[pos] = result[pos]


class FeatureExtractor:
    def __init__(self, feature_extractors=[]):
        self._feature_extractors = feature_extractors
        self.feats = pd.DataFrame()

    def similarity(self):
        self._feature_extractors.append(SimilarityExtractor())
        return self

    def first_word(self):
        self._feature_extractors.append(FirstWordSame())
        return self

    def difference_in_length(self):
        self._feature_extractors.append(DifferenceInLengthExtractor())
        return self

    def jaccard(self):
        self._feature_extractors.append(JaccardExtractor())
        return self

    def cosine(self):
        self._feature_extractors.append(CosineExtractor())
        return self

    def diff_pos_count(self):
        self._feature_extractors.append(DiffPosCountExtractor())
        return self

    def matching_lemma(self):
        self._feature_extractors.append(MatchingLemmaExtractor())
        return self

    def tfidf(self):
        self._feature_extractors.append(TfIdfExtractor())
        return self

    def ont_hot_pos(self):
        self._feature_extractors.append(OneHotPosExtractor())
        return self

    def count_each_pos(self):
        self._feature_extractors.append(CountEachPosExtractor())
        return self

    def avg_count_synsets(self):
        self._feature_extractors.append(AvgSynsetCountExtractor())
        return self

    def extract(self, data, feats_to_scale):
        for extractor in self._feature_extractors:
            extractor.extract(data, self.feats)

        if len(feats_to_scale) > 0:
            for c_name in feats_to_scale:
                self.feats[c_name] = preprocessing.scale(self.feats[c_name])

        return self.feats

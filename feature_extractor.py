import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

import features


class FeatureExtractor:
    def __init__(self, feats_to_scale):
        self.feats_to_scale = feats_to_scale

    @staticmethod
    def __first_word_same(col1, col2):
        return col1.split(' ')[0].lower() == col2.split(' ')[0].lower()

    @staticmethod
    def __difference_in_length(col1, col2):
        return abs(len(col1.split(' ')) - len(col2.split(' ')[0]))

    @staticmethod
    def __jaccard_sim(col1, col2):
        return FeatureExtractor.__get_jaccard_sim(col1, col2)

    @staticmethod
    def __get_jaccard_sim(str1, str2):
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    @staticmethod
    def __cosine(col1, col2):
        return FeatureExtractor.__get_cosine_sim(col1, col2)[0, 1]

    @staticmethod
    def __get_cosine_sim(*strs):
        vectors = [t for t in FeatureExtractor.__get_vectors(*strs)]
        return cosine_similarity(vectors)

    @staticmethod
    def __get_vectors(*strs):
        text = [t for t in strs]
        vectorizer = CountVectorizer(text)
        vectorizer.fit(text)
        return vectorizer.transform(text).toarray()

    @staticmethod
    def __diff_pos_count(col1, col2):
        pos_def1 = list(set([token.pos for token in col1]))
        pos_def2 = list(set([token.pos for token in col2]))

        return len(pos_def1) - len(pos_def2)

    @staticmethod
    def __intersection(lst1, lst2):
        lst3 = [value for value in lst1 if value in lst2]
        return lst3

    @staticmethod
    def __matching_lemma_normalized(doc1, doc2):
        lemma_1_list = [token.text for token in doc1 if token.is_stop != True and token.is_punct != True]
        lemma_2_list = [token.text for token in doc2 if token.is_stop != True and token.is_punct != True]

        combined_length = (len(lemma_1_list) + len(lemma_2_list))

        if combined_length == 0:
            return 0.0

        return len(FeatureExtractor.__intersection(lemma_1_list, lemma_2_list)) / combined_length

    @staticmethod
    def __join_definitions(col1, col2):
        joined_definitions = pd.concat([col1, col2])
        return joined_definitions.apply(lambda tokens: ' '.join(tokens)).values.T

    @staticmethod
    def __tfidf(col1, col2):
        tfidf_holder = pd.DataFrame()
        tfidf_holder['col1'] = col1
        tfidf_holder['col2'] = col2

        values = FeatureExtractor.__join_definitions(col1, col2)
        tfidf_holder['tfidf_1'], tfidf_holder['tfidf_2'] = FeatureExtractor.__tfidf_vectors(values)

        return tfidf_holder.apply(lambda row: cosine_similarity([row['tfidf_1'], row['tfidf_2']])[0, 1], axis=1)

    @staticmethod
    def __tfidf_vectors(values):
        tfidf_matrix = TfidfVectorizer().fit_transform(values)

        split_index = int(tfidf_matrix.get_shape()[0] / 2)
        tfidf_array = tfidf_matrix.todense()

        df_result1 = [row.tolist()[0] for row in tfidf_array[0:split_index]]
        df_result2 = [row.tolist()[0] for row in tfidf_array[split_index:]]

        return df_result1, df_result2

    @staticmethod
    def __one_hot_pos(data, feat):
        data = FeatureExtractor.__one_hot_encode(data)

        feat[features.POS_ADJ] = data[features.POS_ADJ]
        feat[features.POS_ADV] = data[features.POS_ADV]
        feat[features.POS_CONJ] = data[features.POS_CONJ]
        feat[features.POS_INJ] = data[features.POS_INJ]
        feat[features.POS_N] = data[features.POS_N]
        feat[features.POS_NUM] = data[features.POS_NUM]
        feat[features.POS_PN] = data[features.POS_PN]
        feat[features.POS_PP] = data[features.POS_PP]
        feat[features.POS_V] = data[features.POS_V]

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

    def extract_features(self, data):
        feat = pd.DataFrame()

        feat[features.SIMILARITY] = data.apply(lambda row: row['processed_1'].similarity(row['processed_2']), axis=1)
        feat[features.FIRST_WORD_SAME] = data.apply(
            lambda row: FeatureExtractor.__first_word_same(row['def1'], row['def2']), axis=1)
        feat[features.LEN_DIFF] = data.apply(
            lambda row: FeatureExtractor.__difference_in_length(row['def1'], row['def2']), axis=1)
        feat[features.JACCARD] = data.apply(lambda row: FeatureExtractor.__jaccard_sim(row['def1'], row['def2']), axis=1)
        feat[features.COSINE] = data.apply(lambda row: FeatureExtractor.__cosine(row['def1'], row['def2']), axis=1)
        feat[features.POS_COUNT_DIFF] = data.apply(
            lambda row: FeatureExtractor.__diff_pos_count(row['processed_1'], row['processed_2']),
            axis=1)
        feat[features.LEMMA_MATCH] = data.apply(
            lambda row: FeatureExtractor.__matching_lemma_normalized(row['lemmatized_1'], row['lemmatized_2']), axis=1)
        feat[features.TFIDF_COS] = FeatureExtractor.__tfidf(data['stopwords_removed_1'], data['stopwords_removed_2'])
        FeatureExtractor.__one_hot_pos(data, feat)
        feat = pd.concat([feat, FeatureExtractor.__count_pos(data['processed_1'], data['processed_2'])], axis=1)

        self.scale_features(feat)

        return feat

    def scale_features(self, feat):
        if len(self.feats_to_scale) > 0:
            for c_name in self.feats_to_scale:
                feat[c_name] = preprocessing.scale(feat[c_name])

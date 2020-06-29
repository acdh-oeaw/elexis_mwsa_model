import pprint
import sys
import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from mwsa.service.model_trainer import MwsaModelTrainer
from mwsa.service.util import SupportedLanguages
from mwsa.transformers.pipeline import SpacyProcessor, SimilarityProcessor, FeatureSelector, \
    UnsupportedSpacyModelError, DiffPosCountTransformer, OneHotPosTransformer, MatchingLemmaTransformer, \
    CountEachPosTransformer, AvgSynsetCountTransformer, DifferenceInLengthTransformer, \
    ToTargetSimilarityDiffTransformer, MaxDependencyTreeDepthTransformer, TargetWordSynsetCountTransformer, \
    TokenCountNormalizedDiffTransformer, SemicolonCountTransformer, TfidfTransformer, CosineTransformer, \
    JaccardTransformer
from mwsa import features

pprint.pprint(sys.path)
sys.path.append('/Users/seungbinyim/Development/repos/elexis/mwsa_model')

data = {'word': ['test'], 'pos': ['noun'], 'def1': ['test definition'], 'def2': ['test definition 2']}
df = pd.DataFrame(data=data)
data_with_semicolon = {'word': ['test'], 'pos': ['noun'], 'def1': ['test ; definition'],
                       'def2': ['test ; definition 2']}
df_with_semicolon = pd.DataFrame(data=data_with_semicolon)


class TestMwsaModelTrainer:
    # TODO Parameterize this test
    def test_build_pipeline(self):
        model_trainer = MwsaModelTrainer()

        pipeline = model_trainer.build_pipeline(SupportedLanguages.English)

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_configure_grid_search(self):
        trainer = MwsaModelTrainer()
        pipeline = trainer.build_pipeline(SupportedLanguages.English)
        params = {
            'preprocess__lang': [SupportedLanguages.English],
            'random_forest__bootstrap': [True],
            'random_forest__class_weight': ['balanced', 'balanced_subsample'],
            'random_forest__max_depth': [30],
            'random_forest__max_features': ['auto'],
            'random_forest__min_samples_leaf': [3, 5],
            'random_forest__min_samples_split': [2],
            'random_forest__n_estimators': [300],
            'random_forest__n_jobs': [5]
        }

        model = trainer.configure_grid_serach(pipeline, params)
        assert model
        assert isinstance(model, GridSearchCV)

    def test_model_train(self):
        params = {
            'preprocess__lang': [SupportedLanguages.English],
            'random_forest__bootstrap': [True],
            'random_forest__class_weight': ['balanced', 'balanced_subsample'],
            'random_forest__max_depth': [30],
            'random_forest__max_features': ['auto'],
            'random_forest__min_samples_leaf': [1],
            'random_forest__min_samples_split': [2],
            'random_forest__n_estimators': [2],
            'random_forest__n_jobs': [5]
        }
        trainer = MwsaModelTrainer()
        pipeline = trainer.build_pipeline(SupportedLanguages.English)
        grid_search = trainer.configure_grid_serach(pipeline, params, cv=2)
        test_data = {'word': ['test', 'test2', 'test3', 'test4', 'test5'],
                     'pos': ['noun', 'noun', 'noun', 'noun', 'noun'],
                     'def1': ['test definition', 'test def 2', 'test def 3', 'test def 4', 'test def 5'],
                     'def2': ['test definition 2', 'test def 2', 'test def 3', 'test def 4', 'test def 5'],
                     'relation': ['exact', 'none', 'exact', 'related', 'broader']}
        test_df = pd.DataFrame(data=test_data)
        labels = test_df['relation']

        model = trainer.train(test_df, labels, grid_search)

        assert model
        assert model.best_estimator_


class TestTransformer:
    @pytest.fixture
    def spacy_processed(self):
        spacy = SpacyProcessor(lang=SupportedLanguages.English)

        return spacy.transform(df)

    @pytest.fixture
    def spacy_processed_with_semicolon(self):
        spacy = SpacyProcessor(lang=SupportedLanguages.English)

        return spacy.transform(df_with_semicolon)

    def test_count_each_pos_transformer(self, spacy_processed):
        count_pos_transformer = CountEachPosTransformer()

        transformed = count_pos_transformer.fit_transform(spacy_processed)

        assert 'NUM' in transformed.columns
        assert 'NOUN' in transformed.columns

    def test_avg_synset_count_transformer(self, spacy_processed):
        spacy = SpacyProcessor(lang=SupportedLanguages.English, with_wordnet=True)

        spacy_processed = spacy.transform(df)
        avg_synset_count_transformer = AvgSynsetCountTransformer()

        transformed = avg_synset_count_transformer.fit_transform(spacy_processed)

        assert features.SYNSET_COUNT_DIFF in transformed.columns

    def test_matching_lemma_transformer(self, spacy_processed):
        matching_lemma_transformer = MatchingLemmaTransformer()

        transformed = matching_lemma_transformer.fit_transform(spacy_processed)

        assert features.LEMMA_MATCH in transformed.columns
        transformed[features.LEMMA_MATCH].apply(lambda x: x <= 1.0)

    def test_one_hot_pos_transformer(self, spacy_processed):
        one_hot_transformer = OneHotPosTransformer()

        one_hot_transformer.fit(spacy_processed)
        transformed = one_hot_transformer.transform(spacy_processed)

        for pos in spacy_processed['pos']:
            assert pos in transformed.columns
            transformed[pos].apply(lambda x: x == 1.0)

    def test_similarity_transformer_IT(self, spacy_processed):
        similarity_processor = SimilarityProcessor()

        transformed = similarity_processor.transform(spacy_processed)

        assert features.SIMILARITY in transformed.columns

    def test_diff_in_length_transformer(self, spacy_processed):
        diff_in_length_transformer = DifferenceInLengthTransformer()

        transformed = diff_in_length_transformer.fit_transform(spacy_processed)

        assert features.LEN_DIFF in transformed.columns

    def test_to_target_similarity_diff_transformer(self, spacy_processed):
        to_target_similarity_diff_transformer = ToTargetSimilarityDiffTransformer()

        transformed = to_target_similarity_diff_transformer.fit_transform(spacy_processed)

        assert features.SIMILARITY_DIFF_TO_TARGET in transformed.columns
        for sim_diff in transformed[features.SIMILARITY_DIFF_TO_TARGET]:
            assert isinstance(sim_diff, float)

    def test_max_dependency_tree_depth_transformer(self, spacy_processed):
        max_dependency_tree_depth_transformer = MaxDependencyTreeDepthTransformer()

        transformed = max_dependency_tree_depth_transformer.fit_transform(spacy_processed)

        assert features.MAX_DEPTH_TREE_1 in transformed.columns
        assert features.MAX_DEPTH_TREE_2 in transformed.columns
        assert transformed[features.MAX_DEPTH_TREE_1].dtype == int

    def test_target_word_synset_count_transformer(self, spacy_processed):
        target_word_synset_count_transformer = TargetWordSynsetCountTransformer()

        transformed = target_word_synset_count_transformer.fit_transform(spacy_processed)

        assert features.TARGET_WORD_SYNSET_COUNT in transformed.columns
        assert transformed[features.TARGET_WORD_SYNSET_COUNT].dtype == int

    def test_fit_token_count_normalized_diff_transfomer(self, spacy_processed):
        token_count_normalized_diff_transfomer = TokenCountNormalizedDiffTransformer()

        transformer = token_count_normalized_diff_transfomer.fit(spacy_processed)

        assert transformer.token_count_mean_1 > 0.0
        assert transformer.token_count_mean_2 > 0.0

    def test_transform_token_count_normalized_diff_transfomer(self, spacy_processed):
        token_count_normalized_diff_transfomer = TokenCountNormalizedDiffTransformer()
        transformer = token_count_normalized_diff_transfomer.fit(spacy_processed)

        transformed = transformer.transform(spacy_processed)

        assert features.TOKEN_COUNT_NORM_DIFF in transformed.columns
        assert transformed[features.TOKEN_COUNT_NORM_DIFF].dtype == float

    def test_fit_semicolon_count_transformer(self, spacy_processed):
        semicolon_count_transformer = SemicolonCountTransformer()

        transformer = semicolon_count_transformer.fit(spacy_processed)

        assert transformer.semicolon_mean_1 is not None
        assert transformer.semicolon_mean_2 is not None

    def test_transform_semicolon_count_transformer_na(self, spacy_processed):
        semicolon_count_transformer = SemicolonCountTransformer()
        transformer = semicolon_count_transformer.fit(spacy_processed)

        transformed = transformer.transform(spacy_processed)

        assert features.SEMICOLON_DIFF in transformed.columns
        assert transformed[features.SEMICOLON_DIFF].dtype == float
        for val in transformed[features.SEMICOLON_DIFF]:
            assert not pd.isna(val)

    def test_transform_semicolon_count_transformer(self, spacy_processed_with_semicolon):
        semicolon_count_transformer = SemicolonCountTransformer()
        transformer = semicolon_count_transformer.fit(spacy_processed_with_semicolon)

        transformed = transformer.transform(spacy_processed_with_semicolon)

        assert features.SEMICOLON_DIFF in transformed.columns
        assert transformed[features.SEMICOLON_DIFF].dtype.kind in 'if'
        for val in transformed[features.SEMICOLON_DIFF]:
            assert not pd.isna(val)

    def test_tfidf_transformer_fit(self, spacy_processed):
        tfidf_transformer = TfidfTransformer()

        fitted = tfidf_transformer.fit(spacy_processed)

        assert fitted.tfidf_vectorizer is not None

    def test_tfidf_transformer_transform(self, spacy_processed):
        tfidf_transformer = TfidfTransformer()
        tfidf_transformer.fit(spacy_processed)

        transformed = tfidf_transformer.transform(spacy_processed)

        assert features.TFIDF_COS in transformed.columns
        assert transformed[features.TFIDF_COS].dtype.kind in 'if'
        for val in transformed[features.TFIDF_COS]:
            assert not pd.isna(val)

    def test_cosine_similarity_transform(self, spacy_processed):
        cosine_transformaer = CosineTransformer()
        cosine_transformaer.fit(spacy_processed)

        transformed = cosine_transformaer.transform(spacy_processed)

        assert features.COSINE in transformed.columns
        assert transformed[features.COSINE].dtype.kind in 'if'
        for val in transformed[features.COSINE]:
            assert val > 0.0

    def test_jaccard_transform(self, spacy_processed):
        jaccard_transformer = JaccardTransformer()

        transformed = jaccard_transformer.transform(spacy_processed)

        assert features.JACCARD in transformed.columns
        assert transformed[features.JACCARD].dtype.kind in 'if'
        for val in transformed[features.JACCARD]:
            assert val > 0.0

    def test_diff_pos_count(self, spacy_processed):
        diff_pos_counter = DiffPosCountTransformer()

        transformed = diff_pos_counter.transform(spacy_processed)

        assert features.POS_COUNT_DIFF in transformed.columns

    def test_feature_selector(self, spacy_processed):
        feature_selector = FeatureSelector()

        selected = feature_selector.transform(spacy_processed)

        columns = ['word', 'pos', 'def1', 'def2',
                   'processed_1', 'processed_2', 'word_processed',
                   'lemmatized_1', 'stopwords_removed_1', 'lemmatized_2',
                   'stopwords_removed_2', 'relation']
        for col in columns:
            assert col not in selected.columns


class TestSpacy:
    def test_spacy_transformer_IT(self):
        spacy = SpacyProcessor(lang=SupportedLanguages.English)
        expected_columns = ['processed_1',
                            'processed_2',
                            'word_processed',
                            'lemmatized_1',
                            'stopwords_removed_1',
                            'lemmatized_2',
                            'stopwords_removed_2']

        transformed = spacy.transform(df)

        assert isinstance(transformed, DataFrame)
        assert df.index.size == transformed.index.size

        for col in expected_columns:
            assert col in transformed

    def test_unsupported_spacy_model(self):
        spacy = SpacyProcessor(SupportedLanguages.Basque)

        with pytest.raises(UnsupportedSpacyModelError):
            spacy.transform(df)

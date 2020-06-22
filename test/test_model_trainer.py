import pandas as pd
import pytest
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from mwsa.service.model_trainer import MwsaModelTrainer
from mwsa.service.util import SupportedLanguages
from mwsa.transformers.pipeline import SpacyProcessor, SimilarityProcessor, FeatureSelector, \
    UnsupportedSpacyModelError, DiffPosCountTransformer, OneHotPosTransformer, MatchingLemmaTransformer, \
    CountEachPosTransformer, AvgSynsetCountTransformer
import features

data = {'word': ['test'], 'pos': ['noun'], 'def1': ['test definition'], 'def2': ['test definition 2']}
df = pd.DataFrame(data=data)


class Test_Mwsa_Model_Trainer:
    # TODO Parameterize this test
    def test_build_pipeline(self):
        model_trainer = MwsaModelTrainer()

        pipeline = model_trainer.build_pipeline(SupportedLanguages.English)

        assert isinstance(pipeline, Pipeline)
        assert len(pipeline.steps) > 0

    def test_configure_grid_search(self):
        trainer = MwsaModelTrainer()
        pipeline = trainer.build_pipeline(SupportedLanguages.English)
        data = {'word': ['test'], 'pos': ['noun'], 'def1': ['test definition'], 'def2': ['test definition 2']}
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
        data = {'word': ['test', 'test2', 'test3', 'test4', 'test5'],
                'pos': ['noun', 'noun', 'noun', 'noun', 'noun'],
                'def1': ['test definition', 'test def 2', 'test def 3', 'test def 4', 'test def 5'],
                'def2': ['test definition 2', 'test def 2', 'test def 3', 'test def 4', 'test def 5'],
                'relation': ['exact', 'none', 'exact', 'related', 'broader']}
        df = pd.DataFrame(data=data)
        labels = df['relation']

        model = trainer.train(df, labels, grid_search)

        assert model
        assert model.best_estimator_


class Test_Transformer:
    @pytest.fixture
    def spacy_processed(self):
        spacy = SpacyProcessor(lang=SupportedLanguages.English)
        expected_columns = ['processed_1',
                            'processed_2',
                            'word_processed',
                            'lemmatized_1',
                            'stopwords_removed_1',
                            'lemmatized_2',
                            'stopwords_removed_2']

        return spacy.transform(df)

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


class Test_Spacy:
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
        spacy = SpacyProcessor(SupportedLanguages.German)

        with pytest.raises(UnsupportedSpacyModelError):
            transformed = spacy.transform(df)

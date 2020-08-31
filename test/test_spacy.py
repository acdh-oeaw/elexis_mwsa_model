from mwsa_model.transformers.pipeline import SpacyProcessor
from mwsa_model.service.util import SupportedLanguages
from mwsa_model.transformers.pipeline import UnsupportedSpacyModelError

import unittest
import pandas as pd
import pytest

data = {'word': ['test'], 'pos': ['noun'], 'def1': ['test definition'], 'def2': ['test definition 2']}
df = pd.DataFrame(data=data)
data_with_semicolon = {'word': ['test'], 'pos': ['noun'], 'def1': ['test ; definition'],
                       'def2': ['test ; definition 2']}
df_with_semicolon = pd.DataFrame(data=data_with_semicolon)


class TestSpacy(unittest.TestCase):
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

        assert isinstance(transformed, pd.DataFrame)
        assert df.index.size == transformed.index.size

        for col in expected_columns:
            assert col in transformed

    # def test_unsupported_spacy_model(self):
    #     spacy = SpacyProcessor(SupportedLanguages.Basque)
    #
    #     with pytest.raises(UnsupportedSpacyModelError):
    #         spacy.transform(df)

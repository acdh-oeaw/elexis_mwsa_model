import pprint
import sys
import unittest
import pytest
import pandas as pd

from pandas import DataFrame
sys.path.append('../mwsa_model')
sys.path.append('../data/train')
from mwsa_model.service.data_loader import DataLoader
pprint.pprint(sys.path)
sys.path.append('/Users/seungbinyim/Development/repos/elexis/mwsa_model')


class DataLoaderTest(unittest.TestCase):
    def test_load_data(self):
        data_loader = DataLoader()
        file_name = 'english_nuig.tsv'
        directory = 'data/train'

        data = data_loader.load(directory, file_name)

        self.assertIsInstance(data, DataFrame)
        for col in ['word', 'pos', 'def1', 'def2', 'relation']:
            self.assertIn(col, data.columns)

    def test_feature_label_split(self):
        data_loader = DataLoader()
        data = {'word': ['test'], 'pos': ['noun'], 'def1': ['test definition'], 'def2': ['test definition 2'], 'relation':['exact']}
        df = pd.DataFrame(data=data)

        features, labels = data_loader.split_feature_label(df)

        self.assertEqual(labels.name, 'relation')
        self.assertNotIn('relation', features.columns)
        for col in ['word', 'pos', 'def1', 'def2']:
            self.assertIn(col, features.columns)

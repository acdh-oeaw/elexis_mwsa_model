import logging

import pandas as pd


class DataLoader(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load(self, file_path, file_name, testdata=False):
        file = file_path + '/' + file_name

        loaded_data = None
        try:
            loaded_data = pd.read_csv(file, sep='\t', header=None)

            if testdata:
                loaded_data.columns = ['word', 'pos', 'def1', 'def2']
            else:
                loaded_data.columns = ['word', 'pos', 'def1', 'def2', 'relation']

        except FileNotFoundError:
            self.logger.warning('file '+ str(file) + 'not found')

        return loaded_data

    def split_feature_label(self, data):
        labels = data['relation']
        data = data.drop('relation', axis=1)

        return data, labels

        
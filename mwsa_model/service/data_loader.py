import logging

import dvc.api
import pandas as pd

pd.options.mode.chained_assignment = None

class DataLoader(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load(self, file_path, file_name, testdata=False, version=None):
        file = file_path + '/' + file_name
        loaded_data = None
        self.logger.info('file name: ' + file)
        self.logger.info('version:' + version)
        with dvc.api.open(
            file,
            repo='https://'+os.environ['DEPLOY_USERNAME']+':'+os.environ['DEPLOY_TOKEN']+'@'+gitlab.com/acdh-oeaw/elexis/mwsa_data_registry.git',
            mode='r',
            rev=version) as fd:

            try:
                loaded_data = pd.read_csv(fd, sep='\t', header=None)

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

        
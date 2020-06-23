import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from mwsa.service.util import SupportedLanguages
from mwsa.transformers.pipeline import SpacyProcessor, FirstWordSameProcessor, FeatureSelector, SimilarityProcessor, \
    SemicolonCountTransformer


def load_data(file_path, is_testdata=False):
    loaded_data = pd.read_csv(file_path, sep='\t', header=None)
    add_column_names(loaded_data, is_testdata=False)

    return loaded_data


def add_column_names(df, is_testdata=False):
    column_names = ['word', 'pos', 'def1', 'def2', 'relation']
    if is_testdata is True:
        column_names.remove('relation')

    df.columns = column_names


combined_set = {}
folder = '../data/train'
filename = 'english_nuig'

for filename in os.listdir(folder):
    if filename.endswith(".tsv"):
        combined_set[filename.split('.')[0]] = load_data(folder + '/' + filename, is_testdata=False)

english = combined_set['english_nuig'][1:10]
labels = english['relation']
params = {
    'preprocess__lang':[SupportedLanguages.English],
    'random_forest__bootstrap': [True],
    'random_forest__class_weight': ['balanced', 'balanced_subsample'],
    'random_forest__max_depth': [30],
    'random_forest__max_features': ['auto'],
    'random_forest__min_samples_leaf': [3, 5],
    'random_forest__min_samples_split': [2],
    'random_forest__n_estimators': [300],
    'random_forest__n_jobs': [5]
}
spacy_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor()),
                                 ('first_word_same', FirstWordSameProcessor()),
                                 ('similarity', SimilarityProcessor()),
                                 ('semicolon_diff', SemicolonCountTransformer()),
                                 ('feature_selector', FeatureSelector()),
                                 ('random_forest', RandomForestClassifier())])
scores = ['precision', 'recall', 'f1']

grid_search = GridSearchCV(spacy_pipeline, param_grid=params,
                           scoring='%s_weighted' % 'f1', cv=5,
                           n_jobs=-1, verbose=1)


grid_search.fit(english, labels)
#model = spacy_pipeline.fit(english, labels)
filename = 'english_pipeline.pickle'
with open(filename, 'wb') as file:
    pickle.dump(grid_search, file)


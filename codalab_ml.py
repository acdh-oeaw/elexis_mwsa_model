import os
import pandas as pd
import spacy
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from load_data import split_data
from load_data import train_and_test_classifiers
from tqdm import tqdm_notebook as tqdm

folder = 'C:\\Users\\syim\\Documents\\ELEXIS\\codalab\\public_dat\\train'


def add_column_names(df):
    column_names = ['word', 'pos', 'def1', 'def2', 'relation']
    df.columns = column_names


def load_data(file_path):
    loaded_data = pd.read_csv(file_path, sep='\t', header=None)
    add_column_names(loaded_data)

    return loaded_data


def load_training_data():
    combined_set = {}

    for filename in os.listdir(folder):
        if filename.endswith(".tsv"):
            combined_set[filename.split('.')[0]] = load_data(folder + '/' + filename)

    return combined_set


def spacy_nlp(vec):
    doc_list = []

    for doc in tqdm(vec):
        pr = nlp(doc)
        doc_list.append(pr)

    return doc_list


def sentence2vec(row):
    return row['def1'].similarity(row['def2'])


def first_word_same(row):
    return row['def1'].text.split(' ')[0].lower() == row['def2'].text.split(' ')[0].lower()


def extract_features(data):
    feat = pd.DataFrame()
    feat['similarities'] = data.apply(lambda row: sentence2vec(row), axis=1)

    return feat


def convert_to_nltk_dataset(features, labels):
    converted = []
    for index, row in features.iterrows():
        converted.append((row.to_dict(), labels[index]))
    return converted


def prepare_data(df_features, labels):
    data_holder = {'nltk': {}, 'pd': {}}

    features_nltk = convert_to_nltk_dataset(df_features, labels)
    data_holder['nltk']['trainset'], data_holder['nltk']['testset'] = split_data(features_nltk)
    df_features['relation'] = labels
    data_holder['pd']['trainset'], data_holder['pd']['trainset'] = split_data(df_features)
    return data_holder


def train_models(X_train, y_train, X_train_scaled):
    print('baseline: ', str(get_baseline(y_test)) + '\n')

    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial')\
        .fit(X_train, y_train)
    # lr_scaled = LogisticRegression( solver = 'lbfgs', multi_class = 'multinomial').fit(X_train_scaled, y_train)

    ## Linear kernal won't work very well, experiment with nonlinear ones.
    svm_model = svm.LinearSVC(C=10.0)\
        .fit(X_train, y_train)
    # svm_scaled = svm.LinearSVC().fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(max_depth=5, random_state=0)\
        .fit(X_train, y_train)
    # rf_scaled = RandomForestClassifier(max_depth = 5, random_state=0).fit(X_train_scaled, y_train)

    models = {}
    models['unscaled'] = [lr, svm_model, rf]
    # models['scaled'] = [lr_scaled, svm_scaled, rf_scaled]

    return models


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    pd.set_option('display.max_colwidth', -1)

    all_data = load_training_data()
    en_data = all_data['english_kd']

    en_data_spacy = en_data.apply(lambda x: spacy_nlp(x) if x.name in ['def1', 'def2'] else x)

    features = extract_features(en_data_spacy)

    all_train_and_testset = prepare_data(features, en_data_spacy['relation'])

    train_and_test_classifiers(all_train_and_testset['nltk']['trainset'], all_train_and_testset['nltk']['testset'])

    # features['relation'] = en_data_spacy['relation']
#    features_nltk = [(features, label) for index, (feature, label) in features.iterrows()]


# train_and_test_classifiers(data['english_nuig'])

# for lang in data:
#     if lang=='english':
#         continue
#
#     print(lang)
#
#     separated = analyze_by_class(data[lang])
#     balanced = balance_classes(separated)
#
#     print('balanced dataset: ',len(balanced))
#     train_and_test_classifiers(balanced)

# print('whole dataset: ', len(data[lang]))
# train_and_test_classifiers(data[lang])

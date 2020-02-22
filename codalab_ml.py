# TODO: Our own Word2Vec
# TODO: Feature Selection: correlation analysis, feature elimination
# TODO: Ask papers/results
# TODO: Ask Tanja

import warnings

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')
import os
import spacy
import pandas as pd
from pprint import pprint
from datetime import datetime
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

from load_data import split_data, difference_in_length, first_word_same, jaccard_sim, cosine
from load_data import train_and_test_classifiers

folder = 'data/train'


def add_column_names(df):
    column_names = ['word', 'pos', 'def1', 'def2', 'relation']
    df.columns = column_names


def diff_pos_count(col1, col2):
    pos_def1 = list(set([token.pos for token in col1]))
    pos_def2 = list(set([token.pos for token in col2]))

    return len(pos_def1) - len(pos_def2)


def load_data(file_path):
    loaded_data = pd.read_csv(file_path, sep='\t', header=None)
    add_column_names(loaded_data)

    return loaded_data


def extract_features(data, feats_to_scale):

    feat = pd.DataFrame()

    feat['similarities'] = data.apply(lambda row: row['processed_1'].similarity(row['processed_2']), axis=1)
    feat['first_word_same'] = data.apply(lambda row: first_word_same(row['def1'], row['def2']), axis=1)
    feat['len_diff'] = data.apply(lambda row: difference_in_length(row['def1'], row['def2']), axis=1)
    feat['jaccard'] = data.apply(lambda row: jaccard_sim(row['def1'], row['def2']), axis=1)
    feat['cos'] = data.apply(lambda row: cosine(row['def1'], row['def2']), axis=1)
    feat['pos_diff'] = data.apply(lambda row: diff_pos_count(row['processed_1'], row['processed_2']), axis=1)

    for c_name in feats_to_scale:
        feat[c_name] = preprocessing.scale(feat[c_name])

    return feat


def convert_to_nltk_dataset(feats, labels):
    converted = []
    for index, row in feats.iterrows():
        converted.append((row.to_dict(), labels[index]))
    return converted


def prepare_data(df_features, labels):
    data_holder = {'nltk': {}, 'pd': {}}

    features_nltk = convert_to_nltk_dataset(df_features, labels)
    data_holder['nltk']['trainset'], data_holder['nltk']['testset'] = split_data(features_nltk)
    data_holder['pd']['x_trainset'], data_holder['pd']['x_testset'] = split_data(df_features)
    data_holder['pd']['y_trainset'], data_holder['pd']['y_testset'] = split_data(labels)

    return data_holder


def run_cv_with_dataset(model, trainset, y_train):
    scores = cross_val_score(model, trainset, y_train, cv=5)
    report_file.write('Cross validation scores for model' + model.__class__.__name__ + '\n')
    report_file.write("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2) + '\n')


def cross_val_models(models, x_train, y_train):
    for estimator in models:
        run_cv_with_dataset(estimator, x_train, y_train)

    # for estimator in models['scaled']:
    #     run_cv_with_dataset(estimator, all_training_set['scaled'], y_train)


def get_baseline_df(y_test):
    tp = 0
    for index in y_test.index:
        if y_test[index] == 'none':
            tp += 1

    return float(tp / len(y_test))


def train_models_sklearn(x_train, y_train):
    lr = {'estimator': LogisticRegression(),
          'parameters': {
              'penalty': ['l2', 'none', 'elasticnet'],
              # 'dual': [False],
              'C': [1.0, 2.0, 3.0],
              'fit_intercept': [True, False],
              'class_weight': [dict, 'balanced', None],
              # #'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
              'solver': ['newton-cg', 'sag', 'lbfgs', 'saga'],
              'max_iter': [100, 200, 300, 400],
              'multi_class': ['auto', 'ovr', 'multinomial'],
              'n_jobs': [-1]
          }
          }
    svm_model = {
        'estimator': SVC(),
        'parameters': {
            'C': [3, 5, 10],
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
        }
    }
    rf = {
        'estimator': RandomForestClassifier(),
        'parameters': {
            'bootstrap': [True],
            'max_depth': [2, 5, 10, 15],
            'max_features': [2, 3, 0.5, 0.2, 'auto', 'sqrt', 'log2', None],
            'min_samples_leaf': [3, 4, 5],
            'min_samples_split': [2, 5, 8, 10, 15],
            'n_estimators': [100, 200, 500]
        }
    }
    dt = {'estimator': DecisionTreeClassifier(), 'parameters': {}}

    models = {'unscaled': [lr]}

    tuned_models = tune_hyperparams(models, x_train, y_train)

    return tuned_models


# TODO Restructure this function
def tune_hyperparams(estimators, x_train, y_train):
    result = []
    for estimator in estimators['unscaled']:
        params = estimator['parameters']

        scores = ['precision', 'recall', 'f1']

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            grid_search = GridSearchCV(estimator=estimator['estimator'], param_grid=params,
                                       scoring='%s_weighted' % score, cv=5,
                                       n_jobs=-1, verbose=1)

            print("Performing grid search...")
            print("parameters:")
            pprint(params)
            grid_search.fit(x_train, y_train)
            print()

            # means = grid_search.cv_results_['mean_test_score']
            # stds = grid_search.cv_results_['std_test_score']
            report_file.write(score + '\n')
            #  for mean, std, parameters in zip(means, stds, grid_search.cv_results_['params']):
            #      report_file.write("%0.3f (+/-%0.03f) for %r"
            #                        % (mean, std * 2, parameters) + '\n')

            report_file.write("Best score: %0.3f" % grid_search.best_score_ + '\n')
            report_file.write("Best parameters set:\n")
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(params.keys()):
                report_file.write("\t%s: %r" % (param_name, best_parameters[param_name]) + '\n')

            result.append(grid_search.best_estimator_)

    return result


def is_not_none(df):
    return df['relation'] != 'none'


def is_none(df):
    return df['relation'] == 'none'


def compare_on_testset(models, testset_x, testset_y):
    report_file.write('Model Evaluation on Testset: \n')
    report_file.write('\t' + 'BASELINE: ' + str(get_baseline_df(testset_y)) + '\n')

    # TODO Restructure this part
    for estimator in models['unscaled']:
        report_file.write('\t' + estimator.__class__.__name__)
        report_file.write(str(estimator))

        predicted = estimator.predict(testset_x)
        report_file.write(str(classification_report(testset_y, predicted)) + '\n')
        report_file.write(str(confusion_matrix(testset_y, predicted)) + '\n')

        report_file.write('\t\t' + "F-Score:" + str(f1_score(testset_y, predicted, average='weighted')) + '\n')

        score = estimator.score(testset_x, testset_y)
        report_file.write('\t\t' + "Accuracy: %0.4f (+/- %0.4f)" % (score.mean(), score.std() * 2) + '\n')


def open_file():
    now = datetime.now()
    return open("reports/" + now.strftime("%Y%m%d%H%M%S") + ".txt", "a")


def configure():
    pd.set_option('display.max_colwidth', -1)


def load_training_data():
    combined_set = {}

    for filename in os.listdir(folder):
        if filename.endswith(".tsv"):
            combined_set[filename.split('.')[0]] = load_data(folder + '/' + filename)

    return combined_set


# TODO Oversampling, Ensemble Learning Techniques
def balance_dataset(imbalanced_set):
    none = imbalanced_set[is_none(imbalanced_set) == True]
    second_biggest = imbalanced_set.groupby('relation').count().word.sort_values(ascending=False)[1]
    result = imbalanced_set.drop(none.index[second_biggest:])

    return result.sample(frac=1, random_state=7)


def load_and_preprocess():
    def lemmatizer(doc):
        doc = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']
        doc = u' '.join(doc)
        return nlp.make_doc(doc)


    def remove_stopwords(doc):
        # TODO: ADD 'etc' to stopwords list
        doc = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        return doc


    all_data = load_training_data()
    en_data = all_data['english_kd']
    balanced = balance_dataset(en_data)

    balanced['processed_1'] = balanced['def1'].map(nlp)
    balanced['processed_2'] = balanced['def2'].map(nlp)

    balanced['lemmatized_1'] = balanced['processed_1'].map(lemmatizer)
    balanced['stopwords_removed'] = balanced['lemmatized_1'].map(remove_stopwords)

    balanced['lemmatized_2'] = balanced['processed_2'].map(lemmatizer)
    balanced['stopwords_removed'] = balanced['lemmatized_2'].map(remove_stopwords)

    return balanced


def train(data, with_testset=False):
    train_and_test_classifiers(data['nltk']['trainset'], data['nltk']['testset'])
    trained_models = train_models_sklearn(data['pd']['x_trainset'],
                                          data['pd']['y_trainset'])
    cross_val_models(trained_models, data['pd']['x_trainset'],
                     data['pd']['y_trainset'])

    if with_testset:
        compare_on_testset(trained_models, data['pd']['x_testset'],
                           data['pd']['y_testset'])


def count_relation_and_sort():
    return str(balanced_en_data.groupby('relation').count().word.sort_values(ascending=False)) + "\n"


if __name__ == '__main__':
    configure()
    nlp = spacy.load('en_core_web_lg')
    report_file = open_file()

    balanced_en_data = load_and_preprocess()

    report_file.write(count_relation_and_sort())

    features = extract_features(balanced_en_data, ['similarities', 'len_diff', 'pos_diff'])

    all_train_and_testset = prepare_data(features, balanced_en_data['relation'])

    train(all_train_and_testset)

    report_file.close()

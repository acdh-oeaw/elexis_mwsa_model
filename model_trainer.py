from datetime import datetime
from pprint import pprint
from random import random

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def open_file():
    now = datetime.now()
    return open("reports/" + now.strftime("%Y%m%d%H%M%S") + ".txt", "a")

class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array(['none' for x in X.index])

class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X):
        relations = ['related','narrower','broader','exact','none']
        return np.array([random.choice(relations) for x in X.index])


class ModelTrainer:

    def __init__(self, testset_ratio):
        self._report_file = open_file()
        self._testset_ratio = testset_ratio

        self._x_trainset, self._x_testset = None, None
        self._y_trainset, self._y_testset = None, None


    def __split_data(self, featuresets, testset_ratio):
        f = int(len(featuresets) * testset_ratio)
        return featuresets[f:], featuresets[:f]

    def __train_models_sklearn(self):
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
        # rf = {
        #     #     'estimator': RandomForestClassifier(),
        #     #     'parameters': {
        #     #         'bootstrap': [True],
        #     #         'max_depth': [2, 5, 10, 15],
        #     #         'max_features': [2, 3, 0.5, 0.2, 'auto', 'sqrt', 'log2', None],
        #     #         'min_samples_leaf': [3, 4, 5],
        #     #         'min_samples_split': [2, 5, 8, 10, 15],
        #     #         'n_estimators': [100, 200, 500]
        #     #     }
        #     # }
        rf = {
            'estimator': RandomForestClassifier(),
            'parameters': {
                'bootstrap': [True],
                'max_depth': [30, 50],
                'max_features': [None],
                'min_samples_leaf': [3, 5],
                'min_samples_split': [2, 5, 8],
                'n_estimators': [500, 600]
            }
        }
        dt = {'estimator': DecisionTreeClassifier(), 'parameters': {}}

        models = {'unscaled': [rf]}

        tuned_models = self.__tune_hyperparams(models)

        return tuned_models

    def __get_baseline_df(self):
        tp = 0

        for index in self._y_testset.index:
            if self._y_testset[index] == 'none':
                tp += 1

        return float(tp / len(self._y_testset))


    def __run_cv_with_dataset(self, model, trainset, y_train):
        scores = cross_val_score(model, trainset, y_train, cv=5)
        self._report_file.write('Cross validation scores for model' + model.__class__.__name__ + '\n')
        self._report_file.write("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2) + '\n')

    def predict_on_testset(self, models):
        self._report_file.write('Model Evaluation on Testset: \n')
        self._report_file.write('\t' + 'BASELINE: ' + str(self.__get_baseline_df()) + '\n')
        models.append(BaseClassifier())
        # TODO Restructure this part
        for estimator in models:
            self._report_file.write('\t' + estimator.__class__.__name__)
            self._report_file.write(str(estimator))

            predicted = estimator.predict(self._x_testset)
            self._report_file.write(str(classification_report(self._y_testset, predicted)) + '\n')
            self._report_file.write(str(confusion_matrix(self._y_testset, predicted)) + '\n')

            self._report_file.write(
                '\t\t' + "F-Score:" + str(f1_score(self._y_testset, predicted, average='weighted')) + '\n')

            score = estimator.score(self._x_testset, self._y_testset)
            self._report_file.write('\t\t' + "Accuracy: %0.4f (+/- %0.4f)" % (score.mean(), score.std() * 2) + '\n')

    def cross_val_models(self, models, x_train, y_train):
        for estimator in models:
            self.__run_cv_with_dataset(estimator, x_train, y_train)

        # TODO Restructure this function

    def __tune_hyperparams(self, estimators):
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
                grid_search.fit(self._x_trainset, self._y_trainset)
                print()

                # means = grid_search.cv_results_['mean_test_score']
                # stds = grid_search.cv_results_['std_test_score']

                self._report_file.write(score + '\n')
                #  for mean, std, parameters in zip(means, stds, grid_search.cv_results_['params']):
                #      report_file.write("%0.3f (+/-%0.03f) for %r"
                #                        % (mean, std * 2, parameters) + '\n')

                self._report_file.write("Best score: %0.3f" % grid_search.best_score_ + '\n')
                self._report_file.write("Best parameters set:\n")
                best_parameters = grid_search.best_estimator_.get_params()
                for param_name in sorted(params.keys()):
                    self._report_file.write("\t%s: %r" % (param_name, best_parameters[param_name]) + '\n')

                result.append(grid_search.best_estimator_)

        return result

    def train(self, data, labels, with_testset=False):
        self._x_trainset, self._x_testset = self.__split_data(data, self._testset_ratio)
        self._y_trainset, self._y_testset = self.__split_data(labels, self._testset_ratio)

        trained_models = self.__train_models_sklearn()

        self.cross_val_models(trained_models, self._x_trainset,
                              self._y_trainset)
        if with_testset:
            self.predict_on_testset(trained_models)

        self._report_file.close()
        return trained_models

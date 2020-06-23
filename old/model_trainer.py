import logging
import pickle
import uuid
from datetime import datetime
from pprint import pprint
from random import random

import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_validate


class BaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self

    def predict(self, x):
        return np.array(['none' for x in x.index])


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self

    def predict(self, x):
        relations = ['related', 'narrower', 'broader', 'exact', 'none']
        return np.array([random.choice(relations) for x in x.index])


class ModelTrainer:

    def __init__(self, testset_ratio, logger_name):
        self._testset_ratio = testset_ratio

        self._x_trainset, self._x_testset = None, None
        self._y_trainset, self._y_testset = None, None

        self.best_f1_model = None

        self._logger = logging.getLogger(logger_name)
        self._models = []

    @staticmethod
    def split_data(featuresets, testset_ratio):
        f = int(len(featuresets) * testset_ratio)
        return featuresets[f:], featuresets[:f]

    def __train_models_sklearn(self):
        tuned_models = self.__tune_hyperparams(self._models)

        return tuned_models

    def add_estimators(self, estimators):
        self._models = self._models + estimators

    @staticmethod
    def get_baseline_df(y_testset):
        tp = 0

        for index in y_testset.index:
            if y_testset[index] == 'none':
                tp += 1

        return float(tp / len(y_testset))

    def __run_cv_with_dataset(self, model, trainset, y_train):
        scores = cross_validate(model, trainset, y_train, cv=5, scoring=['f1_weighted', 'balanced_accuracy'])
        self._logger.info('Cross validation scores for model' + model.__class__.__name__ + '\n')
        self._logger.info("F1_WEIGHTED: %0.4f (+/- %0.4f)" % (
            scores['test_f1_weighted'].mean(), scores['test_f1_weighted'].std() * 2) + '\n')
        self._logger.info("BALANCED_ACCURACY: %0.4f (+/- %0.4f)" % (
            scores['test_balanced_accuracy'].mean(), scores['test_balanced_accuracy'].std() * 2) + '\n')
        return scores

    def predict_on_testset(self, models):
        self._logger.info('Model Evaluation on Testset: \n')
        self._logger.info('\t' + 'BASELINE: ' + str(self.get_baseline_df(self._y_testset)) + '\n')
        models.append(BaseClassifier())

        for estimator in models:
            self._logger.info('\t' + estimator.__class__.__name__)
            self._logger.info(str(estimator))

            predicted = estimator.predict(self._x_testset)
            self._logger.info(str(classification_report(self._y_testset, predicted)) + '\n')
            self._logger.info(str(confusion_matrix(self._y_testset, predicted)) + '\n')

            self._logger.info(
                '\t\t' + "F-Score:" + str(f1_score(self._y_testset, predicted, average='weighted')) + '\n')

            score = estimator.score(self._x_testset, self._y_testset)
            self._logger.info('\t\t' + "Accuracy: %0.4f (+/- %0.4f)" % (score.mean(), score.std() * 2) + '\n')

    def cross_val_models(self, models, x_train, y_train):
        best_f1_estimator = None
        best_f1_score = None

        for estimator in models:
            scores = self.__run_cv_with_dataset(estimator, x_train, y_train)
            if best_f1_score is None or scores['test_f1_weighted'].mean() > best_f1_score:
                best_f1_score = scores['test_f1_weighted'].mean()
                best_f1_estimator = estimator

        self.best_f1_model = best_f1_estimator

    def __tune_hyperparams(self, estimators):
        result = []
        best_f1 = 0.0
        for estimator in estimators:
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

                best_estimator = grid_search.best_estimator_
                result.append(best_estimator)

                if score is 'f1' and grid_search.best_score_ > best_f1:
                    self.best_f1_model = best_estimator

                self._logger.info(
                    '\n'.join([score, "Best score: %0.3f" % grid_search.best_score_, "Best parameters set: "]))
                for param_name in sorted(params.keys()):
                    self._logger.info("\t%s: %r" % (param_name, best_estimator.get_params()[param_name]) + '\n')

        voting_clf = VotingClassifier(estimators=list(
            map(lambda classifier: ("".join([classifier.__class__.__name__, str(uuid.uuid4())]), classifier), result)),
            voting='hard')
        voting_clf.fit(self._x_trainset, self._y_trainset)

        result.append(voting_clf)
        return result

    def cross_validate(self, trained_models, x, y):
        self.cross_val_models(trained_models, x, y)

        return self

    def save_best_model(self):

        filename = 'models/'+self._logger.name + self.best_f1_model.__class__.__name__ + datetime.now().strftime(
            "%Y%m%d-%H%M") + '.pickle'

        with open(filename, 'wb') as file:
            pickle.dump(self.best_f1_model, file)

    def train(self, data, labels, with_testset=False):
        self._x_trainset, self._x_testset = self.split_data(data, self._testset_ratio)
        self._y_trainset, self._y_testset = self.split_data(labels, self._testset_ratio)

        trained_models = self.__train_models_sklearn()

        self.cross_val_models(trained_models, self._x_trainset,
                              self._y_trainset)

        self.save_best_model()

        if with_testset:
            self.predict_on_testset(trained_models)

        return trained_models

import csv
import os
import nltk
from nltk.corpus import wordnet as wn
import random

from nltk.metrics.scores import (precision, recall)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

# import Levenshtein

folder = 'C:\\Users\\syim\\Documents\\ELEXIS\\codalab\\public_dat\\train'


def load_training_data(folder):
    loaded_data = {}
    for filename in os.listdir(folder):
        if filename.endswith(".tsv"):
            with open(folder + '/' + filename, encoding='utf-8') as tsvfile:

                reader = csv.reader(tsvfile, delimiter='\t')
                lang = filename.split('.')[0]
                loaded_data[lang] = []

                for row in reader:
                    data_row = {'lemma': row[0],
                                'pos': row[1],
                                'def1': row[2],
                                'def2': row[3]}
                    loaded_data[lang].append((data_row, row[4]))

    # for lang in loaded_data:
    #    print(lang, len(loaded_data[lang]))

    return loaded_data


def analyze_by_class(dataset):
    separated = dict()
    for vector in dataset:
        class_value = vector[1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector[0])

    for s in separated:
        print(s, len(separated[s]))

    return separated


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_cosine_sim(*strs):
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors)


def get_vectors(*strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def first_word_same(row):
    return row['def1'].split(' ')[0].lower() == row['def2'].split(' ')[0].lower()


def difference_in_length(row):
    return abs(len(row['def1'].split(' ')) - len(row['def2'].split(' ')[0]))


def jaccard_sim(row):
    return get_jaccard_sim(row['def1'], row['def2'])


def cosine(row):
    return get_cosine_sim(row['def1'], row['def2'])[0, 1]


def find_features(row):
    features = {}
    features['first_word_same'] = (first_word_same(row))
    features['len difference'] = difference_in_length(row)

    features['jaccard'] = jaccard_sim(row)

    features['cosine'] = cosine(row)

    # features['levenshtein'] = Levenshtein.distance(row['def1'], row['def2'])

    # if features['cosine']>0.9:
    #    print(row['def1'], row['def2'], features['cosine'])

    '''
    wordmatch = 0
    for word in row['def1'].split(' ')[0].lower():
        if word in row['def2'].lower():
            wordmatch+=1

    features['wordmatch'] = wordmatch
    '''

    features['synsets'] = len(wn.synsets(row['lemma']))  # for specific pos e.g. wn.synsets('dog', pos=wn.VERB)

    # if features['synsets'] == 0:
    #    print('no synset for ',row['lemma']) TODO MULTILIGNUAL WN

    return features


def prepare_data(dataset):
    featuresets = [(find_features(row), label) for (row, label) in dataset]
    return split_data(featuresets)


def split_data(featuresets):
    f = int(len(featuresets) / 5)
    return featuresets[f:], featuresets[:f]


def get_baseline(test_set):
    TP = 0
    for el in test_set:
        if el[1] == 'none':
            TP += 1

    return float(TP / len(test_set))


def balance_classes(separated):
    second_biggest_class = 0
    for el in separated:
        if el != 'none':
            if len(separated[el]) > second_biggest_class:
                second_biggest_class = len(separated[el])

    separated['none'] = separated['none'][:second_biggest_class]

    balanced = []
    for el in separated:
        for row in separated[el]:
            balanced.append((row, el))

    random.shuffle(balanced)

    return balanced


def train_and_test_classifiers(train_set, test_set):
    naive_bayes = nltk.NaiveBayesClassifier.train(train_set)

    print('naive bayes: ', nltk.classify.accuracy(naive_bayes, test_set))
    # naive_bayes.show_most_informative_features(5)

    decision_tree = nltk.DecisionTreeClassifier.train(train_set)
    print('desicion tree: ', nltk.classify.accuracy(decision_tree, test_set))

    print('baseline: ', str(get_baseline(test_set)) + '\n')

    # quite slow
    # max_ent = nltk.MaxentClassifier.train(train_set, trace=-1)
    # print(nltk.classify.accuracy(max_ent, test_set))
    # max_ent.show_most_informative_features(5)
    # print('\n')


if __name__ == '__main__':

    data = load_training_data(folder)

    for lang in data:
        if lang == 'english':
            continue

        print(lang)

        separated = analyze_by_class(data[lang])
        balanced = balance_classes(separated)

        print('balanced dataset: ', len(balanced))
        train_set, test_set = prepare_data(balanced)
        train_and_test_classifiers(train_set, test_set)

        # print('whole dataset: ', len(data[lang]))
        # train_and_test_classifiers(data[lang])

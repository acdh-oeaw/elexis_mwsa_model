import pickle

import pandas as pd

with open('english_pipeline.pickle', 'rb') as pickle_file:
    clf = pickle.load(pickle_file)
    data = {'word':['test'], 'pos':['noun'], 'def1':['test definition'], 'def2':['test definition 2']}
    testdata = pd.DataFrame(data=data)
    predicted = clf.predict(testdata)
    print(predicted)
    predicted_series= pd.Series(predicted)
    testdata['relation'] = predicted_series
    german_predicted = testdata[['word','pos','def1','def2','relation']]
    german_predicted.to_csv('english_nuig_20200401.csv',sep='\t', index = False)


import pickle
import time
import pandas as pd

df = pd.DataFrame(
    data={'word': ['test'], 'pos': ['noun'], 'def1': ['all my dreams are mad'],
          'def2': ['TEST 2']})

file = 'en.pkl'
with open(file, 'rb') as model_file:
    model = pickle.load(model_file)

    print(time.perf_counter())
    predicted = model.predict(df)
    print(time.perf_counter())
    print(predicted)
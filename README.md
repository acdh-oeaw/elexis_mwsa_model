MWSA Model is trained using Scikit Learn Pipeline.
For each language we build distinct pipelines consisting of following components.

1. Preprocessing.
For languages supported by Spacy, spacy components are used for typical nlp tasks such as tokenization, 
dependency parsing, stopwords removal, etc. For other languages, Stanford NLP components are used.

Spacy Models used are listed here.
https://spacy.io/usage/models

2. Feature Extraction

3. Classifier

4. Grid Search

5. Evaluation

DVC(https://dvc.org/doc/start) is used for defining and version controlling the pipeline.

dvc run -n train -f train.dvc -d mwsa/preprocess.py -d features.pickle -d labels.pickle -o mwsa/output/models/en.pkl python -m mwsa.preprocess features.pickle labels.pickle
dvc run -n evaluate -f evaluate.dvc -d mwsa/evaluate.py -d mwsa/output/models/en.pkl -d data/test/english_nuig.tsv -d data/reference_data/english_nuig.tsv -o mwsa/output/metrics/metrics_en.txt -o mwsa/output/predictions/en_predictions.txt python -m mwsa.evaluate en.pkl english_nuig.tsv metrics_en.txt en_predictions.txt

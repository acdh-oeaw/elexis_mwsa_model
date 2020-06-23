# ACDH Mololingual Word Sense Alignment Model
MWSA Model is trained using Scikit Learn Pipeline.
For each language we build distinct pipelines consisting of following components.

## Pipeline Description
### Preprocessing.
For languages supported by Spacy, spacy components are used for typical nlp tasks such as tokenization, 
dependency parsing, stopwords removal, etc. For other languages, Stanford NLP components are used.

Spacy Models used are listed here.
https://spacy.io/usage/models

### Feature Extraction

### Classifier

### Grid Search

### Evaluation

## Managing Pipeline and Reproducibility

DVC(https://dvc.org/doc/start) is used for defining and version controlling the pipeline.

By running  <code>dvc repro</code> all stages will be run and outputs will be generated.

DVC stages are managed in dvc.yaml and were added by running dvc commands as below.

<code>dvc run -n train -f train.dvc -d mwsa/preprocess.py -d features.pickle -d labels.pickle -o mwsa/output/models/en.pkl python -m mwsa.preprocess features.pickle labels.pickle</code>

<code>dvc run -n evaluate -f evaluate.dvc -d mwsa/evaluate.py -d mwsa/output/models/en.pkl -d data/test/english_nuig.tsv -d data/reference_data/english_nuig.tsv -o mwsa/output/metrics/metrics_en.txt -o mwsa/output/predictions/en_predictions.txt python -m mwsa.evaluate en.pkl english_nuig.tsv metrics_en.txt en_predictions.txt</code>


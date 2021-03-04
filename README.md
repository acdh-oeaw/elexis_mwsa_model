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

| Feature                   | English  | GERMAN |
| ------------------------- | ------   | ------ |
| POS_COUNT_DIFF            |     Y    |   Y    |
| FIRST_WORD_SAME           |     Y    |   Y    |
| SIMILARITY                |     Y    |   Y    |
| LEMMA_MATCH               |     Y    |   Y    |
| LEN_DIFF                  |     Y    |   Y    |
| MAX_DEPTH_TREE_DIFF       |     Y    |        |
| SIMILARITY_DIFF_TO_TARGET |     Y    |   Y    |
| ONE_HOT_POS               |          |   Y    |   
| TFIDF_COS                 |          |   Y    |


### Classifier

### Grid Search

### Evaluation

## Managing Pipeline and Reproducibility.

DVC(https://dvc.org/doc/start) is used for defining and version controlling the pipeline.

By running  <code>dvc repro</code> all stages will be run and outputs will be generated

You can run language specific pipeline by setting the $STAGE environment variable to the stage you want to run and executing following code

<code>dvc repro -pf $STAGE</code>

The option p runs the whole pipeline including the specified stage, f option forcefully runs the pipeline even if there are no changes made.

DVC stages are managed in dvc.yaml and were added by running dvc commands as below.

* <code>dvc run -n train -f train.dvc -d mwsa/preprocess.py -d features.pickle -d labels.pickle -o mwsa/output/models/en.pkl python -m mwsa.preprocess features.pickle labels.pickle</code>

* <code>dvc run -n evaluate -f evaluate.dvc -d mwsa/evaluate.py -d mwsa/output/models/en.pkl -d data/test/english_nuig.tsv -d data/reference_data/english_nuig.tsv -o mwsa/output/metrics/metrics_en.txt -o mwsa/output/predictions/en_predictions.txt python -m mwsa.evaluate en.pkl english_nuig.tsv metrics_en.txt en_predictions.txt</code>

# Evaluation Metrics
Evaluation metrics can be viewed with
zu
* <code>dvc metrics show</code>

If metrics files were changed, the difference can be seen with

* <code>dvc metrics diff</code>

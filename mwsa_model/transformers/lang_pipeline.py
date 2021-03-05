from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from mwsa_model import features
from mwsa_model.service.util import SupportedLanguages
from mwsa_model.transformers.pipeline import SpacyProcessor, DiffPosCountTransformer, FirstWordSameProcessor, \
    SimilarityProcessor, MatchingLemmaTransformer, DifferenceInLengthTransformer, MaxDependencyTreeDepthTransformer, \
    ToTargetSimilarityDiffTransformer, FeatureSelector

english_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.English, with_wordnet=False)),
                                   (features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                   #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                   (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                   (features.SIMILARITY, SimilarityProcessor()),
                                   (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                   #('pos_count', CountEachPosTransformer()),
                                   #(features.SYNSET_COUNT_DIFF, AvgSynsetCountTransformer()),
                                   (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                   (features.MAX_DEPTH_TREE_DIFF, MaxDependencyTreeDepthTransformer()),
                                   #(features.TARGET_WORD_SYNSET_COUNT, TargetWordSynsetCountTransformer()),
                                   (features.SIMILARITY_DIFF_TO_TARGET, ToTargetSimilarityDiffTransformer()),
                                   #(features.SEMICOLON_DIFF, SemicolonCountTransformer()),
                                   ('feature_selector', FeatureSelector()),
                                   ('random_forest', RandomForestClassifier())])
german_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.German)),
                                  (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                  (features.SIMILARITY, SimilarityProcessor()),
                                  #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                  #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                  (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                  (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                  #(features.TFIDF_COS, TfidfTransformer()),
                                  #(features.JACCARD, JaccardTransformer()),
                                  #(features.COSINE, CosineTransformer()),
                                  ('feature_selector', FeatureSelector()),
                                  ('random_forest', RandomForestClassifier())])
russian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Russian)),
                                   (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                   (features.SIMILARITY, SimilarityProcessor()),
                                   #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                   # (features.ONE_HOT_POS, OneHotPosTransformer()),
                                   (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                   (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                   # (features.TFIDF_COS, TfidfTransformer()),
                                   # (features.JACCARD, JaccardTransformer()),
                                   # (features.COSINE, CosineTransformer()),
                                   ('feature_selector', FeatureSelector()),
                                   ('random_forest', RandomForestClassifier())])

italian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Italian)),
                                   (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                   (features.SIMILARITY, SimilarityProcessor()),
                                   #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                   #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                   (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                   (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                   #(features.TFIDF_COS, TfidfTransformer()),
                                   #(features.JACCARD, JaccardTransformer()),
                                   #(features.COSINE, CosineTransformer()),
                                   ('feature_selector', FeatureSelector()),
                                   ('random_forest', RandomForestClassifier())])

portuguese_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Portuguese)),
                                      (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                      (features.SIMILARITY, SimilarityProcessor()),
                                      #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                      #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                      (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                      (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                      #(features.TFIDF_COS, TfidfTransformer()),
                                      #(features.JACCARD, JaccardTransformer()),
                                      #(features.COSINE, CosineTransformer()),
                                      ('feature_selector', FeatureSelector()),
                                      ('random_forest', RandomForestClassifier())])

danish_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Danish)),
                                  (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                  (features.SIMILARITY, SimilarityProcessor()),
                                  #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                  #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                  (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                  (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                  #(features.TFIDF_COS, TfidfTransformer()),
                                  #(features.JACCARD, JaccardTransformer()),
                                  #(features.COSINE, CosineTransformer()),
                                  ('feature_selector', FeatureSelector()),
                                  ('random_forest', RandomForestClassifier())])

dutch_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Dutch)),
                                 (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                 (features.SIMILARITY, SimilarityProcessor()),
                                 #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                 #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                 (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                 (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                 #(features.TFIDF_COS, TfidfTransformer()),
                                 #(features.JACCARD, JaccardTransformer()),
                                 #(features.COSINE, CosineTransformer()),
                                 ('feature_selector', FeatureSelector()),
                                 ('random_forest', RandomForestClassifier())])

serbian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Serbian)),
                                   (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                   (features.SIMILARITY, SimilarityProcessor()),
                                   #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                   #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                   (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                   (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                   #(features.TFIDF_COS, TfidfTransformer()),
                                   #(features.JACCARD, JaccardTransformer()),
                                   #(features.COSINE, CosineTransformer()),
                                   ('feature_selector', FeatureSelector()),
                                   ('random_forest', RandomForestClassifier())])

bulgarian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Bulgarian)),
                                     (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                     (features.SIMILARITY, SimilarityProcessor()),
                                     #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                     #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                     (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                     (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                     #(features.TFIDF_COS, TfidfTransformer()),
                                     #(features.JACCARD, JaccardTransformer()),
                                     #(features.COSINE, CosineTransformer()),
                                     ('feature_selector', FeatureSelector()),
                                     ('random_forest', RandomForestClassifier())])

slovene_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Slovene)),
                                   (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                   (features.SIMILARITY, SimilarityProcessor()),
                                   #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                   #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                   (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                   (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                   #(features.TFIDF_COS, TfidfTransformer()),
                                   #(features.JACCARD, JaccardTransformer()),
                                   #(features.COSINE, CosineTransformer()),
                                   ('feature_selector', FeatureSelector()),
                                   ('random_forest', RandomForestClassifier())])

estonian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Estonian)),
                                    (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                    (features.SIMILARITY, SimilarityProcessor()),
                                    #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                    #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                    (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                    (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                    #(features.TFIDF_COS, TfidfTransformer()),
                                    #(features.JACCARD, JaccardTransformer()),
                                    #(features.COSINE, CosineTransformer()),
                                    ('feature_selector', FeatureSelector()),
                                    ('random_forest', RandomForestClassifier())])

basque_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Basque)),
                                  (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                  (features.SIMILARITY, SimilarityProcessor()),
                                  #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                  #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                  (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                  (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                  #(features.TFIDF_COS, TfidfTransformer()),
                                  #(features.JACCARD, JaccardTransformer()),
                                  #(features.COSINE, CosineTransformer()),
                                  ('feature_selector', FeatureSelector()),
                                  ('random_forest', RandomForestClassifier())])

irish_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Irish)),
                                 (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                 (features.SIMILARITY, SimilarityProcessor()),
                                 #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                 #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                 (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                 (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                 #(features.TFIDF_COS, TfidfTransformer()),
                                 #(features.JACCARD, JaccardTransformer()),
                                 #(features.COSINE, CosineTransformer()),
                                 ('feature_selector', FeatureSelector()),
                                 ('random_forest', RandomForestClassifier())])

hungarian_pipeline = Pipeline(steps=[('preprocess', SpacyProcessor(lang=SupportedLanguages.Hungarian)),
                                     (features.FIRST_WORD_SAME, FirstWordSameProcessor()),
                                     (features.SIMILARITY, SimilarityProcessor()),
                                     #(features.POS_COUNT_DIFF, DiffPosCountTransformer()),
                                     #(features.ONE_HOT_POS, OneHotPosTransformer()),
                                     (features.LEMMA_MATCH, MatchingLemmaTransformer()),
                                     (features.LEN_DIFF, DifferenceInLengthTransformer()),
                                     #(features.TFIDF_COS, TfidfTransformer()),
                                     #(features.JACCARD, JaccardTransformer()),
                                     #(features.COSINE, CosineTransformer()),
                                     ('feature_selector', FeatureSelector()),
                                     ('random_forest', RandomForestClassifier())])
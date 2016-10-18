from parameters import LanguageParams
from feature_extractors.length_evaluator import SentenceLengthEvaluator, WordLengthEvaluator
from feature_extractors.similarity_evaluator import SimilarityEvaluator
from feature_extractors.tfidf_evaluator import TfIdfEvaluator

LANGUAGE = LanguageParams.POLISH
FEATURES_EXTRACTORS = [TfIdfEvaluator,
                       SimilarityEvaluator,
                       SentenceLengthEvaluator,
                       WordLengthEvaluator
                       ]



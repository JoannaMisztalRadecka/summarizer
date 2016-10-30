# from parameters import LanguageParams
from feature_extractors.length_evaluator import SentenceLengthEvaluator, WordLengthEvaluator
from feature_extractors.similarity_evaluator import SimilarityEvaluator
from feature_extractors.tfidf_evaluator import TfIdfEvaluator

LANGUAGE = 'pl'
FEATURES_EXTRACTORS = [TfIdfEvaluator,
                       SimilarityEvaluator,
                       SentenceLengthEvaluator,
                       WordLengthEvaluator
                       ]
ROUGE_PATH = '/home/asia/anaconda3/lib/python3.5/site-packages/pyrouge/tools/ROUGE-1.5.5/'
PYDIC_DICT_PATH = '/home/asia/anaconda3/lib/python3.5/site-packages/pydic/odm.pydic/'
RESOURCES_PATH = '/home/asia/resources/'
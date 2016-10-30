import os
import settings

from stemmers.sjp_stemmer import SJPStemmer
# TODO: create language factory


class LanguageParams(object):
    POLISH = 'pl'
    ENGLISH = 'en'

    LANGUAGE_PARAMS = {
        'pl':
            {
                'stopwords': 'stopwords_pl.txt',
                'stemmer': SJPStemmer(settings.PYDIC_DICT_PATH),
                'summaries_corpora': 'polish_summaries_corpora/PSC_1.0/data/',
                'sentence_tokenizer': 'tokenizers/punkt/polish.pickle'
            }
    }

    def __init__(self, language):
        self.language = language
        self.stopwords_path = self.__get_path('stopwords')
        self.summaries_corpora = self.__get_path('summaries_corpora')
        self.stemmer = self.LANGUAGE_PARAMS[self.language]['stemmer']
        self.tokenizer_path = self.LANGUAGE_PARAMS[self.language]['sentence_tokenizer']

    def __get_path(self, param):
        return os.path.join(settings.RESOURCES_PATH, self.LANGUAGE_PARAMS[self.language][param])
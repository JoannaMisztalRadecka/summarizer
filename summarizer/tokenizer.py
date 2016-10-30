from abc import ABCMeta, abstractmethod
import re
import nltk
import nltk.data
from nltk.tokenize import RegexpTokenizer
from unidecode import unidecode


class Tokenizer(object):
    __metaclass__ = ABCMeta

    def __init__(self, language_params):
        self.language_params = language_params

    @abstractmethod
    def tokenize(self, text):
        pass


class WordTokenizer(Tokenizer):

    def __init__(self, language_params):
        super(WordTokenizer, self).__init__(language_params)
        self.stemmer = self.language_params.stemmer
        self.stopwords = self.__get_stopwords()
        self.word_tokenizer = RegexpTokenizer(r'\w+')

    def tokenize(self, text):
        tokens = self.__get_tokens(text)
        tokens_filtered = [t.lower() for t in tokens if t not in self.stopwords]
        stemmed = list(set([unidecode(self.stemmer.get_stem(s)) for s in tokens_filtered if s]))
        return stemmed

    def __get_stopwords(self):
        f = open(self.language_params.stopwords_path, encoding='utf-8', mode='r')
        return f.read().split(', ')

    def __get_tokens(self, text):
        return self.word_tokenizer.tokenize(text)


class SentenceTokenizer(Tokenizer):

    def tokenize(self, text):
        sent_tokenizer = nltk.data.load(self.language_params.tokenizer_path)
        sentences = []
        for t in sent_tokenizer.tokenize(unidecode(text)):
            sentences += [s for s in t.split('\n') if len(s) > 1]

        return sentences






from abc import ABCMeta, abstractmethod


class SentencesEvaluator(object):
    __metaclass__ = ABCMeta

    def __init__(self, language_params):
        self.language_params = language_params

    @abstractmethod
    def evaluate(self, sentences):
        pass

    def train(self, training_set):
        pass

    def __repr__(self):
        return str(self.__class__)
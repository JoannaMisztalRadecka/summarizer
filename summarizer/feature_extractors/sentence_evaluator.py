from abc import ABCMeta, abstractmethod


class SentencesEvaluator(object):
    __metaclass__ = ABCMeta

    def __init__(self, language_params, name='Evaluator'):
        self.language_params = language_params
        self.name = name

    @abstractmethod
    def evaluate(self, sentences):
        pass

    def train(self, training_set):
        pass

    def __repr__(self):
        return str(self.name)
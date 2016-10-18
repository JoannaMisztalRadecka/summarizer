import numpy as np
from abc import ABCMeta, abstractmethod

from feature_extractors.sentence_evaluator import SentencesEvaluator
from tokenizer import WordTokenizer


class LengthEvaluator(SentencesEvaluator):
    __class__ = ABCMeta

    def evaluate(self, sentences):
        tokenizer = WordTokenizer(self.language_params)
        tokenized_sents = [tokenizer.tokenize(s) for s in sentences]

        return self._get_lengths(tokenized_sents)

    @abstractmethod
    def _get_lengths(self, tokenized_sents):
        pass


class WordLengthEvaluator(LengthEvaluator):

    def _get_lengths(self, sentences):
        avg_lengths = []
        for i, s in enumerate(sentences):
            lengths = [float(len(w)) for w in s]
            avg_lengths.append((i, np.mean(lengths)))

        return avg_lengths


class SentenceLengthEvaluator(LengthEvaluator):

    def _get_lengths(self, sentences):
        avg_lengths = [len(s) for s in sentences]
        return enumerate(avg_lengths)

import math
from collections import Counter

import numpy as np

from feature_extractors.sentence_evaluator import SentencesEvaluator
from tokenizer import WordTokenizer


class SimilarityEvaluator(SentencesEvaluator):
    def __init__(self, language_params):
        super(SimilarityEvaluator, self).__init__(language_params, 'Similarity Evaluator')

    def train(self, training_set):
        pass

    def evaluate(self, sentences):
        sims = []
        for i, s in enumerate(sentences):
            sim = []
            for s2 in sentences:
                if s2 != s:
                    sim.append(self.get_similarity(s, s2))
            sims.append((i, np.mean(sim)))

        return sims

    def get_similarity(self, s1, s2):
        v1 = self.__text_to_vector(s1)
        v2 = self.__text_to_vector(s2)

        return self.__get_cosine(v1, v2)

    def __get_cosine(self, vec1, vec2):
        intersection = set(vec1) & set(vec2)
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x]**2 for x in vec1.keys()])
        sum2 = sum([vec2[x]**2 for x in vec2.keys()])
        denominator = math.sqrt(sum1) * math.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    def __text_to_vector(self, sentence):
        tokenizer = WordTokenizer(self.language_params)
        words = tokenizer.tokenize(sentence)
        return Counter(words)

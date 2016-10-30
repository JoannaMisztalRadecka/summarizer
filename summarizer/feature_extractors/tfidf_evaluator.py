import sys

from sklearn.feature_extraction.text import TfidfVectorizer

from feature_extractors.sentence_evaluator import SentencesEvaluator
from tokenizer import WordTokenizer


class TfIdfEvaluator(SentencesEvaluator):
    def __init__(self, language_params):
        super(TfIdfEvaluator, self).__init__(language_params, "TF-IDF Evaluator")
        self.tokenizer = WordTokenizer(self.language_params)
        self.tf_idf = TfidfVectorizer(tokenizer=self.tokenizer.tokenize)

    def train(self, training_set):
        self.tf_idf.fit(training_set)

    def evaluate(self, sentences):
        words_weights = self.__get_words_weights(sentences)
        sentences_weights = []

        for i, s in enumerate(sentences):
            words = self.tokenizer.tokenize(s)
            weights_sum = sum([words_weights.get(w, 0) for w in words])
            if len(words) > 0:
                sentences_weights.append((i, float(weights_sum)))

        return sorted(sentences_weights, reverse=True)

    def __get_words_weights(self, test_set):
        weights = self.tf_idf.transform([''.join(test_set)]).toarray()[0]
        features = self.tf_idf.get_feature_names()
        f_weights = zip(features, weights)
        return dict(f_weights)

    def encode_list(self, list):
        return [self.__encode_text(a) for a in list]

    def __encode_text(self, text):
        return text.encode(sys.stdout.encoding, errors='replace')



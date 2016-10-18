import pandas as pd


class SentencesFeatures(object):

    def __init__(self, language_params, evaluator_classes, sent_tokenizer, train_texts):
        self.language_params = language_params
        self.evaluator_classes = evaluator_classes
        self.sent_tokenizer = sent_tokenizer
        self.train_texts = train_texts

    def evaluate_sentences(self, test_text):
        sentences = self.sent_tokenizer.tokenize(test_text)
        sentences_weights = {}
        for e in self.evaluator_classes:
            weights = self.__get_sentences_weights(e, sentences)
            for i in weights:
                sentences_weights.setdefault(i, []).append(weights[i])

        return pd.DataFrame(sentences_weights).T

    def __get_sentences_weights(self, evaluator_class, sentences):
        evaluator = evaluator_class(self.language_params)
        print(evaluator)
        evaluator.train(self.train_texts)

        return dict(evaluator.evaluate(sentences))



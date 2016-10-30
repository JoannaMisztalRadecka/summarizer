import operator
import os
from unidecode import unidecode
import pandas as pd
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.svm import SVR

class Summarizer(object):
    def __init__(self, articles, sentence_tokenizer, features_extractor,
                 sentences_selector):
        self.articles = articles
        self.sentence_tokenizer = sentence_tokenizer
        self.features_extractor = features_extractor
        self.selector = sentences_selector
        self.classifier = SVR()

    def train(self):
        features = []
        output = []

        for a in self.articles:
            for f, o in self.selector.get_article_input_output(a):
                features.append(f)
                output.append(o)

        features_df = pd.concat(features)
        print(features_df)
        output_df = pd.concat(output)
        print(output_df)
        self.classifier.fit(features_df, output_df)

    def predict(self, text):
        sentences = self.sentence_tokenizer.tokenize(text)
        sentences_probs = self.classifier.predict(self.features_extractor.evaluate_sentences(text))
        sorted_probs = sorted(enumerate(sentences_probs), key=operator.itemgetter(1), reverse=True)
        summary_sents = sorted([(i, sentences[i]) for i, v in sorted_probs[:5]])
        summary = '\n'.join([s[1] for s in summary_sents])

        return summary

    def test_article(self, article, article_id):
        system_summaries = {}
        for i, s in enumerate(article.summaries):
            sents = self.sentence_tokenizer.tokenize(unidecode(s))
            system_summaries['s_{}'.format(i)] = '\n'.join(sents)
        model_summary = self.predict(article.text)
        return model_summary, system_summaries

import pandas as pd
from feature_extractors.similarity_evaluator import SimilarityEvaluator


class SentencesSelector(object):
    def __init__(self, sentence_tokenizer, features_extractor, language_params):
        self.sentence_tokenizer = sentence_tokenizer
        self.features_extractor = features_extractor
        self.language_params = language_params

    def get_article_input_output(self, article):
        features = self.features_extractor.evaluate_sentences(article.text)
        sentences = self.sentence_tokenizer.tokenize(article.text)
        for s in article.summaries:
            output = self.get_output_distance(s, sentences)
            if output.sum() > 0:
                yield features, output

    def get_output(self, summary, sentences):
        selected_sentences = self.sentence_tokenizer.tokenize(summary)
        sent_indexes = [sentences.index(c) for c in selected_sentences if c in sentences]
        sent_dict = dict.fromkeys(range(len(sentences)), 0)
        sent_dict.update(dict.fromkeys(sent_indexes, 1))

        return pd.Series(sent_dict)

    def get_output_distance(self, summary, sentences):
        selected_sentences = self.sentence_tokenizer.tokenize(summary)
        similarity_evaluator = SimilarityEvaluator(self.language_params)
        sent_dict = {}
        for i, s in enumerate(sentences):
            dists = [similarity_evaluator.get_similarity(s, s1) for s1 in selected_sentences]
            max_dist = max(dists)
            sent_dict[i] = max_dist
        # sent_indexes = [sentences.index(c) for c in selected_sentences if c in sentences]
        # sent_dict = dict.fromkeys(range(len(sentences)), 0)
        # sent_dict.update(dict.fromkeys(sent_indexes, 1))

        return pd.Series(sent_dict)


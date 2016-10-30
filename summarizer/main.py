import settings
from parameters import LanguageParams
from sentences_features import SentencesFeatures
from corpus_reader import SummariesCorpusReader
from tokenizer import SentenceTokenizer
from sentences_selector import SentencesSelector
from summarizer import Summarizer

from pyrouge import Rouge155


language_params = LanguageParams(settings.LANGUAGE)
summary_reader = SummariesCorpusReader(language_params.summaries_corpora)
articles = summary_reader.get_articles(summary_reader.fileids())
texts = [a.text for a in articles]
train_texts = texts[:1000]
sentence_tokenizer = SentenceTokenizer(language_params)
features_extractor = SentencesFeatures(language_params, settings.FEATURES_EXTRACTORS, sentence_tokenizer, train_texts)
selector = SentencesSelector(sentence_tokenizer, features_extractor, language_params)

summarizer = Summarizer(articles, sentence_tokenizer, features_extractor, selector)
features_extractor.train_evaluators()
summarizer.train()

# test_text = test_texts[0]
# test_sentences = sentence_tokenizer.tokenize(test_text)
model_summary, system_summaries = summarizer.test_article(articles[0], 0)


rouge = Rouge155(settings.ROUGE_PATH)
# r.system_dir = 'system_summaries'
# r.model_dir = 'model_summaries'
# r.system_filename_pattern = 'system_summary_(\d+).txt'
# r.model_filename_pattern = 'model_summary_(\d+)_(\d+).txt'
#
# output = r.convert_and_evaluate()
# print(output)
# output_dict = r.output_to_dict(output)
#
score = rouge.score_summary(model_summary, system_summaries)
print(score)
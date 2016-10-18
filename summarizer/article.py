import sys


class Article(object):

    def __init__(self, doc_id, title, author, text, date, summaries):
        self.doc_id = doc_id
        self.title = title
        self.text = text
        self.author = author
        self.date = date
        self.summaries = summaries

    def __repr__(self):
        return str('Article {}: {} ({})'.format(self.doc_id, self.title, self.author).\
            encode(sys.stdout.encoding, errors='replace'))

    def __str__(self):
        return self.__repr__()

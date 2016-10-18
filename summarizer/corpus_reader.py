from nltk.corpus.reader.xmldocs import XMLCorpusReader, XMLCorpusView
from article import Article

class SummariesCorpusReader(XMLCorpusReader):

    def __init__(self, root, fileids='.*'):
        XMLCorpusReader.__init__(self, root, fileids)

    def get_articles(self,  fileids=None, **kwargs):
        """
        Returns specified articles.
        """

        articles = []
        if fileids is None:
            fileids = self._fileids
        for fileid in fileids:
            elt = self.xml(fileid)
            doc_id = fileid.split('.')[0]
            text = elt.find('body').text
            author = elt.find('authors').text
            title = elt.find('title').text
            date = elt.find('date').text
            summaries = [s.text for s in elt.findall('summaries/summary/body')]
            articles.append(Article(doc_id, title, author, text, date, summaries))

        return articles


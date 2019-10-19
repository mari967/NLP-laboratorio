#Steam y stop words del idioma español
from sklearn.feature_extraction.text import CountVectorizer
#from stop_words import get_stop_words
from nltk.corpus import stopwords

#stop_words_es = get_stop_words('spanish')
stop_words_es = stopwords.words('spanish')
vectorizer = CountVectorizer(stop_words = stop_words_es)

import nltk
ES_stemmer = nltk.stem.SnowballStemmer('spanish') 

class StemmedCountVectorizerES(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizerES, self).build_analyzer()
		return lambda doc: (ES_stemmer.stem(w) for w in analyzer(doc))


stem_vectorizer_ES = StemmedCountVectorizerES(min_df = 1,stop_words = stop_words_es)
stem_analyze = stem_vectorizer_ES.build_analyzer()



def imprimir_tok(a, oracion):
	print("\n", oracion, "\n")
	for tok in a:
		print(tok)

oracion1 = "Fran tiene pesadillas con sistemas operativos"
oracion2 = "Hay cuatro frascos de duraznos en la mesa"

A = stem_analyze(oracion1)
B = stem_analyze(oracion2)

imprimir_tok(A, oracion1)
imprimir_tok(B, oracion2)


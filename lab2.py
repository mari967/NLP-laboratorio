import nltk

from nltk.book import *

#treebank.tagged_words("wsj_0001.mrg")[0:]
#x100 = treebank.tagged_words("wsj_0003.mrg")[0:100]
#for p in x100:
  #print(p)
 # print(p[0])

raceNN = nltk.tag.str2tuple('race/NN')
raceVB = nltk.tag.str2tuple('race/VB')

from nltk.corpus import brown
tamanioBrown = len(brown.tagged_words())
print("\n\nTamaño del corpus Brown: ", tamanioBrown)
cantRaceBrownNN =brown.tagged_words().count(raceNN)
cantRaceBrownVB = brown.tagged_words().count(raceVB)

print("Número de veces que \"race\" aparece como NN: ",cantRaceBrownNN)
print("Número de veces que \"race\" aparece como VB: ", cantRaceBrownVB)
if cantRaceBrownNN > cantRaceBrownVB:
	print("Era lo que me imaginaba B)\n")
else: print("No fue lo que esperaba O.o\n")

from nltk.corpus import brown
from nltk.tag import hmm

#Con Unigram tagger
news_sents = brown.tagged_sents(categories = "news")[:10000]
#modelo entrenado
unigram_tagger = nltk.tag.UnigramTagger(news_sents)

from nltk import word_tokenize

sentence = "The secretariat is expected to race tomorrow"
sentence_tokens =  word_tokenize(sentence)
sentence_tagsU = unigram_tagger.tag(sentence_tokens)
print("Para la oración ", sentence,"\nTags con UnigramTagger: ", sentence_tagsU)

#Con el HMM tagger

trainer = hmm.HiddenMarkovModelTrainer()
hmm_tagger = trainer.train_supervised(news_sents)
sentence_tagsHMM = hmm_tagger.tag(sentence_tokens)

print("Tags con hmm_tagger: ",  sentence_tagsHMM)

print("\n\nPrueba con los modelos Unigram y HMM con oraciones ambiguas")

#Oraciones ambiguas
sentence1 = "I saw a man with a telescope"
sentence2 = "Let's stop controlling people"
sentence3 = "I'll book a room for you"
sentence4 = "They are eating apples"

sentence1_tokens = word_tokenize(sentence1)
sentence2_tokens = word_tokenize(sentence2)
sentence3_tokens = word_tokenize(sentence3)
sentence4_tokens = word_tokenize(sentence4)

#Con Unigram model
sentence1_tagsU = unigram_tagger.tag(sentence1_tokens)
sentence2_tagsU = unigram_tagger.tag(sentence2_tokens)
sentence3_tagsU = unigram_tagger.tag(sentence3_tokens)
sentence4_tagsU = unigram_tagger.tag(sentence4_tokens)

#Con HMM 
sentence1_tagsHMM = hmm_tagger.tag(sentence1_tokens)
sentence2_tagsHMM = hmm_tagger.tag(sentence2_tokens)
sentence3_tagsHMM = hmm_tagger.tag(sentence3_tokens)
sentence4_tagsHMM = hmm_tagger.tag(sentence4_tokens)

print("1) ",sentence1,"\nTags unigram: ", sentence1_tagsU)
print("Tags HMM: ", sentence1_tagsHMM, '\n')
print("2) ",sentence2, "\nTags unigram: ", sentence2_tagsU)
print("Tags HMM: ", sentence2_tagsHMM, '\n')
print("3) ",sentence3, "\nTags unigram: ", sentence3_tagsU)
print("Tags HMM: ", sentence3_tagsHMM, '\n')
print("4) ",sentence4, "\nTags unigram: ", sentence4_tagsU)
print("Tags HMM: ", sentence4_tagsHMM, '\n')


headline = "Juvenile court to Try Shooting Defendant"

headline_tokens = word_tokenize(headline)
headline_tags = hmm_tagger.tag(headline_tokens)
print("Tags del headline: ",headline_tags)

print("\n~~~~~~~~~~~~~~ * ~~~~~~~~~~~~~~~\n")

#Tamaño del corpus Brown:  1161192
#Número de veces que "race" aparece como NN:  94
#Número de veces que "race" aparece como VB:  4
#Era lo que me imaginaba B)

#Para la oración  The secretariat is expected to race tomorrow
#Tags con UnigramTagger:  [('The', 'AT'), ('secretariat', None), ('is', 'BEZ'), ('expected', 'VBN'), ('to', 'TO'), ('race', 'NN'), ('tomorrow', 'NR')]
#Tags con hmm_tagger:  [('The', 'AT'), ('secretariat', 'AT'), ('is', 'AT'), ('expected', 'AT'), ('to', 'AT'), ('race', 'AT'), ('tomorrow', 'AT')]


#Prueba con los modelos Unigram y HMM con oraciones ambiguas
#1)  I saw a man with a telescope
#Tags unigram:  [('I', 'PPSS'), ('saw', 'VBD'), ('a', 'AT'), ('man', 'NN'), ('with', 'IN'), ('a', 'AT'), ('telescope', None)]
#Tags HMM:  [('I', 'PPSS'), ('saw', 'VBD'), ('a', 'AT'), ('man', 'NN'), ('with', 'IN'), ('a', 'AT'), ('telescope', 'AT')]

#2)  Let's stop controlling people
#Tags unigram:  [('Let', 'VB'), ("'s", None), ('stop', 'VB'), ('controlling', 'VBG'), ('people', 'NNS')]
#Tags HMM:  [('Let', 'VB'), ("'s", 'AT'), ('stop', 'AT'), ('controlling', 'AT'), ('people', 'AT')]

#3)  I'll book a room for you
#Tags unigram:  [('I', 'PPSS'), ("'ll", None), ('book', 'NN'), ('a', 'AT'), ('room', 'NN'), ('for', 'IN'), ('you', 'PPSS')]
#Tags HMM:  [('I', 'NN'), ("'ll", 'AT'), ('book', 'AT'), ('a', 'AT'), ('room', 'AT'), ('for', 'AT'), ('you', 'AT')]

#4)  They are eating apples
#Tags unigram:  [('They', 'PPSS'), ('are', 'BER'), ('eating', 'VBG'), ('apples', None)]
#Tags HMM:  [('They', 'PPSS'), ('are', 'BER'), ('eating', 'VBG'), ('apples', 'AT')]

#Tags del headline:  [('Juvenile', 'JJ-TL'), ('court', 'NN'), ('to', 'IN'), ('Try', 'AT'), ('Shooting', 'AT'), ('Defendant', 'AT')]

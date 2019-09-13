import nltk
from nltk import word_tokenize
from nltk import re

#nltk.download()
from nltk.book import *

#Ahora con moby dick >CD

moby_tokens = text1.tokens
moby_tokens_sin_puntos = [palabra.lower() for palabra in moby_tokens if re.search("\w", palabra)]
#Numero de tokens en moby dick (sin signos de puntuación)
nro_tokens_sin_puntos = len(moby_tokens_sin_puntos)
print("1) Numero de tokens en Moby Dick: ", nro_tokens_sin_puntos)
#Numero de palabras unicas o nro de TYPES ****
nro_types = len(set(moby_tokens_sin_puntos))
print("2) Numero de TYPES en Moby Dick: ", nro_types)
#Type token ratio de Moby Dick
moby_type_token_ratio = nro_types/nro_tokens_sin_puntos
print("3) Type token ratio de Moby Dick: ", moby_type_token_ratio)



#0.07840051962071347

#----------------------------Ahora el WSJ

wsj_tokens = text7.tokens
wsj_tokens_sin_puntos = [palabra.lower() for palabra in wsj_tokens if re.search("\w", palabra)]
nro_tokens_wsj_sin_puntos = len(wsj_tokens_sin_puntos)
nro_types_wsj = len(set(wsj_tokens_sin_puntos))

wsj_type_token_ratio = nro_types_wsj/nro_tokens_wsj_sin_puntos
print("4) Type token ratio del WSJ: ",wsj_type_token_ratio)

if moby_type_token_ratio > wsj_type_token_ratio:
	print("5) Moby Dick tiene mayor diversidad lexica ")
else:
	print("5) El WSJ tiene mayor diversidad lexica ")

print("6) ...")

#0.129748424801388

#Ahora el MLE
moby_N = nro_tokens_sin_puntos

moby_count_whale = moby_tokens_sin_puntos.count("whale")

moby_MLE_whale = moby_count_whale/moby_N
print("7) Pmoby_dick(\"whale\") = ", moby_MLE_whale)
#0.005607878474620462

#----------------------------------Ahora con el WSJ

wsj_N = nro_tokens_wsj_sin_puntos
wsj_count_whale = wsj_tokens_sin_puntos.count("whale")
wsj_MLE_whale = wsj_count_whale/wsj_N
print("8) Pwsj(\"whale\") = ", wsj_MLE_whale)
#0.0


import time
from keras.models import load_model
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

# ===========================================
# Usa o modelo criado em uma nova sentenca
# ===========================================
print("Inicio da classificacao pelo modelo LSTM")
start_time = time.time()

#carrega o dicionario de palavras e tags
# with open('pickledData/data_train.pkl', 'rb') as f:
with open('pickledData/data1.pkl', 'rb') as f:
	X_train, Y_train, word2int, int2word, tag2int, int2tag = pickle.load(f)

	del X_train
	del Y_train


#carrega o modelo treinado
model = load_model('models/model1.h5')

#carrega o arquivo de teste para validacao do modelo criado
#avalia por classe (sendo classe qual a tag semantica correspondente
TEST_FILE = 'macmorpho-v3//macmorpho-test.txt'

raw_corpus = ''
with open(TEST_FILE, "r", encoding="utf8") as f:
    raw_corpus = raw_corpus + '\n' + f.read()

corpus = raw_corpus.split('\n')

avaliacao_classe_correta = {}
avaliacao_classe_errada = {}
avaliacao_classe_total = {}
corretas = 0
erradas = 0
tags_incorretas = 0
classes_nomes = []

for key, value in tag2int.items():
	avaliacao_classe_correta[key] = 0
	avaliacao_classe_errada[key] = 0
	avaliacao_classe_total[key] = 0
	classes_nomes.append(key)

#sentence = 'consolidado fafa as gelo funda regras reconstrução 136 estudada passadores nomes lavoura espelho gerado 18,01 vôo'.split()
#sentence = 'Salto sete Terragran ; Hemocentro Estragão'.split() #Salto_N sete_ADJ Terragran_NPROP ;_PU Hemocentro_N estragão_N

for line in corpus:
	print("...Nova sentenca...")
	tags_corretas = []
	tags_previstas = []
	classes_nomes_local = []

	for word_tag in line.split():

		word, tag = word_tag.split('_')
		# word = word.lower()
		# words.append(w)

		tokenized_sentence = []
		sentence_limpa = ''

		#remove as palavras que nao sao conhecidas no dicionario
		# for word in sentence:
		if word in word2int:
			tokenized_sentence.append(word2int[word])
			sentence_limpa = sentence_limpa + word + ' '
			# tags_corretas.append(tag)
		else:
			print("Palavra desconhecida (removida): ", word)

		sentence = sentence_limpa.split()
		# print('sentence_limpa')
		# print(sentence_limpa)
		tokenized_sentence = np.asarray([tokenized_sentence])
		padded_tokenized_sentence = pad_sequences(tokenized_sentence, maxlen=50)
		# print('A sentenca tokenized ',tokenized_sentence)
		# print('A sentenca tokenized padded', padded_tokenized_sentence)

		prediction = model.predict(padded_tokenized_sentence)
		i = 0
		for pred in prediction[0]:
			# try:
			resultado = int2tag[np.argmax(pred)]
			if resultado != 'PAD_tag':
				print(sentence[i], ' : ', resultado, ' : ', tag)
				i = i + 1
				tags_corretas.append(tag2int[tag])
				tags_previstas.append(tag2int[resultado])
				classes_nomes_local.append(tag)
				avaliacao_classe_total[tag] = avaliacao_classe_total[tag] + 1
				if tag == resultado:
					avaliacao_classe_correta[tag] = avaliacao_classe_correta[tag] + 1
				else:
					avaliacao_classe_errada[tag] = avaliacao_classe_errada[tag] + 1
			else:
				tags_incorretas = tags_incorretas + 1

			# except Exception as e:
				# print(e)
				# pass

# print('incorretas: ', tags_incorretas)
# print('corretas: ', avaliacao_classe_correta)
# print('erradas: ', avaliacao_classe_errada)
# print('total: ', avaliacao_classe_total)
print('acuracia por classe')
print('classe   ---   acuracia   ---   total corretas --- total incorretas')
for key, value in avaliacao_classe_total.items():
	acuracia = 0
	total = avaliacao_classe_correta[key] + avaliacao_classe_errada[key];
	if total != 0:
		acuracia = avaliacao_classe_correta[key] / (avaliacao_classe_correta[key] + avaliacao_classe_errada[key])

	print(key, ' -- ', acuracia, ' -- ', avaliacao_classe_correta[key], ' -- ', avaliacao_classe_errada[key])

print("Fim da classificacao em %s segundos", (time.time() - start_time))

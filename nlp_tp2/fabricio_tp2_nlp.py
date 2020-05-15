import time
import pickle
import numpy as np
import os, sys
from collections import defaultdict
from functools import partial
from glove import Corpus, Glove

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

def generator(all_X, all_y, n_classes, batch_size=32, max_seq_len=50):
    num_samples = len(all_X)

    while True:

        for offset in range(0, num_samples, batch_size):
            X = all_X[offset: offset + batch_size]
            y = all_y[offset: offset + batch_size]

            X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
            y = pad_sequences(y, maxlen=MAX_SEQUENCE_LENGTH)

            y = to_categorical(y, num_classes=n_classes)

            yield shuffle(X, y)


# ===========================================
# Arquivos e variaveis de inicializacao
# ===========================================
start_time = time.time()
start_time_geral = time.time()
print("Inicio arquivos e variaveis...")

GLOVE_NUM_EPOCHS_TRAINING = 5
GLOVE_LEARNING_RATE = 0.5

MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 50
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 32
MODEL_NUM_EPOCHS = 10

#arquivo que sera SALVO o modelo
MODEL_FILE = "model10.h5"

#arquivo que sera SALVO o modelo GloVe
MODEL_FILE_GLOVE = "glove10.model"

#arquivo que sera SALVO o dicionario com o modelo de palavras, word2int, int2tag....
DICTIONARY_FILE = "data10.pkl"

#arquivo que sera SALVO a estrura
GLOVE_TRAINING_FILE = "macmorpho-v3\\glove_macmorpho10.txt"
glove_file = open(GLOVE_TRAINING_FILE, "w+", encoding="utf8")

#arquivo que sera CARREGADO para treino do modelo
TRAINING_FILE = "macmorpho-v3//macmorpho-train.txt"
raw_corpus = ''
with open(TRAINING_FILE, "r", encoding="utf8") as f:
    raw_corpus = raw_corpus + '\n' + f.read()

corpus = raw_corpus.split('\n')

print('Tamanho do Corpus', len(corpus), '\n')

# ===========================================
# Extrai as palavras e as tags do arquivo de treino
# ===========================================
X_train = []
Y_train = []

words = []
tags = []

for line in corpus:
    if (len(line) > 0):
        tempX = []
        tempY = []

        for word in line.split():
            try:
                w, tag = word.split('_')
            except:
                break

            w = w.lower()
            words.append(w)
            tags.append(tag)

            tempX.append(w)
            tempY.append(tag)

        X_train.append(tempX)
        Y_train.append(tempY)

print('Quantidade total de sentenças: ', len(X_train), '\n')
print('Exemplo de sentença (palavras): ', X_train[1], '\n')
print('Exemplo de sentença (tags): ', Y_train[1], '\n')

words = set(words)
tags = set(tags)

print('Quantidade palavras distintas: ', len(words))
print('Quantidade Tags distintas: ', len(tags))

# assert len(X_train) == len(Y_train)

# ===========================================
# Cria o mapeamento de vetores de palavras e tags para inteiro  word2int; int2word; tag2int; int2tag
# ===========================================

word2int = {}
int2word = {}

# cria o dicionario de palavras word2int e int2word
for i, word in enumerate(words):
    word2int[word] = i + 1
    int2word[i + 1] = word

word2int['PAD_word'] = 0
int2word[0] = 'PAD_word'

# Necessario o tratamento das palavras que nao sao conhecidas no modelo.
# As palavra que compoe a sentenca a ser classificada devem ser conhecidas pelo modelo
# Adicionado o token <UNK> (unknown)
word2int['<UNK>'] = len(int2word)
int2word[len(int2word)] = '<UNK>'

# cria o dicionario de tags tag2int e int2tag
tag2int = {}
int2tag = {}

for i, tag in enumerate(tags):
    tag2int[tag] = i + 1
    int2tag[i + 1] = tag

tag2int['PAD_tag'] = 0
int2tag[0] = 'PAD_tag'


# ===========================================================

X_train_numberised = []
Y_train_numberised = []

for sentence in X_train:
    tempX = []
    for word in sentence:
        tempX.append(word2int[word])
    X_train_numberised.append(tempX)

for tags in Y_train:
    tempY = []
    for tag in tags:
        tempY.append(tag2int[tag])
    Y_train_numberised.append(tempY)

print('Exemplo de sentença (palavras) representada por vetor: ', X_train_numberised[1], '\n')
print('Exemplo de sentença (tags) representada por vetor: ', Y_train_numberised[1], '\n')

X_train_numberised = np.asarray(X_train_numberised)
Y_train_numberised = np.asarray(Y_train_numberised)

#cria um arquivo pickle, compactado, contendo as sentencas e dados extraidos e os dicionarios de palavras e tags
#utilizado para ser carregado a memoria para avaliar novas sentencas
pickle_files = [X_train_numberised, Y_train_numberised, word2int, int2word, tag2int, int2tag]

#salva o arquivo em formato pickled
if not os.path.exists('pickledData/'):
    print('criando diretorio para salvar o arquivo pickled ')
    os.makedirs('pickledData/')
with open('pickledData/' + DICTIONARY_FILE, 'wb') as f:
    pickle.dump(pickle_files, f)

print("Fim arquivos e variaveis em %s segundos", (time.time() - start_time))
# ===========================================

# ===========================================
print("Inicio criacao modelo glove")
start_time = time.time()

corpus = Corpus()
# treina o corpus para gerar a matrix de co ocorrencia utilizado no GloVe
#com o tamanho da janela considerando quantas palavras no contexto
corpus.fit(X_train, window=10)
# cria o arquivo GloVe, contendo a dimensao (no_componentes) e o learning_rate. Constantes declaradas no inicio
glove = Glove(no_components=EMBEDDING_DIM, learning_rate=GLOVE_LEARNING_RATE)
glove.fit(corpus.matrix, epochs=GLOVE_NUM_EPOCHS_TRAINING, no_threads=4, verbose=True) #glove.fit(corpus.matrix, epochs=10, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

if not os.path.exists('models/'):
    print('criando diretorio para salvar os modelos')
    os.makedirs('models/')

glove.save('models/' + MODEL_FILE_GLOVE)

for word_dict in glove.dictionary:
    glove_file.write(word_dict + " ")

    for values in glove.word_vectors[glove.dictionary[word_dict]]:
        glove_file.write(str(values) + " ")
        glove_file.write("\n")

glove_file.close()

print("Termino criacao modelo GloVe em %s segundos", (time.time() - start_time))

# ===========================================
# embeddings_index criacao dos pesos da palavra de acordo com o contexto
# ===========================================
print("Inicio criacao dos pesos da palavra de acordo com o contexto")
start_time = time.time()

embeddings_index = {}

with open(GLOVE_TRAINING_FILE, encoding="utf8") as glove_file:
    for line in glove_file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# print(embeddings_index)

print("Termino criacao dos pesos da palavra de acordo com o contexto em %s segundos", (time.time() - start_time))
# ===========================================
# Treina o Modelo de Classificacao LSTM
# ===========================================
print("Inicio criacao do modelo LSTM")
start_time = time.time()

n_tags = len(tag2int)

# embaralha(shuffle) os dados
X_train_numberised, Y_train_numberised = shuffle(X_train_numberised, Y_train_numberised)

# split os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_train_numberised, Y_train_numberised, test_size=TEST_SPLIT,random_state=42)

# split os dados de treino para treino e validacao
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SPLIT, random_state=1)

n_train_samples = X_train.shape[0]
n_val_samples = X_val.shape[0]
n_test_samples = X_test.shape[0]

print('*******************************')
print(X_train)
print('*******************************')

print('Quantidade de exemplos de TREINO %d ' % n_train_samples)
print('Quantidade de exemplos de VALIDACAO %d ' % n_val_samples)
print('Quantidade de exemplos de TESTE %d ' % n_test_samples)

# o numero de classes devem ser um a mais do que o numero de tags.
# para ser a classificacao da palavra
n_classes = n_tags + 1

train_generator = generator(all_X=X_train, all_y=y_train, n_classes=n_classes, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQUENCE_LENGTH)
validation_generator = generator(all_X=X_val, all_y=y_val, n_classes=n_classes, batch_size=BATCH_SIZE, max_seq_len=MAX_SEQUENCE_LENGTH)

print('Total %s vetores de palavras.' % len(embeddings_index))

# + 1 para incluir <UNK> words
embedding_matrix = np.random.random((len(word2int) + 1, EMBEDDING_DIM))

for word, i in word2int.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # palavras desconhecidas nao serao modificadas (ou quando rodar o random de novo)
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape', embedding_matrix.shape)
print('X_train shape', X_train.shape)

embedding_layer = Embedding(len(word2int) + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=True)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)

l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
preds = TimeDistributed(Dense(n_tags + 1, activation='softmax'))(l_lstm)
model = Model(sequence_input, preds)

#Aqui pode utilizar outros parametros. Estes podem ser encontrados em keras.io
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
#
print("model fitting - Bidirectional LSTM")
model.summary()

model.fit_generator(train_generator, steps_per_epoch=n_train_samples//BATCH_SIZE, validation_data=validation_generator, validation_steps=n_val_samples//BATCH_SIZE,
                     epochs=MODEL_NUM_EPOCHS, verbose=1, workers=4)

#SALVA O MODELO
model.save('models/' + MODEL_FILE)

X_test = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH)
y_test = pad_sequences(y_test, maxlen=MAX_SEQUENCE_LENGTH)
y_test = to_categorical(y_test, num_classes=n_classes)

test_results = model.evaluate(X_test, y_test, verbose=1)
print('TEST LOSS %f \nTEST ACCURACY: %f' % (test_results[0], test_results[1]))
print(test_results)


print("Termino do treino/test do modelo LSTM", (time.time() - start_time))
# ===========================================
# Fim
# ===========================================
print("Tempo gasto para treinamento:", (time.time() - start_time_geral))

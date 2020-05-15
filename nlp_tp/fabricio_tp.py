from word2veclite import Word2Vec
import numpy as np
import matplotlib.pyplot as plt

var_learning_rate = 0.1
var_method = "skipgram" #cbow ou skipgram
var_window_size = 5 #normalmente 2
var_n_epochs = 100
#corpus = "In Brazil, around 09:15, the hard type 6 was traded at R $ 500.00 a bag of 60 kg in Varginha (MG) - stable, in Guaxupé (MG) prices also remained stable at R $ 505.00 a And in Espírito Santo do Pinhal (SP) was being quoted at R $ 520,00 a bag."
corpus = "On the down side, the best weather conditions in Brazil weigh on the market."

cbow = Word2Vec(method=var_method, corpus=corpus,
                window_size=var_window_size, n_hidden=2,
                n_epochs=var_n_epochs, learning_rate=var_learning_rate)
W1, W2, loss_vs_epoch = cbow.run()


plt_title = 'Method={} window_size={} n_epochs = {} learning_rate={}'.format(var_method, var_window_size, var_n_epochs, var_learning_rate)
plt.plot(loss_vs_epoch)
plt.title(plt_title)
plt.xlabel('Numero de Epochs')
plt.ylabel('Loss')
plt.show()

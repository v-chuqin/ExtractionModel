from ipdb import set_trace
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense, Masking
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import rmsprop, sgd, Adagrad
from keras.layers.wrappers import TimeDistributed
from mdata import loaddata,calprecision


# inputData,outputData,simpleWeight = loaddata('tweets.threshold_6.txt.model','tweets.threshold_6.txt','tweetskey.threshold_6.txt')
inputData,outputData,simpleWeight = loaddata('tweetsTest_clean_tweets.threshold_6.txt.model','tweets.threshold_6.txt','tweetskey.threshold_6.txt')
# set_trace()
maxlen = inputData.shape[1]
word_dim = inputData.shape[2]

RNN = GRU
HIDDEN_SIZE = 256
LAYERS = 3
ActivationFunction = 'softmax' #relu
BATCH_SIZE = 32
nb_epoch = 100

model = Sequential()
# M = Masking(mask_value=0.)
# M._input_shape = (maxlen, word_dim)
model.add(Masking(mask_value=0.,input_shape=(maxlen, word_dim)))
for _ in range(LAYERS):
	model.add(RNN(HIDDEN_SIZE,return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(Activation(ActivationFunction))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'],sample_weight_mode="temporal")
model.fit(inputData,outputData,nb_epoch=nb_epoch,batch_size=BATCH_SIZE,sample_weight=simpleWeight[:,:,0])
predicted_output = model.predict(inputData, batch_size=BATCH_SIZE)
set_trace()



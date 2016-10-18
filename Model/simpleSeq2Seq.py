from ipdb import set_trace
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense, Masking, RepeatVector
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import rmsprop, sgd, Adagrad
from keras.layers.wrappers import TimeDistributed
from mdata import loaddata,calprecision,generatetestdata

# inputData,outputData,simpleWeight = loaddata('tweets.threshold_6.txt.model','tweets.threshold_6.txt','tweetskey.threshold_6.txt')
# inputData,outputData,simpleWeight = loaddata('tweetsTest_clean_tweets.threshold_6.txt.model','tweets.threshold_6.txt','tweetskey.threshold_6.txt')

inputData,outputData,simpleWeight = generatetestdata('tweetsTest_clean.tsv','tweetsTest_clean_tweets.threshold_6.txt.model',1000)

# set_trace()

maxlen = inputData.shape[1]
word_dim = inputData.shape[2]
# RNN = SimpleRNN
# RNN = LSTM
RNN = GRU
HIDDEN_SIZE = 256
LAYERS = 1
ActivationFunction = 'sigmoid' 
BATCH_SIZE = 256
nb_epoch = 30
loss_funtion = 'binary_crossentropy' #'binary_crossentropy' ,'mse'

model = Sequential()
# M = Masking(mask_value=0.)
# M._input_shape = (maxlen, word_dim)
# model.add(Masking(mask_value=0.,input_shape=(maxlen, word_dim)))
# model.add(RNN(HIDDEN_SIZE, return_sequences=False))
model.add(RNN(HIDDEN_SIZE, return_sequences=False,input_shape=(maxlen, word_dim)))
model.add(RepeatVector(maxlen))
for _ in range(LAYERS):
	model.add(RNN(HIDDEN_SIZE,return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.add(Activation(ActivationFunction))
model.compile(loss=loss_funtion,optimizer='adam',metrics=['accuracy'])
model.fit(inputData,outputData,nb_epoch=nb_epoch,batch_size=BATCH_SIZE)
# model.compile(loss=loss_funtion,optimizer='adam',metrics=['accuracy'],sample_weight_mode="temporal")
# model.fit(inputData,outputData,nb_epoch=nb_epoch,batch_size=BATCH_SIZE,sample_weight=simpleWeight[:,:,0])
predicted_output = model.predict(inputData, batch_size=BATCH_SIZE)
set_trace()



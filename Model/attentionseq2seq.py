from ipdb import set_trace
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, TimeDistributedDense, Masking, RepeatVector
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
from keras.optimizers import rmsprop, sgd, Adagrad
from keras.layers.wrappers import TimeDistributed
from mdata import loaddata,calprecision,generatetestdata
import seq2seq
from seq2seq.models import AttentionSeq2Seq,Seq2Seq


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
BATCH_SIZE = 128
nb_epoch = 30
loss_funtion = 'binary_crossentropy' #'binary_crossentropy','mse'

model  = Sequential()
seq2seqmodel = AttentionSeq2Seq(input_dim=word_dim, input_length=maxlen, hidden_dim=HIDDEN_SIZE, output_length=maxlen, output_dim=HIDDEN_SIZE, depth=2)
model.add(seq2seqmodel)
model.add(TimeDistributed(Dense(1)))
model.add(Activation(ActivationFunction))
model.compile(loss=loss_funtion,optimizer='adam',metrics=['accuracy'])
model.fit(inputData,outputData,nb_epoch=nb_epoch,batch_size=BATCH_SIZE)
predicted_output = model.predict(inputData, batch_size=BATCH_SIZE)
set_trace()





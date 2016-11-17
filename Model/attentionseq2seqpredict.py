# -*- coding: utf-8 -*-

from mdata import loadembedding,prt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, RepeatVector, Merge
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.layers.core import Lambda
import keras.backend as K
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
import random, sys
import keras
import Levenshtein
import cPickle as pickle
from time import sleep
from ipdb import set_trace
import seq2seq
from seq2seq.models import AttentionSeq2Seq,Seq2Seq
import h5py


FN = 'attention_1000'

maxlend=50 # 0 - if we dont want to use description at all
maxlenh=50
# maxlen = maxlend + maxlenh
maxlen = maxlend
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm=False

activation_rnn_size = 40 if maxlend else 0

seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4

batch_size=64 #64
nflips=10

empty = 0
eos = 1
nb_unknown_words = 1

embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding(embeddingfileName="embedding_vocab_1000.pkl",datafileName="heads_desc_1000.pkl",test_size=128)
# embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding()
vocab_size, embedding_size = embedding.shape

def vocab_fold(xs):
	oov0 = vocab_size-nb_unknown_words
	xs = [x if x < oov0 else glove_idx2idx.get(x,x) for x in xs]
	outside = sorted([x for x in xs if x>= oov0])
	outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
	xs = [outside.get(x,x) for x in xs]
	return xs

def vocab_unfold(desc,xs):
	oov0 = vocab_size-nb_unknown_words
	unfold = {}
	for i,unfold_idx in enumerate(desc):
		fold_idx = xs[i]
		if fold_idx >= oov0:
			unfold[fold_idx] = unfold_idx
	return [unfold.get(x,x) for x in xs]

def lpadd(x,maxlend=maxlend,eos=eos):
	assert maxlend >= 0
	if maxlend == 0:
		return [eos]
	n = len(x)
	if n > maxlend:
		x = x[-maxlend:]
		n = maxlend
	return [empty]*(maxlend-n) + x + [eos]

def flip_headline(x,nflips=None, model=None,debug=False):
	oov0 = vocab_size-nb_unknown_words
	if nflips is None or model is None or nflips <=0:
		return x
	batch_size = len(x)
	assert np.all(x[:,maxlend]==eos)
	probs = model.predict(x,verbose=0,batch_size=batch_size)
	x_out = x.copy()
	for b in range(batch_size):
		flips = sorted(random.sample(xrange(maxlend+1,maxlen),nflips))
		if debug and b < debug:
			print b,
		for input_idx in flips:
			if x[b,input_idx] == empty or x[b,input_idx] == eos:
				continue
			label_index = input_idx - (maxlend+1)
			prob = probs[b,label_index]
			w = prob.argmax()
			if w == empty:
				w = oov0
			if debug and b < debug:
				print '%s => %s'%(idx2word[x_out[b,input_idx]],idx2word[w]),
			x_out[b,input_idx] = w
		if debug and b < debug:
			print 
	return x_out

def conv_seq_labels(xds,xhs,nflips=None,model=None,debug=False):
	batch_size = len(xhs)
	assert len(xds) == batch_size
	x = [vocab_fold(lpadd(xd)) for xd,xh in zip(xds,xhs)]
	x = sequence.pad_sequences(x,maxlen=maxlen,value=empty,padding='post',truncating='post')

	y = np.zeros((batch_size,maxlenh, vocab_size))
	for i,xh in enumerate(xhs):
		xh = vocab_fold(xh) + [eos] + [empty]*maxlenh
		xh = xh[:maxlenh]
		y[i,:,:] = np_utils.to_categorical(xh, vocab_size)
	return x,y

def gen(Xd, Xh, batch_size=batch_size, nb_batches=None, nflips=None, model= None, debug=False,seed=seed):
	c = nb_batches if nb_batches else 0
	while 1:
		xds = []
		xhs = []
		if nb_batches and c >= nb_batches:
			c = 0
		new_seed = random.randint(0,sys.maxint)
		random.seed(c+123456789+seed)
		for b in range(batch_size):
			t = random.randint(0,len(Xd)-1)
			xd = Xd[t]
			# s = random.randint(min(maxlend,len(xd)),max(maxlend,len(xd)))
			xds.append(xd[:maxlend])
			xh = Xh[t]
			# s = random.randint(min(maxlenh,len(xh)),max(maxlenh,len(xh)))
			xhs.append(xh[:maxlend])
		c += 1
		random.seed(new_seed)

		yield conv_seq_labels(xds,xhs,nflips=nflips, model=model,debug=debug)

def Sample(X,Y,index,model=None):
	xds = []
	xd = X[index]
	xds.append(xd[:maxlend])
	z = [vocab_fold(lpadd(t)) for t in xds]
	z = sequence.pad_sequences(z,maxlen=maxlen,value=empty,padding='post',truncating='post')
	# set_trace()
	prt("D",X[index],idx2word)
	prt("D",z[0],idx2word)
	probs = model.predict(z,verbose=0)
	pre = []
	for i in range(len(probs[0])):
		y = sorted(enumerate(probs[0][i]),key=lambda x: x[1],reverse = True)
		pre.append(y[0][0])
	prt("L",Y[index],idx2word)
	prt("H",pre,idx2word)

def str_shape(x):
	return 'x'.join(map(str,x.shape))

def inspect_model(model):
	for i,l in enumerate(model.layers):
		print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
		weights = l.get_weights()
		for weight in weights:
			print str_shape(weight),
		print 

if __name__ == "__main__":

	random.seed(seed)
	np.random.seed(seed)
	regularizer = l2(weight_decay) if weight_decay else None
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size,input_length=maxlen,
						W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,name='embedding_1'))
	seq2seqmodel = AttentionSeq2Seq(input_dim=embedding_size, input_length=maxlen, hidden_dim=rnn_size, output_length=maxlen, output_dim=rnn_size, depth=3,name='seq2seq')
	model.add(seq2seqmodel)
	model.add(TimeDistributed(Dense(vocab_size,W_regularizer=regularizer,b_regularizer=regularizer,name='timeDistributed_1')))
	model.add(Activation('softmax',name='activation_1'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	K.set_value(model.optimizer.lr,np.float32(LR))

	inspect_model(model)

	try:
		file=h5py.File('data/%s.hdf5'%FN,'r')
		weight = []
		for i in range(0,len(file.keys())):
			weight.append(file['weight'+str(i)][:])
		model.set_weights(weight)
		# model.load_weights('data/%s.hdf5'%FN)
	except:
		print "first time training"
	# set_trace()
	try:
		with open('data/%s.history.pkl'%FN, 'rb') as fp:
			history = pickle.load(fp)
	except:
		history = {}
	# traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
	# valgen = gen(X_test, Y_test, nb_batches=len(X_test)//batch_size, batch_size=batch_size)
	# r = next(traingen)

	set_trace()
	Sample(X_train,Y_train,0,model=model)
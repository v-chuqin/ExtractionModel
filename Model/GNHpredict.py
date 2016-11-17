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
import os
import h5py

from ipdb import set_trace


os.environ['THEANO_FLAGS'] = 'device=cpu,floatX=float32'

maxlend=50 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512
rnn_layers = 3  # match FN1
batch_norm=False

activation_rnn_size = 40 if maxlend else 0

seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=64
nflips=10

empty = 0
eos = 1
nb_unknown_words = 1

FN1 = 'train_5000'

context_weight = K.variable(1.)
head_weight = K.variable(1.)
cross_weight = K.variable(0.)


embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding(embeddingfileName="embedding_vocab_5000.pkl",datafileName="heads_desc_5000.pkl")
vocab_size, embedding_size = embedding.shape
# set_trace()


def str_shape(x):
	return 'x'.join(map(str,x.shape))

def inspect_model(model):
	print model.name
	for i,l in enumerate(model.layers):
		print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
		weights = l.get_weights()
		for weight in weights:
			print str_shape(weight),
		print

def load_weights(model, filepath):
	print 'Loading', filepath, 'to', model.name
	flattened_layers = model.layers
	with h5py.File(filepath, mode='r') as f:
		layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
		weight_value_tuples = []
		for name in layer_names:
			print name
			g = f[name]
			weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
			if len(weight_names):
				weight_values = [g[weight_name] for weight_name in weight_names]
				try:
					layer = model.get_layer(name=name)
				except:
					layer = None
				if not layer:
					print 'failed to find layer', name, 'in model'
					print 'weights', ' '.join(str_shape(w) for w in weight_values)
					print 'stopping to load all other layers'
					weight_values = [np.array(w) for w in weight_values]
					break
				symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
				weight_value_tuples += zip(symbolic_weights, weight_values)
				weight_values = None
		K.batch_set_value(weight_value_tuples)
	return weight_values

def simple_context(X, mask, n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
	desc, head = X[:,:maxlend], X[:,maxlend:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
	activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
	assert mask.ndim == 2
	activation_energies = K.switch(mask[:, None, :maxlend], activation_energies, -1e20)
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))
	desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=([2],[1]))
	return K.concatenate((context_weight*desc_avg_word, head_weight*head_words))

class SimpleContext(Lambda):
	def __init__(self,**Kwargs):
		super(SimpleContext,self).__init__(simple_context,**Kwargs)
		self.supports_masking = True
	def compute_mask(self, input, input_mask=None):
		return input_mask[:, maxlend:]
	def get_output_shape_for(self, input_shape):
		nb_samples = input_shape[0]
		n = 2*(rnn_size - activation_rnn_size)
		return (nb_samples, maxlenh, n)

def output2probs(output,weights):
	output = np.dot(output, weights[0]) + weights[1]
	output -= output.max()
	output = np.exp(output)
	output /= output.sum()
	return output

def output2probs1(output,weights):
	output0 = np.dot(output[:n//2], weights[0][:n//2,:])
	output1 = np.dot(output[n//2:], weights[0][n//2:,:])
	output = output0 + output1 # + output0 * output1
	output += weights[1]
	output -= output.max()
	output = np.exp(output)
	output /= output.sum()
	return output

def lpadd(x,maxlend=maxlend,eos=eos):
	assert maxlend >= 0
	if maxlend == 0:
		return [eos]
	n = len(x)
	if n > maxlend:
		x = x[-maxlend:]
		n = maxlend
	return [empty]*(maxlend-n) + x + [eos]

def beamsearch(model,weights,predict,start=[empty]*maxlend + [eos], avoid=None, avoid_score=1,
				k=1, maxsample=maxlen, use_unk=True, oov=vocab_size-1, empty=empty, eos=eos, temperature=1.0):
	def sample(energy, n, temperature=temperature):
		n = min(n,len(energy))
		prb =np.exp(-np.array(energy)/temperature)
		res = []
		for i in xrange(n):
			z = np.sum(prb)
			r = np.argmax(np.random.multinomial(1, prb/z, 1))
			res.append(r)
			prb[r] = 0.
		return res
	dead_samples = []
	dead_scores = []
	live_samples = [list(start)]
	live_scores = [0]

	while live_samples:
		probs = predict(model,weights,live_samples)
		assert vocab_size == probs.shape[1]
		cand_scores = np.array(live_scores)[:,None] - np.log(probs)
		cand_scores[:,empty] = 1e20
		if not use_unk and oov is not None:
			cand_scores[:,oov] = 1e20
		if avoid:
			for a in avoid:
				for i,s , in enumerate(live_samples):
					n = len(s) - len(start)
					if n < len(a):
						cand_scores[i,a[n]] += avoid_score
		live_scores = list(cand_scores.flatten())
		scores = dead_scores + live_scores
		ranks = sample(scores,k)
		n = len(dead_scores)
		dead_scores = [dead_scores[r] for r in ranks if r < n]
		dead_samples = [dead_samples[r] for r in ranks if r < n]
		live_scores = [live_scores[r-n] for r in ranks if r >= n]
		live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]
		def is_zombie(s):
			return s[-1] == eos or len(s) > maxsample
		dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
		dead_samples += [s for s in live_samples if is_zombie(s)]
		live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
		live_samples = [s for s in live_samples if not is_zombie(s)]
	return dead_samples, dead_scores

def keras_rnn_predict(model,weights,samples,empty=empty,maxlen=maxlen):
	sample_lengths = map(len,samples)
	assert all(l > maxlend for l in sample_lengths)
	assert all(l[maxlend]==eos for l in samples)
	data = sequence.pad_sequences(samples,maxlen=maxlen,value=empty,padding='post',truncating='post')
	# set_trace()
	probs = model.predict(data,verbose=0,batch_size=batch_size)
	return np.array([output2probs(prob[sample_length-maxlend-1],weights) for prob, sample_length in zip(probs, sample_lengths)])

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

def gensamples(model,weights,X=None,X_test=None,Y_test=None,avoid=None,avoid_score=1,skips=2,k=10,batch_size=batch_size,short=True,temperature=1.,use_unk=True):
	if X is None or isinstance(X,int):
		if X is None:
			i = random.randint(0,len(X_test)-1)
		else:
			i = X
		print 'HEAD %d:'%i,' '.join(idx2word[w] for w in Y_test[i])
		print 'DESC:',' '.join(idx2word[w] for w in X_test[i])
		sys.stdout.flush()
		x = X_test[i]
	else:
		x = [word2idx[w.rstrip('^')] for w in X.split()]
	if avoid:
		if isinstance(avoid,str) or isinstance(avoid[0], int):
			avoid = [avoid]
		avoid = [a.split() if isinstance(a,str) else a for a in avoid]
		avoid = [vocab_fold([w if isinstance(w,int) else word2idx[w] for w in a]) for a in avoid]
	# set_trace()
	print 'HEADS:'
	samples = []
	if maxlend == 0:
		skips = [0]
	else:
		skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
		if skips == []:
			skips = [maxlend]
	# set_trace()
	for s in skips:
		start = lpadd(x[:s])
		fold_start = vocab_fold(start)
		sample, score = beamsearch(model=model,weights=weights,predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
								k=k, temperature=temperature, use_unk=use_unk)
		# set_trace()
		assert all(s[maxlend] == eos for s in sample)
		samples += [(s,start,scr) for s,scr in zip(sample,score)]
		# set_trace()

	samples.sort(key=lambda x: x[-1])
	codes = []
	for sample, start, score in samples:
		code = ''
		words = []
		sample = vocab_unfold(start,sample)[len(start):]
		for w in sample:
			if w==eos:
				break
			words.append(idx2word[w])
			code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
		if short:
			distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
			if distance > -0.6:
				print score, ' '.join(words)
		else:
				print score,' '.join(words)
		codes.append(code)
	return samples

def wsimple_context(X,mask,n=activation_rnn_size,maxlend=maxlend,maxlenh=maxlenh):
	desc,head = X[:,:maxlend], X[:,maxlend:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
	activation_energies = K.batch_dot(head_activations, desc_activations, axes=([2],[2]))
	assert mask.ndim == 2
	activation_energies = K.switch(mask[:,None,:maxlend], activation_energies, -1e20)
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))
	return activation_weights

class WSimpleContext(Lambda):
	def __init__(self):
		super(WSimpleContext, self).__init__(wsimple_context)
		self.supports_masking = True
	def compute_mask(self, input, input_mask=None):
		return input_mask[:, maxlend:]
	def get_output_shape_for(self, input_shape):
		nb_samples = input_shape[0]
		n = 2*(rnn_size - activation_rnn_size)
		return (nb_samples, maxlenh, n)



if __name__ == "__main__":
	random.seed(seed)
	np.random.seed(seed)
	regularizer = l2(weight_decay) if weight_decay else None

	rnn_model = Sequential()
	rnn_model.add(Embedding(vocab_size,embedding_size,input_length=maxlen,
							W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,name='embedding_1'))
	for i in range(rnn_layers):
		lstm = LSTM(rnn_size,return_sequences=True, # batch_norm=batch_norm,
					W_regularizer=regularizer,U_regularizer=regularizer,
					b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
					name='lstm_%d'%(i+1))
		rnn_model.add(lstm)
		rnn_model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))

	weights = load_weights(rnn_model, 'data/%s.hdf5'%FN1)

	model = Sequential()
	model.add(rnn_model)
	if activation_rnn_size:
		model.add(SimpleContext(name='simplecontext_1'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')

	wmodel = Sequential()
	wmodel.add(rnn_model)
	wmodel.add(WSimpleContext())
	wmodel.compile(loss='categorical_crossentropy', optimizer=optimizer)

	seed = 8
	random.seed(seed)
	np.random.seed(seed)
	context_weight.set_value(np.float32(1.))
	head_weight.set_value(np.float32(1.))

	# X = "sexual violence against female protestors at cairo 's tahrir square appears to be increasing , an activist group devoted to end such assaults said tuesday <eos> operation anti-sexual^ harassment / assault said it intervened in 15 of 19 reported sexual assaults on friday when 10,000 people gathered in the square on the two - year anniversary of the start of the revolution that ousted president hosni mubarak <eos> \" these attacks represent a startling escalation of violence against women in tahrir square in terms of the number of incidents and the extremity^ of the violence which took place , \" the group said in a news release <eos> the organization said some of the cases involved life - threatening violence where the attackers used knives or other weapons <eos> it said some of its members also were attacked during rescue attempts <eos> cairo0 official warns of state 's collapse as protesters defy curfew "	
	# samples = gensamples(model,weights,X, skips=2, batch_size=batch_size, k=10, temperature=1.)
	# X = "the agreement reached by president obama and congress to reopen the government and avoid default was a punt a punt that traveled 15 yards it was n't a budget agreement because nothing in it addressed any budgetary issues nor was it a spending agreement because spending continues at its current"
	gensamples(model,weights,0,X_test=X_train,Y_test=Y_train,skips=1, batch_size=batch_size, k=10, temperature=1, use_unk=True, short=False)
	set_trace()

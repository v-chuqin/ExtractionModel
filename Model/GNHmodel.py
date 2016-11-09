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

from ipdb import set_trace

FN = 'train_5000'

maxlend=50 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen
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
# vocab_size = 0

# nb_train_samples = 30000
# nb_val_samples = 3000

embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding(embeddingfileName="embedding_vocab_5000.pkl",datafileName="heads_desc_5000.pkl")
vocab_size, embedding_size = embedding.shape

def simple_context(X,mask,n=activation_rnn_size,maxlend=maxlend,maxlenh=maxlenh):
	desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
	activation_energies = K.batch_dot(head_activations,desc_activations,axes=(2,2))
	activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))
	desc_avg_word = K.batch_dot(activation_weights,desc_words,axes=(2,1))
	return K.concatenate((desc_avg_word,head_words))

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
	x = [vocab_fold(lpadd(xd)+xh) for xd,xh in zip(xds,xhs)]
	x = sequence.pad_sequences(x,maxlen=maxlen,value=empty,padding='post',truncating='post')
	x = flip_headline(x,nflips=nflips,model=model,debug=debug)

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
			s = random.randint(min(maxlend,len(xd)),max(maxlend,len(xd)))
			xds.append(xd[:s])
			xh = Xh[t]
			s = random.randint(min(maxlenh,len(xh)),max(maxlenh,len(xh)))
			xhs.append(xh[:s])
		c += 1
		random.seed(new_seed)

		yield conv_seq_labels(xds,xhs,nflips=nflips, model=model,debug=debug)

def test_gen(gen, n=5):
    Xtr,Ytr = next(gen)
    for i in range(n):
        assert Xtr[i,maxlend] == eos
        x = Xtr[i,:maxlend]
        y = Xtr[i,maxlend:]
        yy = Ytr[i,:]
        yy = np.where(yy)[1]
        prt('L',yy,idx2word)
        prt('H',y,idx2word)
        if maxlend:
            prt('D',x,idx2word)

def str_shape(x):
	return 'x'.join(map(str,x.shape))

def inspect_model(model):
	for i,l in enumerate(model.layers):
		print i, 'cls=%s name=%s'%(type(l).__name__, l.name)
		weights = l.get_weights()
		for weight in weights:
			print str_shape(weight),
		print 

def lpadd(x,maxlend=maxlend,eos=eos):
	assert maxlend >= 0
	if maxlend == 0:
		return [eos]
	n = len(x)
	if n > maxlend:
		x = x[-maxlend:]
		n = maxlend
	return [empty]*(maxlend-n) + x + [eos]

def beamsearch(model,predict,start=[empty]*maxlend+[eos],k=1,maxsample=maxlen,use_unk=True,empty=empty,eos=eos,temperature=1.0):
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
	dead_k = 0
	dead_samples = []
	dead_scores = []
	live_k = 1
	live_samples = [list(start)]
	live_scores = [0]

	while live_k:
		# set_trace()
		probs = predict(model=model,samples=live_samples,empty=empty)
		cand_scores = np.array(live_scores)[:,None] - np.log(probs)
		cand_scores[:,empty] = 1e20
		if not use_unk:
			for i in range(nb_unknown_words):
				cand_scores[:,vocab_size-1-i] = 1e20
		live_scores = list(cand_scores.flatten())
		scores = dead_scores+live_scores
		ranks = sample(scores,k)
		n= len(dead_scores)
		ranks_dead = [r for r in ranks if r < n]
		ranks_live = [r-n for r in ranks if r >= n]
		dead_scores = [dead_scores[r] for r in ranks_dead]
		dead_samples = [dead_samples[r] for r in ranks_dead]
		live_scores = [live_scores[r] for r in ranks_live]
		voc_size = probs.shape[1]
		live_samples = [live_samples[r//voc_size]+[r%voc_size] for r in ranks_live]
		zombie = [s[-1] == eos or len(s)>maxsample for s in live_samples]
		dead_samples += [s for s,z in zip(live_samples,zombie) if z]
		dead_scores += [s for s,z in zip(live_scores,zombie) if z]
		dead_k = len(dead_samples)
		live_samples = [s for s,z in zip(live_samples,zombie) if not z]
		live_scores = [s for s,z in zip(live_scores,zombie) if not z]
		live_k = len(live_samples)
		# print "live_K = "+str(live_k)

	return dead_samples + live_samples, dead_scores + live_scores

def keras_rnn_predict(model,samples,empty=empty,maxlen=maxlen):
	sample_lengths = map(len,samples)
	assert all(l > maxlend for l in sample_lengths)
	assert all(l[maxlend]==eos for l in samples)
	data = sequence.pad_sequences(samples,maxlen=maxlen,value=empty,padding='post',truncating='post')
	# set_trace()
	probs = model.predict(data,verbose=0,batch_size=batch_size)
	return np.array([prob[sample_length-maxlend-1] for prob,sample_length in zip(probs, sample_lengths)])

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

def gensamples(model,skips=2, k=10, batch_size=batch_size, short=True, temperature=1.0, use_unk=True):
	i = random.randint(0,len(X_test)-1)
	print 'HEAD:',' '.join(idx2word[w] for w in Y_test[i][:maxlenh])
	print 'DESC:',' '.join(idx2word[w] for w in X_test[i][:maxlend])
	sys.stdout.flush()

	print 'HEADS:'
	x = X_test[i]
	samples = []
	if maxlend == 0:
		skips = [0]
	else:
		skips = range(min(maxlend,len(x)),max(maxlend,len(x)),abs(maxlend-len(x))//skips+1)
	for s in skips:
		start = lpadd(x[:s])
		fold_start = vocab_fold(start)
		# print "s = "+str(s)
		# set_trace()
		sample, score = beamsearch(model=model,predict=keras_rnn_predict,start=fold_start,k=k,temperature=temperature,use_unk=use_unk)
		assert all(s[maxlend]==eos for s in sample)
		samples += [(s,start,scr) for s,scr in zip(sample,score)]
		# set_trace()

	samples.sort(key=lambda x:x[-1])
	codes = []
	for sample, start, score in samples:
		code = ''
		words = []
		sample = vocab_unfold(start, sample)[len(start):]
		for w in sample:
			if w == eos:
				break
			words.append(idx2word[w])
			code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
		if short:
			distance = min([100]+[-Levenshtein.jaro(code,c) for c in codes])
			if distance > -0.6:
				print score, ' '.join(words)
			print score, ' '.join(words)
		else:
				print score,' '.join(words)
		codes.append(code)




if __name__ == "__main__":
	
	# embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding()
	# vocab_size, embedding_size = embedding.shape
	random.seed(seed)
	np.random.seed(seed)
	regularizer = l2(weight_decay) if weight_decay else None
	
	model = Sequential()
	model.add(Embedding(vocab_size, embedding_size,input_length=maxlen,
						W_regularizer=regularizer, dropout=p_emb, weights=[embedding], mask_zero=True,name='embedding_1'))
	for i in range(rnn_layers):
		lstm = LSTM(rnn_size,return_sequences=True,# batch_norm=batch_norm,
					W_regularizer=regularizer,U_regularizer=regularizer,
					b_regularizer=regularizer,dropout_W=p_W,dropout_U=p_U,name='lstm_%d'%(i+1))
		model.add(lstm)
		model.add(Dropout(p_dense,name='dropout_%d'%(i+1)))
	if activation_rnn_size:
		model.add(SimpleContext(name='simplecontext_1'))
	model.add(TimeDistributed(Dense(vocab_size,W_regularizer=regularizer,b_regularizer=regularizer,name='timeDistributed_1')))
	model.add(Activation('softmax',name='activation_1'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	K.set_value(model.optimizer.lr,np.float32(LR))
	inspect_model(model)

	try:
		model.load_weights('data/%s.hdf5'%FN)
	except:
		print "first time training"

	# gensamples(skips=2,batch_size=batch_size,k=10,temperature=1.0,model=model)
	
	# r = next(gen(X_train, Y_train, batch_size=batch_size))
	# set_trace()
	try:
		with open('data/%s.history.pkl'%FN, 'rb') as fp:
			history = pickle.load(fp)
	except:
		history = {}
	traingen = gen(X_train, Y_train, batch_size=batch_size, nflips=nflips, model=model)
	valgen = gen(X_test, Y_test, nb_batches=len(X_test)//batch_size, batch_size=batch_size)
	r = next(traingen)

	for iteration in range(30):
		print 'Iteration', iteration
		h = model.fit_generator(traingen, samples_per_epoch=len(X_train),
			nb_epoch=1, validation_data=valgen, nb_val_samples=len(X_test))
		for k,v in h.history.iteritems():
			history[k] = history.get(k,[]) + v
		with open('data/%s.history.pkl'%FN,'wb') as fp:
			pickle.dump(history,fp,-1)
		model.save_weights('data/%s.hdf5'%FN, overwrite=True)
		gensamples(model=model,batch_size=batch_size)







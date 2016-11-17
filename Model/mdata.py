from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from collections import Counter
from ipdb import set_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import cPickle as pickle
from sklearn.cross_validation import train_test_split
import os

seed=42
maxlend = 50 

import subprocess

def file_len(fname):
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def generateheadlinedata(inputfile,word2vecmodel,vocabMaxsize,eosflag=False,stopword=False,Sample=-1,maxlend=None,embedding_function='word2vec',embeddingfileName="embedding_vocab.pkl",datafileName="heads_desc.pkl"):
	desc, headline = getdatalist(inputfile,eosflag,stopword,Sample=Sample)
	vocab, vocabcount = buildvocabulary(desc+headline,maxlend=maxlend)
	word2idx, idx2word = get_idx(vocab, vocabcount)
	if embedding_function == 'word2vec':
		embedding,wordvec_idx2idx = get_embedding(vocab,word2vecmodel,vocabMaxsize,word2idx,idx2word)
	elif embedding_function == 'glove':
		embedding,wordvec_idx2idx = get_embedding_glove(vocab,word2vecmodel,vocabMaxsize,word2idx,idx2word)

	# elif embedding_function == 'word2vecGoolge':
	# 	embedding,wordvec_idx2idx = get_embedding(vocab,word2vecmodel,vocabMaxsize,word2idx,idx2word)
	else:
		print 'get_embedding error'
	# set_trace()
	X = [[word2idx[token] for token in d.split()[:maxlend]] for d in desc]
	Y = [[word2idx[token] for token in head.split()] for head in headline]
	plt.hist(map(len,Y),bins=50)
	plt.savefig('Y.png')
	plt.close()
	plt.hist(map(len,X),bins=50)
	plt.savefig('X.png')
	with open(embeddingfileName,'wb') as fp:
		pickle.dump((embedding, idx2word, word2idx, wordvec_idx2idx),fp,-1)
	with open(datafileName,'wb') as fp:
		pickle.dump((X,Y),fp,-1)
	# return X,Y,word2idx,idx2word,embedding,wordvec_idx2idx

def getdatalist(inputfile,eosflag=False,stopword=False,Sample=-1):
	print Sample
	desc = []
	headline = []
	file = open('../Data/'+inputfile)
	count = -1
	while 1:
		line = file.readline()
		if not line:
			break
		set_trace
		count = count + 1
		# set_trace()
		if count == Sample:
			break
		tmp = line.split('\t')
		desc_tmp = tmp[1]
		headline_tmp = tmp[3][:-2]
		if eosflag == True:
			desc_tmp = desc_tmp.replace('<eos>','')
			headline_tmp = headline_tmp.replace('<eos>','')
		if stopword == True:
			filein = open('stopwords.txt')
			if desc_tmp[:2] == '\' ':
				desc_tmp = desc_tmp[2:]
			if headline_tmp[:2] == '\' ':
				headline_tmp = headline_tmp[2:]
			while 1:
				line = filein.readline()
				if not line:
					break
				if line[:-2] == '':
					break
				desc_tmp = desc_tmp.replace(line[:-2],' ')
				headline_tmp = headline_tmp.replace(line[:-2],' ')
			filein.close()
		# set_trace()
		desc.append(desc_tmp)
		headline.append(headline_tmp)
	file.close()
	print "Success get all text data"
	return desc,headline

def buildvocabulary(lst,maxlend=None):
	vocabcount = Counter(w for txt in lst for w in txt.split()[:maxlend])
	vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
	print "Success build vocab"
	return vocab, vocabcount

def get_idx(vocab, vocabcount):
	empty = 0
	end = 1
	# start_idx = eos + 1
	start_idx = end + 1
	word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
	word2idx['<empty>'] = empty
	word2idx['<end>'] = end
	# eos = word2idx['<eos>']
	idx2word = dict((idx,word) for word,idx in word2idx.iteritems())
	print "Success link word and index"
	return word2idx, idx2word

def get_embedding_glove(vocab,word2vecmodel,vocab_size,word2idx,idx2word,nb_unknown_words=1):
	embedding_dim = 100
	glove_name = '../Data/'+word2vecmodel
	glove_n_symbols = file_len('../Data/'+word2vecmodel)
	# set_trace()
	glove_index_dict = {}
	glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
	globale_scale=.1
	glove_index_dict = {}
	glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
	globale_scale=.1
	with open(glove_name, 'r') as fp:
		i = 0
		for l in fp:
			l = l.strip().split()
			w = l[0]
			glove_index_dict[w] = i
			glove_embedding_weights[i,:] = map(float,l[1:])
			i += 1
	glove_embedding_weights *= globale_scale
	for w,i in glove_index_dict.iteritems():
		w = w.lower()
		if w not in glove_index_dict:
			glove_index_dict[w] = i
	np.random.seed(seed)
	shape = (vocab_size, embedding_dim)
	scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
	embedding = np.random.uniform(low=-scale, high=scale, size=shape)
	print 'random-embedding/glove scale', scale, 'std', embedding.std()
	# set_trace()
	c = 0
	for i in range(vocab_size):
		if i in idx2word:
			w = idx2word[i]
			g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
			if g is None and w.startswith('#'): # glove has no hastags (I think...)
				w = w[1:]
				g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
			if g is not None:
				embedding[i,:] = glove_embedding_weights[g,:]
				c+=1
	print 'number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size)
	glove_thr = 0.5
	word2glove = {}
	for w in word2idx:
		if w in glove_index_dict:
			g = w
		elif w.lower() in glove_index_dict:
			g = w.lower()
		elif w.startswith('#') and w[1:] in glove_index_dict:
			g = w[1:]
		elif w.startswith('#') and w[1:].lower() in glove_index_dict:
			g = w[1:].lower()
		else:
			continue
		word2glove[w] = g
	# set_trace()
	normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]
	nb_unknown_words = nb_unknown_words
	glove_match = []
	count = 0
	print "total have "+str(len(word2idx))+" to cal"
	for w,idx in word2idx.iteritems():
		if count % 100 == 0:
			sys.stdout.write('\r')
			sys.stdout.write("[%s>%s] %s" % ('-'*int(count/float(len(word2idx))*100), ' '*(100-int(count/float(len(word2idx))*100)),str(round((count+100)/float(len(word2idx))*100,2))+'%'))
			sys.stdout.flush()
		count += 1

		if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
			gidx = glove_index_dict[word2glove[w]]
			gweight = glove_embedding_weights[gidx,:].copy()
			gweight /= np.sqrt(np.dot(gweight,gweight))
			score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
			while True:
				embedding_idx = score.argmax()
				s = score[embedding_idx]
				if s < glove_thr:
					break
				if idx2word[embedding_idx] in word2glove :
					glove_match.append((w, embedding_idx, s)) 
					break
				score[embedding_idx] = -1
	glove_match.sort(key = lambda x: -x[2])
	print '# of glove substitutes found', len(glove_match)
	for orig, sub, score in glove_match[-100:]:
		print score, orig,'=>', idx2word[sub]
	glove_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in glove_match)
	return embedding,glove_idx2idx

def get_embedding(vocab,word2vecmodel,vocabMaxsize,word2idx,idx2word,nb_unknown_words=1):
	model = Word2Vec.load('../Data/'+word2vecmodel)
	word2vec_size = len(model.vocab)
	word2vec_dim = len(model['is'])
	embedding_weights = np.empty((word2vec_size,word2vec_dim))
	global_scale = .1
	i = 0
	word_list = model.vocab.keys()
	for word in word_list:
		embedding_weights[i,:] = model[word]
		i += 1
	embedding_weights *= global_scale
	np.random.seed(seed)
	shape = (vocabMaxsize,word2vec_dim)
	scale = embedding_weights.std()*np.sqrt(12)/2
	embedding = np.random.uniform(low=-scale,high=scale,size=shape)
	print 'random-embedding/glove scale', scale, 'std', embedding.std()
	c = 0
	for i in range(vocabMaxsize):
		if i in idx2word:
			w = idx2word[i].decode('utf-8')
			if w in model:
				embedding[i,:] = model[w]*global_scale
				c+=1
	print 'number of tokens, in small vocab, found in word2vec and copied to embedding', c,c/float(vocabMaxsize)
	word2vec_thr = 0.5
	normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]
	word2vec_match = []
	nb_unknown_words = nb_unknown_words
	count = 0
	print "total have "+str(len(word2idx))+" to cal"
	for _w,idx in word2idx.iteritems():
		if count % 100 == 0:
			sys.stdout.write('\r')
			sys.stdout.write("[%s>%s] %s" % ('-'*int(count/float(len(word2idx))*100), ' '*(100-int(count/float(len(word2idx))*100)),str(round((count+100)/float(len(word2idx))*100,2))+'%'))
			sys.stdout.flush()
		count += 1
		# if count % 10000 == 0:
		# 	print count
		w = _w.decode('utf-8')
		if idx >= vocabMaxsize-nb_unknown_words and w in model:
			# if idx == 109406:
			# 	set_trace()
			gweight = model[w].copy()
			gweight *= global_scale
			gweight /= np.sqrt(np.dot(gweight,gweight))
			score = np.dot(normed_embedding[:vocabMaxsize-nb_unknown_words],gweight)
			while 1:
				embedding_idx = score.argmax()
				s = score[embedding_idx]
			 	if s < word2vec_thr:
			 		break
			 	word_decode = idx2word[embedding_idx].decode('utf-8')
			 	if word_decode in model:
			 		word2vec_match.append((_w,embedding_idx,s))
			 		break
			 	score[embedding_idx] = -1
	word2vec_match.sort(key = lambda x: -x[2])
	print '\n# of glove substitutes found', len(word2vec_match)
	# for orig, sub, score in word2vec_match[:100]:
	# 	print score, orig,'=>', idx2word[sub]
	wordvec_idx2idx = dict((word2idx[w],embedding_idx) for  w, embedding_idx, _ in word2vec_match)
	return embedding,wordvec_idx2idx

def loadembedding(embeddingfileName="embedding_vocab.pkl",datafileName="heads_desc.pkl",test_size=0.1,nb_unknown_words=1):
	with open(embeddingfileName, 'rb') as fp:
		embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
	with open(datafileName, 'rb') as fp:
		X, Y = pickle.load(fp)
	vocab_size, embedding_size = embedding.shape
	print 'number of examples',len(X),len(Y)
	print 'dimension of embedding space for words',embedding_size
	# print 'vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words
	print 'total number of different words',len(idx2word), len(word2idx)
	print 'number of words outside vocabulary which we can substitue using glove similarity', len(glove_idx2idx)
	print 'number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(glove_idx2idx)

	oov0 = vocab_size - nb_unknown_words
	for i in range(oov0,len(idx2word)):
		idx2word[i] = idx2word[i]+'^'
	# set_trace()
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)

	print 'train file: '+str(len(X_train))+' test file: '+str(len(X_test))
	# set_trace()
	empty = 0
	end = 1
	idx2word[empty] = '<empty>'
	idx2word[end] = '<end>'
	return embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test

def prt(label,x,idx2word):
	print label + ":"
	for w in x:
		print idx2word[w],
	print 

'''

def generatetestdata(inputfile,word2vecmodel,SampleNum):
	model = Word2Vec.load('../Data/'+word2vecmodel)
	file = open('../Data/'+inputfile)
	Sentences = []
	keywords = []
	simpleWeight = []
	maxlen = 32
	while 1:
		line = file.readline()[:-2]
		if not line or SampleNum == 0:
			break
		tmp_list = line.split()
		tmpkeyword = []
		tmpsentence = []
		tmpsimpleWeight = []
		for i in range(0,len(tmp_list)):
			try:
				tmpsentence.append(model[tmp_list[i]])
			except:
				continue
			tmpkeyword.append(np.asarray([1], dtype=np.float32))
			tmpsimpleWeight.append(np.asarray([1], dtype=np.float32))
			if i == (maxlen-2):
				break
		SampleNum = SampleNum - 1
		Sentences.append(tmpsentence)
		keywords.append(tmpkeyword)
		simpleWeight.append(tmpsimpleWeight)
	print "maxlen is "+str(maxlen)
	word_dim = len(Sentences[0][0])
	print "word_dim is "+str(word_dim)
	mask_array = []
	for i in range(0,word_dim):
		mask_array.append(0.0)
	mask_array_ = np.asarray(mask_array, dtype=np.float32)
	for i in range(0,len(Sentences)):
		for j in range(len(Sentences[i]),maxlen):
			Sentences[i].append(mask_array_)
			keywords[i].append(np.asarray([0], dtype=np.float32))
			simpleWeight[i].append(np.asarray([0], dtype=np.float32))
		keywords[i] = np.asarray(keywords[i], dtype=np.float32)
		simpleWeight[i] = np.asarray(simpleWeight[i], dtype=np.float32)
		# set_trace()
	Sentences = np.asarray(Sentences, dtype=np.float32)
	keywords = np.asarray(keywords, dtype=np.float32)
	simpleWeight = np.asarray(simpleWeight, dtype=np.float32)
	file.close()
	# set_trace()
	return Sentences,keywords,simpleWeight

def loaddata(word2vecmodel,sentencefile,keywordfile):
	# for testing output
	# Sentences,keywords = loaddata('tweets.threshold_6.txt.model','tweets.threshold_6.txt','tweetskey.threshold_6.txt')
	# for i in range(0,len(Sentences)):
	# 	if len(Sentences[i]) != len(keywords[i]):
	# 		print "error at "+str(i)
	model = Word2Vec.load('../Data/'+word2vecmodel)
	file_sentence = open('../Data/'+sentencefile)
	file_keyword = open('../Data/'+keywordfile)
	Sentences = []
	keywords = []
	simpleWeight = []
	maxlen = 0
	while 1:
		line = file_sentence.readline()[:-2]
		line_keyword = file_keyword.readline()
		if not line or not line_keyword:
			break
		tmp_list = line.split()
		tmpkeyword_list = line_keyword.split()
		tmpkeyword = []
		tmpsentence = []
		tmpsimpleWeight = []
		for i in range(0,len(tmp_list)):
			# set_trace()
			try:
				tmpsentence.append(model[tmp_list[i]])
			except:
				continue
			if tmp_list[i] in tmpkeyword_list:
				tmpkeyword.append(np.asarray([1], dtype=np.float32))
			else:
				tmpkeyword.append(np.asarray([0], dtype=np.float32))
			tmpsimpleWeight.append(np.asarray([1], dtype=np.float32))
		Sentences.append(tmpsentence)
		keywords.append(tmpkeyword)
		simpleWeight.append(tmpsimpleWeight)
		if len(tmpsentence) != len(tmpkeyword):
			return False
		if len(tmpkeyword) > maxlen:
			maxlen = len(tmpkeyword)
	print "maxlen is "+str(maxlen)
	word_dim = len(Sentences[0][0])
	print "word_dim is "+str(word_dim)
	mask_array = []
	for i in range(0,word_dim):
		mask_array.append(0.0)
	mask_array_ = np.asarray(mask_array, dtype=np.float32)
	# set_trace()
	for i in range(0,len(Sentences)):
		for j in range(len(Sentences[i]),maxlen):
			Sentences[i].append(mask_array_)
			keywords[i].append(np.asarray([0], dtype=np.float32))
			simpleWeight[i].append(np.asarray([0], dtype=np.float32))
		keywords[i] = np.asarray(keywords[i], dtype=np.float32)
		simpleWeight[i] = np.asarray(simpleWeight[i], dtype=np.float32)
		# set_trace()
	Sentences = np.asarray(Sentences, dtype=np.float32)
	keywords = np.asarray(keywords, dtype=np.float32)
	simpleWeight = np.asarray(simpleWeight, dtype=np.float32)
	# set_trace()
	return Sentences,keywords,simpleWeight

def calprecision(TopN,outputData,predicted_output):
	sum = 0.0
	for i in range(0,len(outputData)):
		realList = []
		predictList = []
		predictList_value = []
		for j in range(outputData[i]):
			realList.append(outputData[i][j][0])
			predictList.append(predicted_output[i][j][0])
			predictList_value.append(0)
		rank_list = sorted(range(len(predictList)), key=lambda k: predictList[k])
		TopN_tmp = TopN
		for j in range(0,len(TopN)):
			if predictList[rank_list[j]] == 0:
				TopN_tmp = j + 1
				break
			predictList_value[rank_list[j]] = 1
		predictList_array = np.asarray(predictList_value, dtype=np.intc)
		realList_array = np.asarray(realList_array, dtype=np.intc)
		sum = sum + realList_array*predictList_array/TopN_tmp
	return sum/len(outputData)


	# plt.plot([vocabcount[w] for w in vocab])
	# plt.gca().set_xscale("log", nonposx='clip')
	# plt.gca().set_yscale("log", nonposy='clip')
	# plt.title('word distribution in headlines and discription')
	# plt.xlabel('rank')
	# plt.ylabel('total appearances')
	# plt.savefig('vocab.png')

'''

if __name__ == "__main__":

	generateheadlinedata('cnnCombined_clean.txt.utf-8','cnnCropus_clean.txt.utf-8.model',8000,eosflag=True,stopword=True,Sample=1024,maxlend=None,embedding_function='word2vec',embeddingfileName="embedding_vocab_1000.pkl",datafileName="heads_desc_1000.pkl")
	# loadembedding()
	# embedding, idx2word, word2idx, glove_idx2idx, X_train, X_test, Y_train, Y_test = loadembedding(embeddingfileName="embedding_vocab_5000.pkl",datafileName="heads_desc_5000.pkl")

	# desc, headline = getdatalist('cnnCombined_clean.txt.utf-8',True,True,2)
	# vocab, vocabcount = buildvocabulary(desc[:maxlend]+headline)
	# word2idx, idx2word = get_idx(vocab, vocabcount)
	# embedding,wordvec_idx2idx = get_embedding_glove(vocab,'glove.6B.100d.txt',20000,word2idx,idx2word)

	set_trace()

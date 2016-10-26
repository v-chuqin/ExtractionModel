from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
from collections import Counter
from ipdb import set_trace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generateheadlinedata(inputfile,word2vecmodel):
	desc, headline = getdatalist(inputfile)
	vocab, vocabcount = buildvocabulary(desc+headline)
	word2idx, idx2word = get_idx(vocab, vocabcount)
	embedding = get_embedding(vocab,word2vecmodel)
	X = [[word2idx[token] for token in d.split()] for d in desc]
	Y = [[word2idx[token] for token in head.split()] for head in headline]
	plt.hist(map(len,Y),bins=50)
	plt.savefig('Y.png')
	plt.close()
	plt.hist(map(len,X),bins=50)
	plt.savefig('X.png')

def getdatalist(inputfile):
	desc = []
	headline = []
	file = open('../Data/'+inputfile)
	while 1:
		line = file.readline()
		if not line:
			break
		tmp = line.split('\t')
		desc.append(tmp[1])
		headline.append(tmp[3])
	file.close()
	return desc,headline

def buildvocabulary(lst):
	vocabcount = Counter(w for txt in lst for w in txt.split())
	vocab = map(lambda x: x[0], sorted(vocabcount.items(), key=lambda x: -x[1]))
	return vocab, vocabcount

def get_idx(vocab, vocabcount):
	eos = 1
	start_idx = eos + 1
	word2idx = dict((word, idx+start_idx) for idx,word in enumerate(vocab))
	word2idx['<eos>'] = eos
	idx2word = dict((idx,word) for word,idx in word2idx.iteritems())
	return word2idx, idx2word

def get_embedding(vocab,word2vecmodel):
	model = Word2Vec.load('../Data/'+word2vecmodel)
	outofvocab = []
	embedding = {}
	for word in vocab:
		try:
			embedding[word] = model[word]
		except:
			outofvocab.append(word)
	set_trace()
	return embedding

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

if __name__ == "__main__":
	# desc, headline = getdatalist('cnnCombined.txt.utf-8')
	# vocab, vocabcount = buildvocabulary(desc+headline)
	# word2idx, idx2word = get_idx(vocab, vocabcount)
	# embedding = get_embedding(vocab,'cnnCropus.txt.utf-8.model')

	generateheadlinedata('cnnCombined.txt.utf-8','cnnCropus.txt.utf-8.model')
	
	set_trace()

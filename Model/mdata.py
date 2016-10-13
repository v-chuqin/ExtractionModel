from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np

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
		line = file_sentence.readline()
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
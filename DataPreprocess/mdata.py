import os
from ipdb import set_trace
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

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
	while 1:
		line = file_sentence.readline()
		line_keyword = file_keyword.readline()
		if not line or not line_keyword:
			break
		tmp_list = line.split()
		tmpkeyword_list = line_keyword.split()
		tmpkeyword = []
		tmpsentence = []
		for i in range(0,len(tmp_list)):
			# set_trace()
			try:
				tmpsentence.append(model[tmp_list[i]])
			except:
				continue
			if tmp_list[i] in tmpkeyword_list:
				tmpkeyword.append(1)
			else:
				tmpkeyword.append(0)
		Sentences.append(tmpsentence)
		keywords.append(tmpkeyword)
	return Sentences,keywords


def loadStopwords():
	Stopwords = []
	file = open('stopwords.txt')
	while 1:
		line = file.readline()
		if not line:
			break
		Stopwords.append(line[:-2])
	return Stopwords

def cleanEnglishString(input,Stopwords):
	tmp_list = input.split()
	output = ''
	for i in range(0,len(tmp_list)):
		for stopword in Stopwords:
			# Stop words model
			if len(tmp_list[i].split(stopword)) > 1 or tmp_list[i] == '\'' or tmp_list[i] == '-':
				tmp_list[i] = ''
				break
	for word in tmp_list:
		if word != '':
			output = output+' '+word
	return output[1:]

def handleTweet(type,threshold):
	# tweet.train.tok.en-n.key/txt/CrowdCountskey
	# train: 1000, dev: 327, test: 500
	Stopwords = loadStopwords()
	# set_trace()
	Tweets = []
	Keywords = []
	fileCount = 0
	if type == 'train':
		fileCount = 1000
	elif type == 'test':
		fileCount = 500
	elif type  == 'dev':
		fileCount = 320
	for fileIndex in range(1,fileCount+1):
		file = open(os.getcwd()+'/../Data/dataset-lmarujo-ACL2015/'+type+'/tweet.'+type+'.tok.en-'+str(fileIndex)+'.txt')
		line = file.readline()
		cleanline = cleanEnglishString(line,Stopwords)
		print str(fileIndex) +' : ' + cleanline.lower()
		# set_trace()
		tmp_list = cleanline.lower().split(' ')
		if len(tmp_list) == 1:
			continue
		Tweets.append(tmp_list)
		file.close()
		file = open(os.getcwd()+'/../Data/dataset-lmarujo-ACL2015/'+type+'/tweet.'+type+'.tok.en-'+str(fileIndex)+'-CrowdCountskey')
		words_tmp = []
		while 1:
			line = file.readline()[:-1]
			if not line:
				break
			_tmp_list = line.split('\t')
			word = _tmp_list[0]
			value = int(_tmp_list[1])
			if value >= threshold and word in tmp_list:
				words_tmp.append(word)
		print words_tmp
		# if len(words_tmp) >= threshold:
		Keywords.append(words_tmp)
		# Tweets.append(tmp_list)
	return Tweets,Keywords

def getTweetfile(threshold):
	# file = open(os.getcwd()+'/../Data/tweets.txt','w')
	file = open(os.getcwd()+'/../Data/tweets.threshold_'+str(threshold)+'.txt','w')
	filekey = open(os.getcwd()+'/../Data/tweetskey.threshold_'+str(threshold)+'.txt','w')
	Tweets,Keywords = handleTweet('train',threshold)
	for i in range(0,len(Tweets)):
		line = ''
		for j in range(0,len(Tweets[i])):
			line = line + ' ' + Tweets[i][j]
		file.write(line[1:]+'\n')
	for i in range(0,len(Keywords)):
		line = ''
		for j in range(0,len(Keywords[i])):
			line = line + ' ' + Keywords[i][j]
		filekey.write(line[1:]+'\n')

	Tweets,Keywords = handleTweet('test',threshold)
	for i in range(0,len(Tweets)):
		line = ''
		for j in range(0,len(Tweets[i])):
			line = line + ' ' + Tweets[i][j]
		file.write(line[1:]+'\n')
	for i in range(0,len(Keywords)):
		line = ''
		for j in range(0,len(Keywords[i])):
			line = line + ' ' + Keywords[i][j]
		filekey.write(line[1:]+'\n')

	Tweets,Keywords = handleTweet('dev',threshold)
	for i in range(0,len(Tweets)):
		line = ''
		for j in range(0,len(Tweets[i])):
			line = line + ' ' + Tweets[i][j]
		file.write(line[1:]+'\n')
	for i in range(0,len(Keywords)):
		line = ''
		for j in range(0,len(Keywords[i])):
			line = line + ' ' + Keywords[i][j]
		filekey.write(line[1:]+'\n')

	file.close()
	filekey.close()

def generateTweetfile(threshold,type):
	file = open(os.getcwd()+'/../Data/tweets.'+type+'.threshold_'+str(threshold)+'.txt','w')
	Tweets,Keywords = handleTweet(type,threshold)
	for i in range(0,len(Tweets)):
		line = ''
		for j in range(0,len(Tweets[i])):
			line = line + ' ' + Tweets[i][j]
		file.write(line[1:]+'\n')
	file.close()

# getTweetfile(7)
getTweetfile(6)
# generateTweetfile(6,'train')
# generateTweetfile(6,'test')


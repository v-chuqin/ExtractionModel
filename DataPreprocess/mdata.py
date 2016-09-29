import os
from ipdb import set_trace

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
		fileCount = 327
	for fileIndex in range(1,fileCount+1):
		file = open(os.getcwd()+'/../Data/dataset-lmarujo-ACL2015/'+type+'/tweet.'+type+'.tok.en-'+str(fileIndex)+'.txt')
		line = file.readline()
		cleanline = cleanEnglishString(line,Stopwords)
		print str(fileIndex) +' : ' + cleanline
		# set_trace()
		tmp_list = cleanline.split(' ')
		if len(tmp_list) == 0:
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
		Keywords.append(words_tmp)
		print words_tmp
	return Tweets,Keywords





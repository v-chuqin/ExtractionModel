from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from ipdb import set_trace
from multiprocessing import cpu_count
from copy import deepcopy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# waiting for the gensim udpate, support the new vocabulary to the pretrain model
def retrainWord2vec(pretrainModel,trainfile):
	trainsentences = LineSentence('../Data/'+trainfile)
	premodel = Word2Vec.load('../Data/'+pretrainModel)
	oldmodel = deepcopy(premodel)
	premodel.min_count = 0
	premodel.build_vocab(trainsentences, update=True)
	premodel.train(trainsentences)
	premodel.save('../Data/'+pretrainModel.split('.')[0]+'_'+trainfile+'.model')
	for m in ['oldmodel', 'premodel']:
		print 'The vocabulary size of the' +m+ 'is ' + str(len(eval(m).vocab)) 
	return premodel

def trainWord2vec(trainfile):
	sentences = LineSentence('../Data/'+trainfile)
	model = Word2Vec(sentences,size=200,window=4,min_count=5,workers=cpu_count())
	model.save('../Data/'+trainfile+ '.model')
	model.save_word2vec_format('../Data/'+trainfile+ '.model.bin', binary=True)
	return model

def test(trainfile,testfile):
	trainsentences = LineSentence('../Data/'+trainfile)
	testsentences = LineSentence('../Data/'+testfile)
	model = Word2Vec(trainsentences,size=50,window=5,min_count=3,workers=cpu_count())
	# model.save('oldmodel')
	# model = Word2Vec.load('oldmodel')
	oldmodel = deepcopy(model)
	oldmodel.save('oldmodel')
	model.min_count = 0
	model.build_vocab(testsentences, update=True)
	model.train(testsentences)
	model.save('newmodel')
	for m in ['oldmodel', 'model']:
		print 'The vocabulary size of the' +m+ 'is ' + str(len(eval(m).vocab)) 
	return model


# model = trainWord2vec('tweets.threshold_6.txt')
# model = retrainWord2vec('GoogleNews-vectors-negative300.bin','tweets.threshold.txt')
# model = test('tweets.train.threshold_6.txt','tweets.test.threshold_6.txt')
model = trainWord2vec('tweetsTest_clean.tsv')
# model = retrainWord2vec('tweetsTest_clean.tsv.model','tweets.threshold_6.txt')
set_trace()


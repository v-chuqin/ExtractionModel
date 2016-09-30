from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from ipdb import set_trace

def trainWord2vec(pretrainModel,trainfile):
	model = Word2Vec.load_word2vec_format('../embedding/pretrain/'+pretrainModel,binary=True)
	sentences = LineSentence('../Data/'+trainfile)

def trainWord2vec(trainfile):
	sentences = LineSentence('../Data/'+trainfile)
	model = Word2Vec(sentences,size=50,window=5,min_count=1,workers=4)
	model.save_word2vec_format('../Data/'+trainfile+ '.model.bin', binary=True)
	return model


model = trainWord2vec('tweets.threshold.txt')
set_trace()




from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from ipdb import set_trace
from multiprocessing import cpu_count


# waiting for the gensim udpate, support the new vocabulary to the pretrain model
def retrainWord2vec(pretrainModel,trainfile):
	premodel = Word2Vec.load_word2vec_format(pretrainModel,binary=True)
	sentences = LineSentence('../Data/'+trainfile)
	set_trace()
	# model.update_vocab(new_sentences)
	premodel.build_vocab(sentences, update=True)
	premodel.train(sentences,min_count=0,workers=cpu_count())
	# model = premodel.train(sentences,size=50,window=5,min_count=1,workers=cpu_count())
	premodel.save_word2vec_format('../Data/'+trainfile+ '.retrian.model.bin', binary=True)
	return premodel

def trainWord2vec(trainfile):
	sentences = LineSentence('../Data/'+trainfile)
	model = Word2Vec(sentences,size=50,window=5,min_count=1,workers=cpu_count())
	model.save_word2vec_format('../Data/'+trainfile+ '.model.bin', binary=True)
	return model

def test(trainfile):
	sentences = LineSentence('../Data/'+trainfile)
	model = Word2Vec(sentences,size=50,window=5,min_count=0,workers=cpu_count())
	model.build_vocab(sentences, update=True)
	model.train(sentences,min_count=0,workers=cpu_count())
	return model


model = trainWord2vec('tweets.threshold.txt')
# model = retrainWord2vec('GoogleNews-vectors-negative300.bin','tweets.threshold.txt')
# model = test('tweets.threshold.txt')
set_trace()


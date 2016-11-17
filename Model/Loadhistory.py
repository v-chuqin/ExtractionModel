import cPickle as pickle
from ipdb import set_trace

with open('data/train.history.pkl', 'rb') as fp:
	x = pickle.load(fp)
set_trace()

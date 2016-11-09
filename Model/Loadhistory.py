import cPickle as pickle
from ipdb import set_trace

with open('data/train_5000.history.pkl', 'rb') as fp:
	x = pickle.load(fp)
set_trace()
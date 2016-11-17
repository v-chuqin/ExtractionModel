from seq2seq import SimpleSeq2Seq, Seq2Seq, AttentionSeq2Seq
import numpy as np
from keras.utils.test_utils import keras_test
from ipdb import set_trace
from keras.models import Sequential

input_length = 10
input_dim = 2

output_length = 8
output_dim = 3

samples = 100


# def test_SimpleSeq2Seq():
# 	x = np.random.random((samples, input_length, input_dim))
# 	y = np.random.random((samples, output_length, output_dim))

# 	models = []
# 	models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
# 	models += [SimpleSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]

# 	for model in models:
# 		model.compile(loss='mse', optimizer='sgd')
# 		model.fit(x, y, nb_epoch=1)


# def test_Seq2Seq():
# 	x = np.random.random((samples, input_length, input_dim))
# 	y = np.random.random((samples, output_length, output_dim))

# 	models = []
# 	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
# 	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True)]
# 	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
# 	models += [Seq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), peek=True, depth=2)]

# 	for model in models:
# 		model.compile(loss='mse', optimizer='sgd')
# 		model.fit(x, y, nb_epoch=1)


def test_AttentionSeq2Seq():
	x = np.random.random((samples, input_length, input_dim))
	y = np.random.random((samples, output_length, output_dim))

	model = Sequential()
	model.add(AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim)))
	model.compile(loss='mse', optimizer='sgd')
	model.fit(x, y, nb_epoch=1)
	set_trace()

	# models = []
	# models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim))]
	# models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=2)]
	# models += [AttentionSeq2Seq(output_dim=output_dim, output_length=output_length, input_shape=(input_length, input_dim), depth=3)]

	# for model in models:
	# 	model.compile(loss='mse', optimizer='sgd')
	# 	set_trace()
	# 	model.fit(x, y, nb_epoch=1)

test_AttentionSeq2Seq()
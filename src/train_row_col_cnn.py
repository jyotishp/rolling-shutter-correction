import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from keras.models import Sequential, Model
from keras.callbacks import TensorBoard
from keras.optimizers import *
from keras.layers import *
from generator import *
from models import *
from keras.utils import multi_gpu_model

# Set Tensorflow backend to avoid full GPU pre-loading
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.global_variables_initializer()
set_session(tf.Session(config = config))

if __name__ == '__main__':
	print('Loading Generator')
	gen = Generator(batch_size=128)
	print('Loading model')
	#  Use model = rowColCNN() to train on single GPU or CPU
	model = multi_gpu_model(rowColCNN(), gpus=3)

	# Tensorboard callback
	tb_callback = TensorBoard(log_dir='./logs/no-pad-bn-sgd')
	optimizer = SGD(lr=0.005, momentum=0.0, decay=0.0, nesterov=False)
	model.compile(loss='mean_squared_error', optimizer=optimizer)

	model.fit_generator(
		generator = gen.train(),
		epochs = 60,
		steps_per_epoch = int(gen.total_files *0.8 / gen.batch_size),
		verbose = 1,
		callbacks = [tb_callback],
		validation_data = gen.validate(),
		validation_steps = int(gen.total_files*0.2 / gen.batch_size),
		initial_epoch = 0
	)
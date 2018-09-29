#encoding:utf-8
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
import operator
import os
from keras.models import Model
from keras.layers import Dense, Input, Reshape, BatchNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adam
from keras import backend as K






# 将图片转化为灰度图
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gen_data(wavpath):
	path2id = {'train/aa':0, 'train/bb':1, 'train/cc':2, 'train/dd':3, 
				'test/aa':0, 'test/bb':1, 'test/cc':2, 'test/dd':3}
	picdata = []
	label = []
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.jpg'):
				filepath = os.sep.join([dirpath, filename])
				lena = mpimg.imread(filepath)
				gray = rgb2gray(lena)
				picdata.append(gray)
				label.append(path2id[dirpath])
	picdata = np.array(picdata)
	label = np.array(label)
	return picdata, label





def creatModel():
	input_data = Input(name='the_input', shape=(560, 560, 1))
	layer_h1 = MaxPooling2D(pool_size=(5,5), strides=None, padding="valid")(input_data)
	# 112,112,16
	layer_h1 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
	layer_h1 = BatchNormalization(mode=0,axis=-1)(layer_h1)
	layer_h2 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h1)
	layer_h2 = BatchNormalization(axis=-1)(layer_h2)
	layer_h3 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h2)
	# 56,56,16
	layer_h4 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h3)
	layer_h4 = BatchNormalization(axis=-1)(layer_h4)
	layer_h5 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h4)
	layer_h5 = BatchNormalization(axis=-1)(layer_h5)
	layer_h5 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h5)
	# 28,28,16
	layer_h6 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h5)
	layer_h6 = BatchNormalization(axis=-1)(layer_h6)
	layer_h7 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h6)
	layer_h7 = BatchNormalization(axis=-1)(layer_h7)
	layer_h7 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h7)
	# 14,14,32
	layer_h8 = Conv2D(32, (1,1), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h7)
	layer_h8 = BatchNormalization(axis=-1)(layer_h8)
	layer_h9 = Conv2D(32, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h8)
	layer_h9 = BatchNormalization(axis=-1)(layer_h9)
	layer_h9 = MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(layer_h9)
	# 7,7,16
	layer_h10 = Conv2D(16, (1,1), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h9)
	layer_h10 = BatchNormalization(axis=-1)(layer_h10)
	layer_h11 = Conv2D(16, (3,3), use_bias=True, activation='relu', padding='same', kernel_initializer='he_normal')(layer_h10)
	layer_h11 = BatchNormalization(axis=-1)(layer_h11)
	# Reshape层
	layer_h12 = Reshape((1, 784))(layer_h11) 
	# 全连接层
	layer_h13 = Dense(256, activation="relu", use_bias=True, kernel_initializer='he_normal')(layer_h12)
	layer_h13 = BatchNormalization(axis=1)(layer_h13)
	layer_h14 = Dense(4, use_bias=True, kernel_initializer='he_normal')(layer_h13)
	output = Activation('softmax', name='Activation0')(layer_h14)
	model = Model(inputs=input_data, outputs=output)
	# clipnorm seems to speeds up convergence
	#sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
	#ada_d = Adadelta(lr = 0.01, rho = 0.95, epsilon = 1e-06)
	#rms = RMSprop(lr=0.01,rho=0.9,epsilon=1e-06)		
	opt = Adam(lr = 0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0, epsilon = 10e-8)
	#ada_d = Adadelta(lr=0.01, rho=0.95, epsilon=1e-06)
	#model=multi_gpu_model(model,gpus=2)
	model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
	#test_func = K.function([input_data], [output])
	print("model compiled successful!")
	return model

id2word = ['横','竖','撇','捺']

def train():
	X, y = gen_data('train/')
	X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
	one_hot_y = np.zeros((len(y), 4))
	for i,j in enumerate(y):
		one_hot_y[i][j] = 1
	one_hot_y = one_hot_y.reshape(320, 1, 4)
	model = creatModel()
	model.fit(X,one_hot_y,epochs=4)
	model.save_weights('model.md')



def test():
	X, y = gen_data('test/')
	X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
	model = creatModel()
	model.load_weights('model.md')
	predict = model.predict(X)
	result = np.vstack(predict).argmax(axis=1)
	print('label:\n', [id2word[i] for i in y])
	print('predict:\n',[id2word[i] for i in result])


if __name__ == '__main__':
	runtype = 'test'
	#runtype = input('input run type:')
	if runtype == 'train':
		train()
	else:
		test()
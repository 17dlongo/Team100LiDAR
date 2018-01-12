import sys
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import numpy as np
sys.path.append("./lidarInterpret.py")
from lidarInterpret import interpretCSV 


import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from keras.models import Model, load_model, Sequential
from keras.optimizers import Adam
from keras.layers import Input, Convolution1D, MaxPooling1D, Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard

import time
# to save and then restore a model
#model.save(filepath)
#load_model
#del model 
#h5 is the extension


#creation of batches
X,y = interpretCSV('LiDARData')

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')
X_train = X[10:]
y_train = y[10:]
X_test = X[:10]
y_test = X[:10]

def model(): #for Id Arrays
	img = Input(shape=(100,3), name='img')
	x = Convolution1D(8, 3)(img)
	x = Activation('relu')(x)
	x = MaxPooling1D(pool_size=2)(x)
	#Convolution/Pooling Layer 2
	x = Convolution1D(16, 3)(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(pool_size=2)(x)
	#Convolution/Po1ling Layer 3
	x = Convolution1D(32, 3)(x)
	x = Activation('relu')(x)
	x = MaxPooling1D(pool_size=2)(x)
	#Flattens into 1D array for Usage in final layer
	merged = Flatten()(x)
	#One final fully connected layer for figuring output values
	x = Dense(128)(merged)
	x = Activation('linear')(x)
	x = Dropout(.3)(x)
	#Final output
	jstk = Dense(1, name='jstk')(x)
	#Compiled and initializes the model
	steerNet = Model(input=[img], output=[jstk])
	steerNet.compile(optimizer='adam', loss='mean_squared_error')
	print(steerNet.summary())
	return steerNet

model().fit(x=X_train,
	y=y_train,
	batch_size=4, 
	epochs=30000, 
	verbose=2, 
	callbacks=[TensorBoard(log_dir="logs/train/save_"+str(int(time.time()))+"", 
	histogram_freq=1,
	write_graph=True,
	write_images=True)],
	validation_split=0.2,
	shuffle=True,
	initial_epoch=0)

# def model():
# 	#Model with 3 hidden layers
# 	#Input takes in image
# 	img = Input(shape = (142,100,3), name = 'img')
# 	#Convolution/Pooling Layer 1
# 	x = Convolution2D(8, 3, 3)(img)
# 	x = Activation('relu')(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	#Convolution/Pooling Layer 2
# 	x = Convolution2D(16, 3, 3)(x)
# 	x = Activation('relu')(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	#Convolution/Pooling Layer 3
# 	x = Convolution2D(32, 3, 3)(x)
# 	x = Activation('relu')(x)
# 	x = MaxPooling2D(pool_size=(2, 2))(x)
# 	#Flattens into 1D array for Usage in final layer
# 	merged = Flatten()(x)
# 	#One final fully connected layer for figuring output values
# 	x = Dense(128)(merged)
# 	x = Activation('linear')(x)
# 	x = Dropout(.3)(x)
# 	#Final output
# 	jstk = Dense(1, name='jstk')(x)
# 	#Compiled and initializes the model
# 	steerNet = Model(input=[img], output=[jstk])
# 	steerNet.compile(optimizer='adam', loss='mean_squared_error')
# 	print(steerNet.summary())
# 	return steerNet


# model.fit(x=X_train, y=y_train, batch_size=4, epochs=100, verbose=2, callbacks=None, validation_split=0.2, shuffle=True, initial_epoch=0)

# print(np.shape(X),np.shape(X_train))

# model = Sequential([
#	 Dense(32, input_shape=(100,3)),
#	 Activation('relu'),
#	 Flatten(),
#	 Dense(1,142),
#	 Activation('softmax'),
# ])

# model.add(Dense(32, input_shape=(100,3), acitavation='relu'))
# Dense(10),
# Activation('softmax'),

# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(100,3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))


# model.compile(optimizer='rmsprop',
# 	  loss='mse')

# model.fit(X_train,y_train, epochs=10, batch_size=4)

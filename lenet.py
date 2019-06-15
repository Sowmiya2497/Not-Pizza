from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:
	@staticmethod
	def build(height,width,depth,classes):
		model = Sequential()
		inputShape = (height,width,depth)
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)

		#Add first set of layers

		#Default stride = (1,1). Always include input_shape argument if it is the first layer in CNN. 
		model.add(Conv2D(20,(5,5),padding='same',input_shape=inputShape))
		
		model.add(Activation('relu'))
		
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		#Add second set of layers
		model.add(Conv2D(50,(5,5),padding='same'))

		model.add(Activation('relu'))
		
		model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

		#Flatten the preceding maxpool layer to a single vector

		model.add(Flatten())

		
		model.add(Dense(500))
		model.add(Activation('relu'))

		model.add(Dense(classes))
		model.add(Activation('softmax'))
		return model



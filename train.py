import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

#Include paths in command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d','--dataset',required=True,help='path to input dataset')
ap.add_argument('-m','--model',required=True,help='path to model')
args = vars(ap.parse_args())

#Initialize essential parameters
epochs = 300
learning_rate = 1e-3
bsize = 32

data = []
labels = []

#Load paths to images and shuffle them
imagePaths = sorted(list(paths.list_images(args['dataset'])))#Works recursively to find all image files with valid extensions in current directory
random.seed(42)
random.shuffle(imagePaths)

#Preprocessing images
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(28,28))
	image = img_to_array(image)
	data.append(image)

	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == 'pizza' else 0
	labels.append(label)

#Scale to range (0,1)
data = np.array(data,dtype = 'float32')/255.0
labels = np.array(labels)

#Create train-test split
trainX,testX,trainY,testY = train_test_split(data,labels,test_size=0.25,random_state = 42)

#Convert categorical data to one-hot encoding
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print('Compiling model....')
model = LeNet.build(height=28,width=28,depth=3,classes=2)
opt = Adam(lr=learning_rate,decay=learning_rate/epochs)

#Configure the model parameters before training
#loss parameter is the objective function. Optimizer is the gradient descent algorithm. metrics is accuracy for classification problem.
#it is used to evaluate the performance of the model at the end of training an epoch(not taken into consideration while training to change params-that is the loss function's job)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])


#Train the network
#the image augmenter aug applies random transformations to original data (within the specified range) for each EPOCH and aug.flow() spits out a batch of the
#transformed data. Thus, the data is transformed to generate "new" data for as many epochs. 
#Mini-batch gradient descent is implemented, meaning that the average error for the samples
#in one batch is calculated and gradients are updated at the end of each batch. batch_size defaults to 32
#So,you step through each batch, and at the end of the epoch, the model with current params is evaluated against validation(test set) for the loss 
H = model.fit_generator(aug.flow(trainX,trainY,batch_size = bsize),validation_data=(testX,testY),steps_per_epoch=len(trainX)//bsize,epochs=epochs,verbose=1)

print("serializing network...")
model.save(args["model"])


















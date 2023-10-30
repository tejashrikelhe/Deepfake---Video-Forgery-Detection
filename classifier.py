# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:22:11 2020

@author: TEJASHRI
"""

#importing the libraries
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

#NOTE: While using Keras the 1st number inside a function specifies how many neurons are there for the current layer.

classifier= Sequential() # Initialise the CNN

# Ist step of Convoltional layer to get feature maps using feature detector
# Conv2d----keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
#                               dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform',
#                               bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
#                               kernel_constraint=None, bias_constraint=None)
# 1st hidden layer
classifier.add(Convolution2D(filters=32, # output feature maps #the op of d 1st conv is d ip 4 applying d kernal 2nd time 2 d image and this will happen 32 times since filter=32  
                             kernel_size=(3,3), # matrix size for feature detector i.e filter size
                             input_shape=(64, 64, 3), # input image shape, 3 is for rgb coloured image with 128*128 px
                             kernel_initializer='he_uniform', # uniform weights distribution #intialises d random matrix eles of d kernel matrix by multiplying them with some no  
                             activation='relu')) # activation function # Rectified Linear Unit
#  Pooling layer
# Pooling is a hidden layer
classifier.add(MaxPooling2D(pool_size=(2,2))) # To collect prominent features .i.e. Max pixel value from the final convulution output matrix by convolving this matrix with a matrix of size (2*2) in our case.  

#2nd convolutional and pooling layer.
classifier.add(Convolution2D(filters=32,
                             kernel_size=(3,3), 
                             kernel_initializer='he_uniform', 
                             activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
# it converts d 3D op (as RGB so 3 layers) from d above layers to 1D op i.e. a single vector since this layer is gonna b fully connected with d next layer which does d predictions and which will only take a 1D layer as an ip . 

classifier.add(Flatten())

#Step 4 full connection in which input we have from flattening

classifier.add(Dense(units=128,kernel_initializer='glorot_uniform', activation='relu')) # Dense= full connection with d next layer .i.e fully/ densly connected.

#step 5 output layer

classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid')) # units= number of nodes in that layer i.e. the neurons i.e the circles

# Compiling the CNN
#This will create a Python object which will build the CNN with the parameters you've given. This is done by building the computation graph in the correct format based on the Keras backend you are using.

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) #For a classification task categorical cross-entropy works very well.


#applying all the transformation we want to apply to training data set
# ImageDataGenerator trains the model in such a way that the orientation of the images do not matter.Its used 2 artificially increacse d variations in our dataset. 
# ImageDataGenerator is a class. Thus, train_datagen is an object of this class.
train_datagen = ImageDataGenerator(rescale = 1./255, # v r taking RGB image whose coeff are in d range of 0 to 255 and is 2 high 2 process 4 our model thus v divide d matrix by 255 to get d values between 1 n 0.  
                                   shear_range = 0.2, #it slants d image
                                   zoom_range = 0.2, #it zooms d img
                                   horizontal_flip = True) #flips it horizontally

#Rescling the test data set images to use for validation.
test_datagen= ImageDataGenerator(rescale=1./255)

#Getting My training data ready for validation, so it will read all the data with the pixel size we gave.
training_set= train_datagen.flow_from_directory(directory= 'dataset/training_set',
                                               target_size=(64,64), # As we choose 64*64 for our convolution model
                                               batch_size=50,       # Num of imgs 2 b yielded from d generator per batch.  
                                               class_mode='binary' # for 2 class binary 
                                               )

#Getting My test data ready for validation, so it will read all the data with the px size we gave.
test_set= test_datagen.flow_from_directory(directory= 'dataset/test_set',
                                               target_size=(64,64), # As we choose 64*64 for our convolution model
                                               batch_size=50,
                                               class_mode='binary' # for 2 class binary
                                          )
classifier.fit_generator(training_set, #training data to fit
                        steps_per_epoch=200, # Data in training set 
                        epochs=1, # No of epochs to run # epoch= the number of times d dataset will b trained.
                        validation_data=test_set, # Test or validation set
                        validation_steps=100 # no of data point for validation
                        )

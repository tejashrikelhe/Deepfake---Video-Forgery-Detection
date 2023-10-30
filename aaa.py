# -*- coding: utf-8 -*-
"""
Created on Fri May  1 18:37:59 2020

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
import numpy as np
from keras.preprocessing import image

classifier= Sequential() # Initialise the CNN
# Ist step of Convoltional layer to get feature maps using feature detector
classifier.add(Convolution2D(filters=32, # output feature maps
                             kernel_size=(3,3), # matrix size for feature detector
                             input_shape=(64, 64, 3), # input image shape, 3 is for rgb coloured image with 128*128 px
                             kernel_initializer='he_uniform', # weights distriution
                             activation='relu')) # activation function
# 2nd Pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))
#2nd convolutional and pooling layer.
classifier.add(Convolution2D(filters=32,
                             kernel_size=(3,3), 
                             kernel_initializer='he_uniform', 
                             activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
# Step 3 - Flattening
classifier.add(Flatten())
#Step 4 full connection in which input we have from flattening

classifier.add(Dense(units=128,kernel_initializer='glorot_uniform', activation='relu')) 
#step 5 output layer
classifier.add(Dense(units=1,kernel_initializer='glorot_uniform',activation='sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#applying all the transformation we want to apply to training data set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
#Rescling the test data set images to use for validation.
test_datagen= ImageDataGenerator(rescale=1./255)
#Getting My training data ready for validation, so it will read all the data with the px size we gave.

training_set= train_datagen.flow_from_directory(directory= 'dataset/training_set',
                                               target_size=(64,64), # As we choose 64*64 for our convolution model
                                               batch_size=50,
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
                        epochs=1, # No of epochs to run
                        validation_data=test_set, # Test or validation set
                        validation_steps=100 # no of data point for validation
                        )

# Part 3 - Making new predictions
test_image = image.load_img('C:\Users\TEJASHRI\Desktop\projs\videoForg\dataset\single_prediction/fake_or_real.jpg', target_size = (64, 64))
# Loading the image and converting the pixels into array whcih will be used as input to predict.
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'real'
else:
    prediction = 'fake'
print(prediction)
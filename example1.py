# -*- coding: utf-8 -*-
"""
Created on Sat May  2 23:22:42 2020

@author: TEJASHRI
"""

import os
import glob
import classifier
from classifier import *
import keras
from keras.preprocessing import image
import numpy as np
import cv2


vc = cv2.VideoCapture(r'dataset\test_video\Ariana Grande Pete Davidson Deepfake.mp4')  
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd) 

c = 1


if vc.isOpened(): 
    rval, frame = vc.read()
    print("Read Successfully")

else:
    rval = False

# timeF = 2  
count = 0
name_no = 0
os.chdir(r"C:\Users\TEJASHRI\Desktop\projs\videoForg\dogs-cats-images\dataset\test_video") #C:\Users\TEJASHRI\Desktop\projs\videoForg\dataset\code
  
# varify the path using getcwd() 
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd) 
while rval:  
    rval, frame = vc.read()
    count += 1
    
    #img_dir = "dataset/test_video"
    if count >= 0 & count < 500:
        if (count % 500 == 0): 
            name_no += 1
            # frame = frame[100:380, :]
            #frame = frame[100:350, :]
            cv2.imwrite("image"+str(count)+".jpg", frame)
            print('saving image')
            cv2.waitKey(1)
#vc.release()    
os.chdir(r"C:/Users/TEJASHRI/Desktop/projs/videoForg/dogs-cats-images") #C:\Users\TEJASHRI\Desktop\projs\videoForg   C:\Users\TEJASHRI\Desktop\projs\videoForg\dogs-cats-images
  
# varify the path using getcwd() 
cwd = os.getcwd() 
  
# print the current directory 
print("Current working directory is:", cwd) 

# Part 3 - Making new predictions
img_dir = "dataset/test_video" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g') #joins d path of img_dir with all the path ending with 'g' 
files = glob.glob(data_path) # it finds all d path links matching a specified pattern, it arranges all of them in a list. 
for f1 in files:
    test_image = image.load_img(f1, target_size = (64, 64))
    # Loading the image and converting the pixels into array whcih will be used as input to predict.
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0) # expand_dims adds dimensions, it inserts a new axis at a specified position. thus converts it into a matrix.
    result = classifier.predict(test_image)
    training_set.class_indices      # this attribute is used to see d dictionary containing the mapping from class name to class indices. V used to understand what 0 and 1 stand for.  
    if result[0][0] == 1:
        prediction = 'The video is REAL'
    else:
        prediction = 'The video is a DeepFake i.e. FAKE'
        break
print(prediction)
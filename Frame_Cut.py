# -*- coding: utf-8 -*-
"""
Created on Fri May  1 19:27:02 2020

@author: TEJASHRI
"""

import cv2

vc = cv2.VideoCapture('Ariana Grande Pete Davidson Deepfake.mp4')
c = 1

if vc.isOpened():  
    rval, frame = vc.read()
    print("Read Successfully")

else:
    rval = False

# timeF = 2  
count = 0
name_no = 0
while rval:  
    rval, frame = vc.read()
    count += 1
    if count >= 0 & count < 500:
        if (count % 50 == 0):  
            name_no += 1
            # frame = frame[100:380, :]
            #frame = frame[100:350, :]
            cv2.imwrite("image"+str(count)+".jpg", frame)      
vc.release()
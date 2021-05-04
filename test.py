import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import time
header=st.beta_container()
project=st.beta_container()
proj2=st.beta_container()
count=0
p=0

	
with proj2:
	st.header('Gender Verification page')
	#st.text(label)
	my_placeholder = st.empty() 
	# load model
	model = load_model('gender_detection.model')
	
	# open webcam
	webcam = cv2.VideoCapture(0)
	    
	classes = ['man','woman']
	
	# loop through frames
	while webcam.isOpened():
	    
	    # read frame from webcam 
	    status, frame = webcam.read()
	   
	
	    # apply face detection
	    face, confidence = cv.detect_face(frame)
	
	
	    # loop through detected faces
	    for idx, f in enumerate(face):
	
	        # get corner points of face rectangle        
	        (startX, startY) = f[0], f[1]
	        (endX, endY) = f[2], f[3]
	
	        # draw rectangle over face
	        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
	
	        # crop the detected face region
	        face_crop = np.copy(frame[startY:endY,startX:endX])
	
	        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
	            continue
	
	        # preprocessing for gender detection model
	        face_crop = cv2.resize(face_crop, (96,96))
	        face_crop = face_crop.astype("float") / 255.0
	        face_crop = img_to_array(face_crop)
	        face_crop = np.expand_dims(face_crop, axis=0)
	
	        # apply gender detection on face
	        conf = model.predict(face_crop)[0] # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]
	
	        # get label with max accuracy
	        idx = np.argmax(conf)
	        label = classes[idx]
	        z=label
	        #disp(z)
	      
	
	        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
	
	        Y = startY - 10 if startY - 10 > 10 else startY + 10
	
	        # write label and confidence above face rectangle
	        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
	                    0.7, (0, 255, 0), 2)
	
	    # display output
	    
	    my_placeholder.image(frame,use_column_width=True)
	    if count != 1:
		    st.success(label)
		    count=1
		#else :
		#	pass
  


			

from __future__ import absolute_import, unicode_literals
from __future__ import print_function
import numpy as np
from scipy.misc import imsave, imread, imresize
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import cv2
import sys
from keras.models import load_model
import h5py

import tensorflow as tf
import shutil
import os.path

#%matplotlib inline
#import matplotlib.pyplot as plt
import string


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


model = load_model('model.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
mapping = pickle.load(open('mapping.p','rb'))

def predictint(roii):
   b=[]
   for pi in roii:
    try:
         #pi = cv2.resize(im1, (28, 28), interpolation=cv2.INTER_AREA)
         pi = imresize(roi,(28,28))
    except:
         continue
    #mydata = reformat(pi)
    mydata = pi.reshape(1,28,28,1)
    mydata = mydata.astype('float32')
    mydata /= 255
    #cv2.imshow('hello',mydata)
    #cv2.waitKey()
    #print('mydata size:' ,mydata.shape)
    #cv2.imshow('final',mydata)
    #cv2.waitKey()
    out = model.predict(mydata)
    # Generate response
    response = {'prediction': chr(mapping[(int(np.argmax(out, axis=1)[0]))]),'confidence': str(max(out[0]) * 100)[:6]}		
    #predictions = model.predict_classes(mydata)
    #print(response)
    #b.append(int(np.argmax(out, axis=1)[0]))
    #b.append(str(response['prediction']))
    b.append(response['prediction'])

   return(b)

# Read the input image 
#im = cv2.imread(sys.argv[1]
cap=cv2.VideoCapture(0)

while True:
    ret,im=cap.read()
    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(im_gray,70,255,cv2.THRESH_BINARY_INV) 
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilated = cv2.dilate(thresh,kernel,iterations = 0) 
    _,ctrs, hierarchy = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
    
    # Threshold the image
    #ret, im_th = cv2.threshold(im_gray, 90, 255,cv2.THRESH_BINARY_INV)
    
    # Find contours in the image
    #retimg,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (ctrs,boundbxs) = sort_contours(ctrs, method = 'left-to-right')
    
    # Get rectangles contains each contour
    #rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = boundbxs
    """for ctr in ctrs:
        area=cv2.contourArea(ctr)
        if area<576:
            continue
        rects.append(cv2.boundingRect(ctr))"""
    if rects is None:
        cv2.imshow("original",im)
        continue

    # For each rectangular region, calculate HOG features and predict
    #i = 0
    j=0
    #a=[]
    for rect in rects:
        # Draw the rectangles
        #cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = dilated[pt1:pt1+leng, pt2:pt2+leng]
        #try:
        # morp= cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT,kernel)
        #dilated = cv2.dilate(thresh,kernel,iterations = 0) 
        #cv2.imshow('Features',morp)
        #except:
        # continue
        #cv2.imshow('Features',morp)
        #cv2.imwrite(str(i)+'.jpg', roi) 
        #images.append(str(i)+'.jpg')
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        a=predictint(roi)
        #cv2.imshow('roi',roi)
        try:
            cv2.putText(im, str(a[j]), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)  
        except IndexError:
            continue
        #Calculate the HOG features
        cv2.imshow("Resulting Image with Rectangular ROIs", im)
        j=j+1
        #cv2.waitKey()
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
        #roi_final = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        #nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        #nbr=predictint(str(i)+'.png')
        #nbr = predictint('0.pn1g')
        #cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        #i=i+1


    """a=predictint(images)
    print("a: ",a)

    #letters = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","+","-","*","/","="]
    letters = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

    b=[]
    for i in a:
    #print("i", i)
    for z in range(36):
    if(i==z):
     b.append(letters[z])

    j=0
    for rect in rects: 
    #for contour in ctrs:
        try:
            cv2.putText(im, str((a[j])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 3)  
        except IndexError:
            continue
        j=j+1

    cv2.imshow("Resulting Image with Rectangular ROIs", im)
    cv2.waitKey()"""
    
cap.release()
# De-allocate any associated memory usage
cv2.destroyAllWindows()

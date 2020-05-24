import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import load_model
import time

# define filepath to saved model
filepath = "C:/186B/Models/EyeStatus_v1"

# load model
model = tf.keras.models.load_model(filepath)
#model = tf.saved_model.load(filepath)
#model = tf.lite.TFLiteConverter.from_saved_model(filepath)

# load the required trained XML classifiers 
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# redefine original array to be indexed by prediction later
CLASSES = ["Open","Closed"]

cap = cv2.VideoCapture(0) # initialize camera

# define resize dimensions
width = 100
height = 100
dim = (width,height)

# Initialize frame counter, blink threshold, number of blinks
consec_frame = 0
blink_thresh = 3
BLINKS = 0



###############################################################################
# Format sample taken from webcam
def getsample():

    ret, frame = cap.read() # grab frame from webcam
    
    # Detects faces of different sizes in the input image
    faces = detector.detectMultiScale(frame, 1.3, 5)
    
    # convert img to gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    for (x,y,w,h) in faces:

        # Find border coordinates to declare new image of only the face
        gray = gray[y:y+h,x:x+w]
        
        # Resize Image
        gray = cv2.resize(gray,dim)
        #write the face to a file
        #cv2.imwrite("C:/186B/test_getsample/test." + ".jpg",gray)
    #cv2.imshow('Frame',gray) # show sample image
    
    return gray.reshape(-1,width,height,1)

#################################################################################
# make predictions continuously
while True:
    
    sample = getsample() # call function to grab sample from webcam
    prediction = model.predict(sample) # make prediction on sample
    print(CLASSES[int(prediction[0][0])]) # index array to print prediction (string)
    #time.sleep(.001) # add delay if necessary
    sample = sample.reshape(height,width) # change shape to use cv2.imshow

    # Count number of consecutive closed frames 
    if int(prediction[0][0]) == 1:
        consec_frame += 1

    else:
        # if consecutive frames is greater than blinking threshold 
        if consec_frame >= blink_thresh:
            BLINKS += 1 # increment blink counter
        consec_frame = 0 # reset consecutive frames after blink
     
    if consec_frame > 8:
         sample = cv2.putText(sample, "***ALERT DRIVER***", (5, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
         cv2.imshow('Frame', sample)
         
         consec_frame = 0
            
    else:     
        sample = cv2.putText(sample, "Blinks: {}".format(BLINKS), (5, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
        cv2.imshow('Frame', sample)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
##################################################################################








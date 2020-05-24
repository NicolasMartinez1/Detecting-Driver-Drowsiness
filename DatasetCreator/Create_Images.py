import cv2
import numpy as np
  
# load the required trained XML classifiers 
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# set dimensions of photos
width = 100
height = 100
dim = (width,height)

cap = cv2.VideoCapture(0) # capture frames from cam

id = input('Enter User ID ') # Created ID for user
count = 0; # user count
 
# loop runs if capturing has been initialized. 
while (True):  
  
    # read from cam
    ret, img = cap.read()
     
    # convert to gray scale
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
  
    # Detects faces of different sizes in the input image 
    faces = detector.detectMultiScale(img, 1.3, 5)
    
    for (x,y,w,h) in faces:
        
        count = count + 1; #increment count

        # Find border coordinates to declare new image of only the face
        img = img[y:y+h,x:x+w]
        
        # Resize Image
        img = cv2.resize(img,dim)
        
        #write the face to a file
        cv2.imwrite("C:/186B/User_Images/Closed/User." + str(id) + "." + str(count)+ ".jpg",img)
        
        # draw a rectangle around face  
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)    
        cv2.waitKey(100); # pause 100ms
                
    cv2.imshow('img',img) # Display image in window
    
    # Wait for (count) to be reached
    cv2.waitKey(1);
    if(count == 100):
        break
       
# Close the window 
cap.release() 
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows()

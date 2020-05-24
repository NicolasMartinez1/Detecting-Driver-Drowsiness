import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

DataDir = "C:/186B/Datasets/Faces100"  # store data directory file path
CLASSES = ["Open", "Closed"]
width = 100
height = 100


####################################################################################################

# Create a function to append indeces to each photo in the dataset
def create_training_data(imgsize):
    training_data = [] # Create array for training images
    IMG_SIZE = imgsize
    
    for classes in CLASSES:
        path = os.path.join(DataDir, classes)  # Define path to data directory
        classID = CLASSES.index(classes) # assign index to each class (open = 0, closed = 1)
        
        for img in os.listdir(path):     # for each image in each class
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE)) # resize data if necessary (h x w)
                training_data.append([new_array, classID])
            except Exception as e:
                pass
            
                
    # call function to create appended dataset
    #create_training_data(100)
    print(len(training_data)) # show length of array (should be total number of images in database)

    random.shuffle(training_data) # shuffle array prior to feeding it to network or else NN may overfit

    # Check shuffled images
    #for n in training_data[:10]: # for first 20 elements of array
        #print(n[1])             # print classID (open = 0, closed = 1) n[0] = img array, n[1] = label array

    

    # Save data to file

    xtrain = []
    ytrain = []

    for features,label in training_data:
        xtrain.append(features)
        ytrain.append(label)

    xtrain = np.array(xtrain).reshape(-1, IMG_SIZE,IMG_SIZE,1) 
    ytrain = np.array(ytrain)

    pickle_out = open("xtrain.pickle","wb")
    pickle.dump(xtrain, pickle_out)
    pickle_out.close()

    pickle_out = open("ytrain.pickle","wb")
    pickle.dump(ytrain, pickle_out)
    pickle_out.close()


create_training_data(100)



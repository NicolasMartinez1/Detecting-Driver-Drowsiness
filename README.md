# Detecting-Driver-Drowsiness
California State University of Fresno, Electrical and Computer Engineering
Senior Design - 186B Final Project Spring 2020

Project Leader: Nicolas Martinez
Team Member:    Jose Baca
Team Member:    Amro Bahaddein
Technical Advisor and Course Instructor: Dr. Hovannes Kulhandjian

This project explores detecting driver drowsiness using Machine-Learning. The goal of this project is to use
a convolutional neural network (CNN) to classify whether a driver's eyes are open or closed. The predictions made
by the neural network model can be used to then determine if a driver is in fact drowsy or not. The blinking patterns
are extracted from the model's predictions and are used to alert the driver of their drowsy state; in which case the
driver may then pull over and call for assistance. 

DIRECTORIES:

DatasetCreator:
There are two python files in the DatasetCreator folder titled Create_Images and DataSetCreator.

  Create_Images - This python script is was used to add images of the team members faces to the database that was to
                  be used for training and testing the model. Once the program is started, it will ask for a "User ID",
                  which can be a number or name of the user. Once the ID is entered the program will take a specified
                  number of pictures from the USB webcam and add them to the database directory of the user's choice.
                  
  DataSetCreator - This program uses the previous database of images created, consisting of faces with both eye's closed
                   or open, and appends a binary label to each image, thereby classifying each image as open or closed.
                   The program then shuffles the image array so that both sets of images are mixed with one another. The
                   data is then written to a txt file as a numpy array and are saved to the Models folder where they will
                   be called by the "Model_1" program.

Models:
The Models folder contains the two previously created pickle files for the x and y training data, the trained model directories, the Model_1 python script used to train the model, and the mode_Results folder containing the results of the trained model within the "EyeStatus_v1" folder. 

 Model_1 - This python program is where the convolutional neural network was trained on the database previously created.
           The pickle files are opened to access the traning data needed to be fed to the neural network. The model characteristics
           returned are the traning and validation results for both the accuracy and loss. The training parameters were altered in 
           order to achieve more desireable results in terms of accuracy and loss. The model was then saved, which automatically
           creates a saved model directory labeled at the user's descretion. 
            
  EyeStatus_v1 - This folder is one of the saved model directories that is automatically created by the "Model_1" python script.
                 The files and folder within this folder are automatically generated will be used once the trained model is to be
                 called in the "predict" script in the "Main" folder.
                 
Main:
This folder consists of the "predict" python script and the haarcascade_frontalface_default file used to detect the user's face within an image.

  predict - This is the main python script that runs the neural network and makes predictions based on the samples taken
            from the USB webcam. First, the previously saved model and face detector are loaded and initialized. Next,
            the "getsample" function is created to take an image from the webcam, resize, grayscale, and reshape it before
            sending it to the model as an input. The "predict" function is then called on the sample image and the model
            returns its prediction in the form of a 0 or 1; which is then converted into a string labeled as "Open" or "Closed".
            From the predictions, the blinking pattern of the user is provided and the "if/then" statements are used to determine
            whether or not the user(driver) is drowsy. The number of blinks is written to the image that is shown to the screen after
            each prediction. 
            
  
  
  
  
  
  
  
  
  
            

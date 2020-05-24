# CNN Model 

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import time
import datetime
import matplotlib.pyplot as plt


# Just disables the warning, doesn't enable AVX/FMA
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TF_CONFIG_ = tf.compat.v1.ConfigProto()
TF_CONFIG_.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = TF_CONFIG_)


#NAME = "EyeStatus-cnn-64x2-{}".format(int(time.time()))
#tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
#sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))

pickle_in = open("xtrain.pickle","rb")
xtrain = pickle.load(pickle_in)

pickle_in = open("ytrain.pickle","rb")
ytrain = pickle.load(pickle_in)

xtrain = xtrain/255.0   # Normalize data, not as computationally expensive

model = Sequential()

# Hidden Layer 1
model.add(Conv2D(32,(3,3), input_shape = xtrain.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

# Hidden Layer 2
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
# Flatten data to 1D
model.add(Flatten())

#model.add(Dense(64))
#model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

# return model characteristics
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

# save model characterstics to plot data
history = model.fit(xtrain, ytrain,
          batch_size = 32,
          validation_split = 0.2,
          epochs = 15)

model.test_on_batch(xtrain, ytrain)
model.metrics_names

# print model data using history()
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.grid()
plt.show()

model.save("EyeStatus.model")

model = tf.keras.models.load_model("EyeStatus.model")

        

                 


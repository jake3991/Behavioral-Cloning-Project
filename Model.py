import pickle
import numpy as np
import math
import cv2
from keras.layers.core import Dense, Activation, Flatten
from keras.activations import relu, softmax
from keras.layers import Lambda, Dropout,MaxPooling2D
from keras.layers import Cropping2D
from keras.layers import Convolution2D
#import import_data
from import_data import images_long
from import_data import labels
from keras.models import Sequential
import matplotlib.pyplot as plt

import json
import os
import h5py
import pickle
from sklearn.utils import shuffle


#Shuffle data
images_long,labels = shuffle(images_long,labels,random_state=0)


# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

#name model, model
model = Sequential()

#Keras model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(20, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(10,3,3,border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(5,3,3,border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(2,3,3,border_mode='valid'))
model.add(MaxPooling2D(pool_size=(2,2))) 
model.add(Convolution2D(10,3,3,border_mode='valid'))
model.add(Dropout(0.33))
model.add(Flatten())
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile('adam','mse',['accuracy'])
history_object = model.fit(images_long,labels,batch_size=50,nb_epoch=10, validation_split=0.2)


# Save model as json file
with open('model.json', 'w') as fd:
   json.dump(model.to_json(), fd)
model.save('model.h5')


print(history_object.history.keys())

#plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()loss



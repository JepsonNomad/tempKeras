# Training a small temperature reader
import sys
import os
import glob

import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

np.random.seed(19150119) # Endurance gets frozen in

# Define path to directory
path_to_dir = sys.argv[1]

# Set wd
os.chdir(path_to_dir)
# Generate list of training images
imList = glob.glob("train/*.JPG")
imList = sorted(imList)
# Load known numbers 
answers = pd.read_csv("train/y.csv",header=None)

# Convert answers to useable format
y = answers.loc[:,1:5]
y = y.to_numpy()

# Load images and convert to useable format
xCV = [cv.imread(i) for i in imList]
xNP = np.asarray(xCV)
x = xNP

x.shape
y.shape

# Convert y to One Hot Encoding
yOHE = to_categorical(y)

# Rescale the images from [0,255] to the [0.0,1.0] range.
x = x/255.0

# Split up data for training and testing
X_train, X_test, y_train, y_test = train_test_split(x,
                                                    yOHE,
                                                    test_size=0.1,
                                                    random_state=42)
# Format to work with the keras model workflow
y_train = y_train.swapaxes(1,0)
y_test = y_test.swapaxes(1,0)
y_train = list(y_train)
y_test = list(y_test)

X_train.shape
X_test.shape


#### DESIGN THE MODEL ----
# See https://sajalsharma.com/portfolio/digit_sequence_recognition

# Hyperparameters 
batchsize = 64
nclasses = 14
nepoch = 100

# Image dimensions
nrows = X_train.shape[1]
ncols = X_train.shape[2]
nchannels = X_train.shape[3]

# Number of conv2d filters to use
nfilters = 32
# Size of pooling area for max pooling
pool_size = (2, 2)
# Convolution kernel size
kernel_size = (3, 3)

# Define the input
inputs = Input(shape=(nrows, ncols, nchannels))

# Set up convolutional layers first
cov = Convolution2D(nfilters,kernel_size[0],kernel_size[1],border_mode='same')(inputs)
cov = Activation('relu')(cov)
cov = Convolution2D(nfilters,kernel_size[0],kernel_size[1])(cov)
cov = Activation('relu')(cov)
cov = MaxPooling2D(pool_size=pool_size)(cov)
cov = Dropout(0.25)(cov)
cov_out = Flatten()(cov)

# Next the dense Layers
cov2 = Dense(128, activation='relu')(cov_out)
cov2 = Dropout(0.5)(cov2)

# Finally the positional prediction layers
c0 = Dense(nclasses, activation='softmax')(cov2)
c1 = Dense(nclasses, activation='softmax')(cov2)
c2 = Dense(nclasses, activation='softmax')(cov2)
c3 = Dense(nclasses, activation='softmax')(cov2)
c4 = Dense(nclasses, activation='softmax')(cov2)

# Define the model
model = Model(input=inputs,output=[c0,c1,c2,c3,c4])

#Compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Look at the model
model.summary()

# Fit the model
history = model.fit(X_train, y_train, batch_size = batchsize, 
          nb_epoch = nepoch, verbose=1,
          validation_data=(X_test, y_test))

# Look at the fit
history.history

# Summarize accuracy
plt.plot(history.history['dense_2_acc'])
plt.plot(history.history['dense_3_acc'])
plt.plot(history.history['dense_4_acc'])
plt.plot(history.history['dense_5_acc'])
plt.plot(history.history['dense_6_acc'])
plt.plot(history.history['val_dense_2_acc'])
plt.plot(history.history['val_dense_3_acc'])
plt.plot(history.history['val_dense_4_acc'])
plt.plot(history.history['val_dense_5_acc'])
plt.plot(history.history['val_dense_6_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Position 1', 'Position 2', 'Position 3', 'Position 4', 'Position 5', 'Validation 1', 'Validation 2', 'Validation 3', 'Validation 4', 'Validation 6'], loc='lower right')
plt.savefig("conv2D_accuracy.jpeg")
plt.close()

# Summarize loss
plt.plot(history.history['loss'])
plt.plot(history.history['dense_2_loss'])
plt.plot(history.history['dense_3_loss'])
plt.plot(history.history['dense_4_loss'])
plt.plot(history.history['dense_5_loss'])
plt.plot(history.history['dense_6_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Global loss', 'Position 1', 'Position 2', 'Position 3', 'Position 4', 'Position 5'], loc='upper right')
plt.savefig("conv2D_loss.jpeg")

#### Save the model ----
model.save("tempSequence_conv2d.h5")



#### Load libraries ----
import os
import glob 
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Define image directory 
imgDir = sys.argv[1]


# Load model
modelDir = sys.argv[2]
modelName = "tempSequence_conv2d.h5"

modelPath = os.path.join(modelDir, modelName)

my_model = tf.keras.models.load_model(modelPath)
my_model.summary()

# Generate predictions from model
os.chdir(imgDir)

test_files = glob.glob("*.JPG")
test_files = sorted(test_files)
# test_imagery = [os.path.join(imgDir, i) for i in test_files]
test_imagery = [cv.imread(i) for i in test_files]
test_array = np.asarray(test_imagery)
test_array_norm = test_array/255.0

test_data_output = my_model.predict(test_array_norm)


#### Convert predicted values to temperature data ----

p0 = test_data_output[0]
p1 = test_data_output[1]
p2 = test_data_output[2]
p3 = test_data_output[3]
p4 = test_data_output[4]

p0vals = [np.argmax(p0[i,:]) for i in range(p0.shape[0])]
p1vals = [np.argmax(p1[i,:]) for i in range(p1.shape[0])]
p2vals = [np.argmax(p2[i,:]) for i in range(p2.shape[0])]
p3vals = [np.argmax(p3[i,:]) for i in range(p3.shape[0])]
p4vals = [np.argmax(p4[i,:]) for i in range(p4.shape[0])]

p0ser = pd.Series(p0vals).astype(str)
p1ser = pd.Series(p1vals).astype(str)
p2ser = pd.Series(p2vals).astype(str)
p3ser = pd.Series(p3vals).astype(str)
p4ser = pd.Series(p4vals).astype(str)

p0ser = p0ser.replace('13','-')
p1ser = p1ser.replace('10',' ')
p1ser = p1ser.replace('11',' ')
p1ser = p1ser.replace('12',' ')
p2ser = p2ser.replace('10',' ')
p2ser = p2ser.replace('11',' ')
p2ser = p2ser.replace('12',' ')
p3ser = p3ser.replace('10',' ')
p3ser = p3ser.replace('11',' ')
p3ser = p3ser.replace('12',' ')
p4ser = p3ser.replace('10',' ')
p4ser = p3ser.replace('12',' ')

temp = p0ser+p1ser+p2ser+p3ser+p4ser
temp.astype(int)

#### Format into dataframe and export ----
predsDF = pd.DataFrame({'file':test_files,'temp':temp})
predsDF.to_csv("DATA_digitPreds.csv", index = False)

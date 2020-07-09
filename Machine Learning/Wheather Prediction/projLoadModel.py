# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 23:23:36 2019

@author: Golferate
"""

import numpy as np
np.random.seed(123)  # for reproducibility
import keras
from keras import optimizers	
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
import h5py
from timeit import default_timer as timer
from keras.utils import normalize
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold 
from scipy import ndimage
import scipy
from skimage.color import rgb2gray
import cv2
from keras.layers import Convolution2D, MaxPooling2D,Conv2D,MaxPool2D
from keras.models import model_from_json
import warnings
warnings.filterwarnings("ignore")

def load_data():
    train_dataset = h5py.File('dataset/weather.h5', "r")
    X = np.array(train_dataset["test_img"][:]) # your train set features
    y = np.array(train_dataset["test_labels"][:]) # your train set labels
    return X, y
#Load dataset
X, y = load_data()

# load json and create model
json_file = open('model_weatherB256_E100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weather_weightB256_E100.h5")
print("Loaded model from disk")
 
c_name = ["cloudy","rainy","sunny","sunrise/sunset"]
##Test by outside image
#my_images = [ "cloudy1.jpg", "cloudy2.jpg", "cloudy3.jpg",  "cloudy4.jpg"
#             , "rain1.jpg", "rain20.jpg","rain3.jpg","rain7.jpg"
#             , "shine39.jpg", "shine236.jpg", "shine3.jpg", "shine6.jpg"
#             , "sunrise200.jpg", "sunrise199.jpg", "sunrise198.jpg", "sunrise197.jpg"
#             ]   
#num_px = 64
#for im in my_images:
#    fname = "images/img/" + im
#    image = np.array(ndimage.imread(fname, flatten=False))
#    my_image = scipy.misc.imresize(image, size=(num_px, num_px,3))/255
#    reshape_my_image = my_image.reshape(1,3,num_px,num_px)
##    reshape_my_image = reshape_my_image.astype('float32')
##    reshape_my_image /= 255 
#    prediction = loaded_model.predict_classes(reshape_my_image)
#   # print('This Picture is :',int(prediction))
#    plt.figure()
#    plt.title(c_name[int(prediction)])
#    plt.imshow(my_image)

#Test img by test image
print(X.shape[0])
X_test = X.reshape(X.shape[0],  3, 64, 64)
X_test = X_test.astype('float32')
X_test /= 255

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer= 'rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

no_img = 253
k = np.array(X_test[no_img])
y = k.reshape(1,3,64,64)
prediction = loaded_model.predict_classes(y)   
plt.figure()
plt.title(c_name[int(prediction)])
Test_img = X[no_img]
plt.imshow(Test_img)

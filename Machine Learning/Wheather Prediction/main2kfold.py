# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 16:38:00 2019

@author: Golferate
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:01:41 2019

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

#k-fold cross validation
kf = RepeatedKFold(n_splits=7, n_repeats=10, random_state=None) 
for train_index, test_index in kf.split(X):
      X_train, X_test = X[train_index], X[test_index] 
      y_train, y_test = y[train_index], y[test_index]
X_train = X_train.reshape(X_train.shape[0], 3, 64, 64)
X_test = X_test.reshape(X_test.shape[0],  3, 64, 64)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
Y_train = np_utils.to_categorical(y_train, 4)
Y_test = np_utils.to_categorical(y_test, 4)

#Model
model = Sequential()
model.add(Conv2D(64, kernel_size =5, activation='relu' , data_format='channels_first',input_shape= X_train.shape[1:] ))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))
model.add(Dropout(0.5))
model.add(Conv2D(32, kernel_size =3, activation='relu', padding='valid' , data_format='channels_first'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(4, activation='sigmoid'))


optimize = 'adam'
#optimize = 'rmsprop'
#optimize = keras.optimizers.SGD(lr=0.075, momentum=0.0, decay=0., nesterov=False)
#optimize = keras.optimizers.Adam(lr=0.75, beta_1=0.9, beta_2=0.999, decay=0., amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer= optimize,
              metrics=['accuracy'])
start = timer() 
history = model.fit(X_train, Y_train, 
          batch_size=256, nb_epoch=100 , verbose=2, validation_data = (X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
end = timer()
print("You run : {} s".format(end - start))
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

#
## serialize model to JSON
#model_json = model.to_json()
#with open("model_weatherB256_E100.json", "w") as json_file:
#    json_file.write(model_json)
## serialize weights to HDF5
#model.save_weights("model_weather_weightB256_E100.h5")
#print("Saved model to disk")
# 
## later...
 
# load json and create model
json_file = open('model_weatherB256_E100.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weatherB256_E100.h5")
print("Loaded model from disk")
 
## evaluate loaded model on test data
#loaded_model.compile(loss='categorical_crossentropy', optimizer= optimize, metrics=['accuracy'])
#score = model.evaluate(X_test, Y_test, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


c_name = ["cloudy","rainy","sunny","sunrise/sunset"]

#my_images = [ "cloudy1.jpg", "cloudy2.jpg", "cloudy3.jpg",  "cloudy4.jpg"
#             , "rain1.jpg", "rain2.jpg","rain3.jpg","rain4.jpg"
#             , "shine20.jpg", "shine21.jpg", "shine22.jpg", "shine23.jpg"
#             , "sunrise200.jpg", "sunrise199.jpg", "sunrise198.jpg", "sunrise197.jpg"
#             ]  
#
#num_px = 64
#for im in my_images:
#    fname = "images/img/" + im
#    image = np.array(ndimage.imread(fname, flatten=False))
#    my_image = scipy.misc.imresize(image, size=(num_px, num_px,3))/255
#    reshape_my_image = my_image.reshape(1,3,num_px,num_px)
#    prediction = model.predict_classes(reshape_my_image)
#    prediction2 = loaded_model.predict_classes(reshape_my_image)
#   # print('This Picture is :',int(prediction))
#    plt.figure()
#    plt.title("new_"+ c_name[int(prediction)])
#    plt.imshow(my_image)
#    plt.figure()
#    plt.title("old_"+ c_name[int(prediction2)])
#    plt.imshow(my_image)
#    

k = np.array(X_train[704]) #
y = k.reshape(1,3,64,64)
prediction = model.predict_classes(y)   
plt.figure()
plt.title(c_name[int(prediction)])
Test_img = X[704]
plt.imshow(Test_img)


plt.figure()
print(history.history.keys())
plt.plot(np.squeeze(history.history['loss']),label='Train Loss')
plt.plot(np.squeeze(history.history['val_loss']),label='Test Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("loss&val_loss")
#plt.savefig('loss.png')
plt.legend()
plt.show()

plt.plot(np.squeeze(history.history['acc']),label="Train Accuracy")
plt.plot(np.squeeze(history.history['val_acc']),label="Test Accuracy")
plt.ylabel('acc')
plt.xlabel('epoch')
plt.title("acc&val_acc")
#plt.savefig('acc.png')
plt.legend()
plt.show()
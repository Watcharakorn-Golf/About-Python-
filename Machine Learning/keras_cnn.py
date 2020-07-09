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
from keras.layers import Conv2D
# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#print() # (60000, 28, 28)
#plt.imshow(X_train[0])
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

#print(y_train.shape)# (60000,)
#print (y_train[0])
# Convert 1-dimensional class arrays to 10-dimensional class matrices
y_train = np.transpose(y_train)
y_test = np.transpose(y_test)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
#print (Y_train.shape )# (60000, 10)

model = Sequential()
model.add(Conv2D(64, kernel_size =5, activation='relu' , data_format='channels_first',input_shape=(1, 28, 28)))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))
model.add(Conv2D(32, kernel_size =3, activation='relu', padding='valid' , data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid', data_format='channels_first'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

print(model.summary())

#optimize = 'adam'
optimize = keras.optimizers.SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
#optimize = keras.optimizers.Adam(lr=0.001, beta_1=0.09, beta_2=0.099, decay=0., amsgrad=False)
#optimize = keras.optimizers.RMSprop(lr=0.001, rho=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer= optimize,
              metrics=['accuracy'])
start = timer() 
history = model.fit(X_train, Y_train, 
          batch_size=512, nb_epoch=50, verbose=2, validation_data = (X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
end = timer()
print("You run : {} s".format(end - start))
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
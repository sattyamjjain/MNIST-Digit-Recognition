#Simple Convolutional Neural Network for MNIST

#import libraries
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#fix random seed for reproducibility
seed=7
numpy.random.seed(seed)

#load the dataset
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#reshape to be [sample][pixel][width][height]
X_train = X_train.reshape(X_train.shape[0],1,28,28).astype('float32')
X_test = X_test.reshape(X_test.shape[0],1,28,28).astype('float32')

#normalize input from 0-255 to 0-1
X_train = X_train/255
X_test = X_test/255

#one hot encode output
Y_train = np_utils.to_categorical(Y_train) 
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

#define baseline model
def baseline_model():
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(50,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=200,verbose=2)
    
#final evaluation of the model
scores = model.evaluate(X_test,Y_test,verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
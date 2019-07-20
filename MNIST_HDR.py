#Handwritten Digit Recognition
#BASELNE MODEL FOR MULTILAYER PERCEPTRON
#import libraries
import numpy
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#plot Images
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))

plt.show()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#fix random seed for reproducibility
seed=7
numpy.random.seed(seed)

#load the dataset
(X_train,Y_train),(X_test,Y_test) = mnist.load_data()

#flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

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
    model.add(Dense(num_pixels,input_dim = num_pixels,kernel_initializer = 'normal',activation = 'relu'))
    model.add(Dense(num_classes,kernel_initializer = 'normal',activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

#build the model
model = baseline_model()
#fit the model
model.fit(X_train,Y_train,validation_data=(X_test,Y_test),epochs=10,batch_size=200,verbose=2)

#final evaluation of the model
scores = model.evaluate(X_test,Y_test,verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 18:31:49 2018

@author: Ibo Turk
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from IboTurk_part1 import training_set as X_train
from IboTurk_part1 import test_set as X_test
from IboTurk_part1 import training_set_values as y_train
from IboTurk_part1 import test_set_values as y_test
from IboTurk_part1 import image_shape
from IboTurk_part1 import results_path
from matplotlib import pyplot as plt

print (X_train.shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_train /= 255

print(y_train[:10])

Y_train = np_utils.to_categorical(y_train, 2)
Y_test = np_utils.to_categorical(y_test, 2)

print(Y_train[:10])

def VGG16(name, X_train, Y_train):
    
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(300,400,3)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=32, nb_epoch=10, verbose=1)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    model_json = model.to_json()
    with open(results_path + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(results_path + name + ".h5")
    print("Saved model to disk")
    return score
    
if __name__ == "__main__":
    filename = "NN_"
    x = 1
    #for x in range(5):
    print("training neural net :", x+1)
    set_size = 5*(x+1)
    score = VGG16(filename + str(x), X_train[:set_size], Y_train[:set_size])
    
    file = open(results_path + "results.txt", "a")
    file.write(str(score[0]))
    file.write(str(score[1]))
    
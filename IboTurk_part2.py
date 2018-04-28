# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:30:18 2018

@author: Ibo Turk
"""
#from IboTurk_part1 import data
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import ZeroPadding2D, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import Callback
from IboTurk_part0 import data as X_train
from IboTurk_part0 import test_set as X_test
from IboTurk_part0 import data_values as y_train
from IboTurk_part0 import test_set_values as y_test
from IboTurk_part0 import image_shape
from IboTurk_part0 import data_number
from IboTurk_part0 import result_path
from IboTurk_part0 import number_name
from IboTurk_part0 import nn_name
from IboTurk_part0 import bs
from IboTurk_part0 import epoch_num
from IDG import ITdatagen
from matplotlib import pyplot as plt

Y_train = np_utils.to_categorical(y_train, data_number)
Y_test = np_utils.to_categorical(y_test, data_number)

def createNN():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=image_shape))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(data_number, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    ITdatagen.fit(X_train)
    
    his = model.fit_generator(ITdatagen.flow(X_train, Y_train, 
                                             batch_size=bs),
                                             callbacks=[FitCallBack((X_test, Y_test))],
                                             steps_per_epoch=bs,
                                             epochs = epoch_num)
    
    saveModel(model, his)

def createNN2():
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=image_shape))
    print("0-Zero Padding Layer-", model.output_shape)
    model.add(Conv2D(64, 3, 3, activation='relu', input_shape=image_shape))
    print("1-Convolution Layer-" , model.output_shape)
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("2-Pooling Layer-" , model.output_shape)
    model.add(Conv2D(64, 3, 3, activation='relu'))
    print("3-Convolution Layer-" , model.output_shape)
    model.add(MaxPooling2D(pool_size=(2,2)))
    print("4-Pooling Layer-" , model.output_shape)
    model.add(Dropout(0.25))
    print("5-Dropout Layer-" , model.output_shape)
    model.add(Flatten())
    print("6-Flatten Layer-" , model.output_shape)
    model.add(Dense(128, activation='relu'))
    print("7-Dense Layer-" , model.output_shape)
    model.add(Dropout(0.5))
    print("8-Dropout Layer-" , model.output_shape)
    model.add(Dense(data_number, activation='softmax'))
    print("9-Dense Layer-" , model.output_shape)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    ITdatagen.fit(X_train)
    his = model.fit_generator(ITdatagen.flow(X_train, Y_train, batch_size=bs),
                              callbacks=[FitCallBack((X_test, Y_test))],
                              steps_per_epoch=128,
                              epochs = epoch_num)
    saveModel(model, his)

def createNN3():
    model = Sequential()
    model.add(Conv2D(46, 3, 3, activation='relu', input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(92, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(138, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(276, activation='relu'))
    model.add(Dense(data_number, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    ITdatagen.fit(X_train)
    his = model.fit_generator(ITdatagen.flow(X_train, Y_train, batch_size=bs),
                              callbacks=[FitCallBack((X_test, Y_test))],
                              steps_per_epoch=128,
                              epochs = epoch_num)
    saveModel(model, his)

def trainNN(x):
    with open(number_name, "r") as rptr:
        numofnn = int(rptr.readline())
    if x <1 or x > numofnn:
        print("invaid entry. Please choose from 1-", numofnn)
        return None
    model = load_model(result_path + nn_name + str(x) + '.h5')
    hist = model.fit_generator(ITdatagen.flow(X_train, Y_train,
              batch_size=bs), steps_per_epoch=128, epochs = epoch_num)
    updateModel(model, x, hist)
    return model
  
def saveModel(model, his):
    #Testing the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    #Reading the model numbers
    with open(number_name, "r") as rptr:
        numofnn = int(rptr.readline()) + 1
    #Saving the model
    model.save(result_path + nn_name + str(numofnn) + '.h5')
    #rewriting number.txt
    with open(number_name, "w") as rptr:
        rptr.write(str(numofnn))
    saveResults(numofnn, score, hist=his)

def saveResults(x, score, isTrain = False, hist = ''):
    opr = str(hist.history) + resultString(x, score, data_number, bs, epoch_num, X_test.shape[0], image_shape)
    if isTrain:
        opr = opr[:-1] + "(RETRAIN)\n"
    with open(result_path + "results.txt", "a") as rptr:
        rptr.write(opr)

def resultString(x, score, dn, b, e, ts, i):
    return str(x) + ' ' + str(score[0]) + ' ' + str(score[1]) + ' ' + str(dn) + ' ' + str(b) + ' ' + str(e) + ' ' + str(ts) + ' ' + str(i) + '\n'

def updateModel(model, x, his):
    #Testing the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print(score)
    #Saving the model
    model.save(result_path + nn_name + str(x) + '.h5')
    saveResults(x, score, isTrain = True, hist=his)
    
class FitCallBack(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    
    def on_epoch_end(self, epoch, logs={}):
        x,y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose = 0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))
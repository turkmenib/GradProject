# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:04:06 2018

@author: Ibo Turk
"""

import cv2
import os
import numpy as np
from preprocessing import ApplyFunction, ZiscFunction, printinfo
from matplotlib import pyplot as plt

data_path = 'books/'
test_set_path = 'test set/'
result_path = 'results/'
number_name = 'number.txt'
nn_name = "NN_"
bs = 1024
epoch_num = 1
data_number = 4
image_shape = (200, 133, 3)

data = []
data_values = []

for filename in os.listdir(data_path):
    image = cv2.imread(data_path + filename)
    image = image[:,:,::-1]
    data.append(image)
    data_values.append(int(filename[1:4]))
    
data = np.array(data)
data_values = np.array(data_values)

test_set = []
test_set_values = []

for filename in os.listdir(test_set_path):
    image = cv2.imread(test_set_path + filename)
    image = image[:,:,::-1]
    test_set.append(image)
    test_set_values.append(int(filename[-5]) - 1)
test_set = np.array(test_set)
test_set_values = np.array(test_set_values)

data = data.astype('float32')
test_set = test_set.astype('float32')

#X_train /= 255
#X_test /= 255
test_set = ApplyFunction(test_set, ZiscFunction)

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:39:44 2018

@author: Ibo Turk
"""
import cv2
import numpy as np
mean = 0
std = 0
def ZiscFunction(image):
    global mean
    global std
    mean, std = cv2.meanStdDev(image)
    n = len(image[0])
    m = len(image)
    for x in range(m):
        for y in range(n):
            image[x][y][0] = (image[x][y][0] - mean[0][0])/std[0][0]
            image[x][y][1] = (image[x][y][1] - mean[1][0])/std[1][0]
            image[x][y][2] = (image[x][y][2] - mean[2][0])/std[2][0]
    return image

def ApplyFunction(imageset, preprocessing_function):
    newimageset = []
    for image in imageset:
        if preprocessing_function == ZiscFunction:
            newimage = ZiscFunction(image)
            newimageset.append(newimage)
        else:
            print("Function Not Found")
            exit()
    newimageset = np.array(newimageset)
    return newimageset

def printinfo():
    print("mean is: ", mean, "\nstd is: ", std)
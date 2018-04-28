# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 20:49:15 2018

@author: Ibo Turk
"""

from IboTurk_part2 import createNN, createNN2, trainNN, createNN3
from IboTurk_part2 import saveModel

print("Welcome to my program.",
      "\nWould you like to make a new neural net or train an existing one?",
      "\nCreate New VGG16: 1",
      "\nCreate New MNIST: 2",
      "\nTrain Existing: 3",
      "\nCreate New K46: 4",
      "\nExit: 0")
x = -1

while x != 0:
    if x == 1:
        createNN()
    elif x == 2:
        createNN2()
    elif x == 3:
        print("input the model to train: ")
        model_n = int(input())
        trainNN(model_n)
    elif x == 4:
        createNN3()
    x = int(input())
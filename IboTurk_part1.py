import cv2
import os
import numpy as np

data_path = 'books/'
test_set_path = 'test set/'
result_path = 'results/'
data_number = 2
image_shape = (3, 133, 200)

data = []
data_values = []
x = 0
for filename in os.listdir(data_path):
    image = cv2.imread(data_path + filename)
    image = [:,:,::-1]
    data.append(image)
    data_values.append(x)
    x += 1

print(data_values)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:52:26 2018

@author: Ibo Turk
"""

from keras.preprocessing.image import ImageDataGenerator as IDG
from IboTurk_part0 import data as x_train
from IboTurk_part0 import data_values as y_train
from matplotlib import pyplot as plt
from keras.utils import np_utils
from preprocessing import ZiscFunction

ITdatagen = IDG(
        samplewise_center = False,
        rotation_range = 360,
        shear_range = 360,
        data_format = "channels_last",
        fill_mode = "constant",
        preprocessing_function = ZiscFunction
        )

#y_train = np_utils.to_categorical(y_train, 2)
#print(y_train)
#ITdatagen.fit(x_train)
#batches = 0
#for x_batch, y_batch in ITdatagen.flow(x_train, y_train, batch_size=32):
#        batches += 1
#        if batches >= 1:
            # we need to break the loop by hand because
            # the generator loops indefinitely
 #           break

#plt.subplot(121)
#plt.imshow(x_batch[0])
#plt.subplot(122)
#plt.imshow(x_train[0])
#plt.show()
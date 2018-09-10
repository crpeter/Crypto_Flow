from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Define Python imports
import os
import sys
import time
import threading
import math
import logging
import logging.handlers
import re
import random

# Define ML imports
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split


class Model():

    def __init__(self, option):
        print(option)
        self.createNPFromFile(option.path)


    def file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    ## END FILE_LEN


    def plot_history(self, history, test_history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Error')
        for e in history.history.keys():
            plt.plot(history.epoch, history.history[e], label=e)
        for e in test_history.history.keys():
            plt.plot(test_history.epoch, test_history.history[e], label=e)
        plt.legend()
        plt.ylim([0,2])
        plt.show()

    def plot_shapes(self, history, test_history):
        print('history:', history.histor)

        # plt.clf()   # clear figure
        # plt.figure()

        # plt.plot(history.epoch, acc_train, 'bo', label='Training acc')
        # plt.plot(history.epoch, loss_train, 'bo', label='Training loss')
        # plt.plot(history.epoch, val_loss_train, 'b', label='Validation loss')
        # plt.title('Training and validation accuracy')
        # plt.xlabel('Epochs')
        # plt.ylabel('Accuracy')
        # plt.legend()


    def build_model(self, train_data):
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(3,20,)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Flatten(data_format=None),
            keras.layers.Dense(1)
        ])
        ## regression algorithm ##
        # model.compile(loss='mse',
        #         optimizer=tf.train.RMSPropOptimizer(0.001),
        #         metrics=['mae'])
        # binary crossentropy algorithm ##
        # model.compile(optimizer=tf.train.AdamOptimizer(),
        #       loss='binary_crossentropy',
        #       metrics=['accuracy'])
        # binary crossentroy accuracy and binary_cross
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])

        return model

    
    def createNPFromFile(self, filename):
        data_bundle_count = 3
        dataLength = 10134
        train_data = np.zeros((dataLength, data_bundle_count, 20))
        # try:
        with open(filename, 'r') as f:
            countMod = 0
            main_index = 0
            for x in f:
                if not x or main_index == dataLength: break
                currentData = re.findall('\d+\.\d+|\d+', x)
                if len(currentData) != 0:
                    currentDataIndex = 0
                    for e in currentData:
                        train_data[main_index][countMod][currentDataIndex] = e
                        currentDataIndex+=1
                    if countMod == 2:
                        main_index+=1
                    countMod+=1
                    countMod%=data_bundle_count
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std

        model = self.build_model(train_data)
        print(model.summary())
        print(train_data.shape)
        EPOCHS = 500

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

        train_labels = np.zeros(dataLength)
        self.createLabelsFromNumPy(train_data, train_labels)
        order = np.argsort(np.random.random(train_labels.shape))
        train_data = train_data[order]
        train_labels = train_labels[order]

        ## Conver train_data np array to tensors
        data_np = np.asarray(train_data, np.float32)
        train_tensor = tf.convert_to_tensor(data_np)

        ## Split tensors in half
        X_train, X_test = tf.split(
                                train_tensor,
                                2,
                                axis=0,
                                num=None,
                                name='split')
        
        ## Split numPy in half
        # X_train, X_test = train_test_split(train_data,
        #                                     test_size=0.5,
        #                                     random_state=42)
        y_train, y_test = train_test_split(train_labels,
                                            test_size=0.5,
                                            random_state=42)

        history = model.fit(X_train, y_train, 
                                epochs=EPOCHS,
                                steps_per_epoch=1,
                                # Only validation split on numpy array
                                #validation_split=0.2, 
                                verbose=0,
                                callbacks=[early_stop, PrintDot()])

        test_history = model.fit(X_test, y_test, 
                                epochs=EPOCHS,
                                steps_per_epoch=1,
                                # Only validation split on numpy array
                                #validation_split=0.2, 
                                verbose=0,
                                callbacks=[early_stop, PrintDot()])

        history_dict = history.history
        print('history:', history_dict.keys())
        self.plot_history(history, test_history)
        #self.plot_history(history, test_history)


        #[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

        #print("Testing set Mean Abs Error: ${:7.2f}".format(mae * 1000))
        #print(std)
        # for (e, i, o) in train_data:
        #     print('e:', e, '\ni:', i, '\no:', o, '\n')

    
    def putThroughTheOleNumberCruncher(self, npArr):
        s = 0
        s+=1

    def createLabelsFromNumPy(self, npArr, labels):
        index = 0
        maxIncrease = 0
        for e in labels:
            if index == 0:
                e = 1000
            else:
                thisChange = npArr[index][0][5] - npArr[index-1][0][5]
                if thisChange > 0:
                    if thisChange > maxIncrease:
                        maxIncrease = thisChange
                        e = 1000
                    else:
                        e+=100
                        e = float(e) / random.randint(1, 101)
                else:
                    e+=1
                    e = float(e) / random.randint(1, 101)


class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
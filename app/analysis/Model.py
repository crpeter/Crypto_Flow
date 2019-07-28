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
from scipy.signal import argrelextrema

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils.data_utils import get_file
from sklearn.cross_validation import train_test_split


class Model():

    def __init__(self, option):
        print(option)
        data = self.createNPFromFile(option.path, option.bundle_length)
        print(data)
        #self.plot_data(data, [0,1])
        self.learnFromNP(data)


    def file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    ## END FILE_LEN

    def plot_history(self, history, test_history = None):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Error')
        for e in history.history.keys():
            plt.plot(history.epoch, history.history[e], label=e)
        if test_history is not None:
            for e in test_history.history.keys():
                plt.plot(test_history.epoch, test_history.history[e], label=e)
        plt.legend()
        plt.ylim([0,1])
        plt.show()

    def plot_data(self, data, plot_values):
        plt.figure()
        plt.xlabel('Time')
        plt.ylabel('i=1 value')
        min_val = data[0][0][plot_values[0]]
        max_val = data[0][0][plot_values[0]]
        for value in plot_values:
            values = np.zeros(len(data))
            ## Create indices
            indices = np.zeros(len(values))
            index = 0
            while index < len(indices):
                indices[index] = index
                index+= 1
            ##
            avg = 0
            index = 0
            for e in data:
                values[index] = e[0][value]
                if e[0][value] > max_val:
                    max_val = e[0][value]
                elif e[0][value] < min_val:
                    min_val = e[0][value]
                index+= 1
            label_string = "data at index: " + str(value)
            plt.plot(indices, values, label=label_string)
        ## END for value in plot_values        
        plt.legend()
        plt.xlim(1000, 1500)
        plt.ylim([min_val - 0.0001, max_val + 0.0001])
        plt.show()

    def plot_labels(self, data):
        plt.figure()
        plt.xlabel('time')
        plt.ylabel('buy / sell')
        plt.plot(data, label='poop')
        plt.legend()
        plt.ylim([0,2])
        plt.show()


    def build_model(self, train_data):
        print('shape: ', train_data.shape)
        model = keras.Sequential([
            keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],train_data.shape[2],)),
            keras.layers.Dense(64, activation=tf.nn.relu),
            keras.layers.Flatten(data_format=None),
            keras.layers.Dense(1,)
        ])
        ## regression algorithm ##
        model.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.001),
                #optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, decay=0.0),
                metrics=['mae'])
        ### binary crossentropy algorithm ##
        # model.compile(optimizer=tf.train.AdamOptimizer(),
        #       loss='binary_crossentropy',
        #       metrics=['accuracy'])
        ### binary crossentroy accuracy and binary_cross
        # model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
        #                loss='binary_crossentropy',
        #                metrics=['accuracy', 'binary_crossentropy'])

        return model

    
    def createNPFromFile(self, filename, tuple_length):
        data_bundle_count = tuple_length
        dataLength = 2550
        train_data = np.zeros((dataLength, 3, 100))
        #print(train_data)
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
                    if countMod == data_bundle_count-1:
                        main_index+=1
                    countMod+=1
                    countMod%=data_bundle_count
        return train_data
    ## END CREATE NP ARRAY FROM FILE

    def learnFromNP(self, train_data):
        mean = train_data.mean(axis=0)
        std = train_data.std(axis=0)
        train_data = (train_data - mean) / std

        model = self.build_model(train_data)
        print(model.summary())
        print(train_data.shape)
        EPOCHS = 500

        early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

        train_labels = np.zeros(train_data.shape[0])
        self.createLabelsFromNumPy(train_data, train_labels)
        order = np.argsort(np.random.random(train_labels.shape))
        train_data = train_data[order]
        train_labels = train_labels[order]

        ## Conver train_data np array to tensors
        data_np = np.asarray(train_data, np.float32)
        train_tensor = tf.convert_to_tensor(data_np)

        ## Split tensors in half
        train_data, test_data = tf.split(
                                train_tensor,
                                2,
                                axis=0,
                                num=None,
                                name='split')
        
        # ## Split numPy in half
        # train_data, test_data = train_test_split(train_data,
        #                                     test_size=0.5,
        #                                     random_state=42)
        train_labels, test_labels = train_test_split(train_labels,
                                            test_size=0.5,
                                            random_state=42)

        history = model.fit(train_data, train_labels, 
                                epochs=EPOCHS,
                                steps_per_epoch=1,
                                # Only validation split on numpy array
                                #validation_split=0.2,
                                verbose=0,
                                callbacks=[early_stop, PrintDot()])

        test_history = model.fit(test_data, test_labels, 
                                epochs=EPOCHS,
                                steps_per_epoch=1,
                                # Only validation split on numpy array
                                #validation_split=0.2, 
                                verbose=0,
                                callbacks=[early_stop, PrintDot()])

        # test_history = model.evaluate(X_test, y_test, steps=1, verbose=1)
        history_dict = history.history
        print('history:', history_dict.keys())
        self.plot_history(history, test_history)
    ## END LEARN FROM NP

    
    def putThroughTheOleNumberCruncher(self, npArr):
        s = 0
        s+=1

    def createLabelsFromNumPy(self, npArr, labels):
        prices = np.zeros(len(npArr))
        price_index = 0
        for e in npArr:
            # print(e)
            prices[price_index] = e[0][4]
            price_index+=1
        local_max = argrelextrema(prices, np.greater, order=20)
        local_min = argrelextrema(prices, np.less, order=20)
        #print('max:', local_max, 'min:', local_min)
        index = 0
        for e in labels:
            labels[index] = 0
            index+=1
        index = 0
        for e in local_max:
            print(e)
            for p in e:
                print('max:', p)
            #for p in e:
#            labels[index] = 2
            index+=1
        index = 0
        for e in local_min:
            print('min:', e)
            labels[index] = 1
            index+=1
        # main_index = 0
        # for e in labels:
        #     current_price_change = npArr[main_index][0][0]
        #     next_price_change = npArr[main_index+1][0][0]
        #     num_consecutive_increases = 0
        #     index = 0
        #     while current_price_change < next_price_change and main_index+1+index < len(npArr):
        #         current_price_change = npArr[main_index+index][0][0]
        #         next_price_change = npArr[main_index+1+index][0][0]
        #         index+=1
        #         num_consecutive_increases+=1
        #     if num_consecutive_increases > 2:
        #         labels[main_index] = 1
        #     else:
        #         labels[main_index] = 0
        #     main_index += 1
        #     if main_index+1 == len(npArr):
        #         break
        self.plot_labels(prices)



class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
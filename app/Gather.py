from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Define Python imports
import os
import sys
import time
import config
import threading
import math
import logging
import logging.handlers
import re
import random

# Define ML imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt

# Define Custom imports
from Database import Database
from Orders import Orders
from BinanceAPI import BinanceAPI


formater_str = '%(asctime)s,%(msecs)d %(levelname)s %(name)s: %(message)s'
formatter = logging.Formatter(formater_str)
datefmt="%Y-%b-%d %H:%M:%S"

LOGGER_ENUM = {'debug':'debug.log', 'data':'data.log', 'errors':'general.log'}
#LOGGER_FILE = LOGGER_ENUM['pre']
LOGGER_FILE = "data_points.log"
FORMAT = '%(asctime)-15s - %(levelname)s:  %(message)s'


logger = logging.basicConfig(filename=LOGGER_FILE, filemode='a',
                             format=formater_str, datefmt=datefmt,
                             level=logging.INFO)

# Define Custom import vars
client = BinanceAPI(config.api_key, config.api_secret)


class Gather():

    def __init__(self, option):
        print("options: {0}".format(option))

        # Get argument parse options
        self.option = option

        # Define parser vars
        self.pair_id = self.option.pair_id
        self.targetFile = self.option.path

        # setup Logger
        self.logger =  self.setup_logger(self.pair_id, debug=self.option.debug)

        if self.option.test:
            self.createNPFromFile(self.targetFile)
        else:
            self.get_pair_info(self.pair_id, self.targetFile)

    ## END INIT

    def setup_logger(self, symbol, debug=True):
        """Function setup as many loggers as you want"""
        #handler = logging.FileHandler(log_file)
        #handler.setFormatter(formatter)
        #logger.addHandler(handler)
        logger = logging.getLogger(symbol)

        stout_handler = logging.StreamHandler(sys.stdout)
        if debug:
            logger.setLevel(logging.DEBUG)
            stout_handler.setLevel(logging.DEBUG)

        #handler = logging.handlers.SysLogHandler(address='/dev/log')
        #logger.addHandler(handler)
        stout_handler.setFormatter(formatter)
        logger.addHandler(stout_handler)
        return logger
    ## END SETUP_LOGGER

    def file_len(self, fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    ## END FILE_LEN

    def plot_history(self, history):
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Error')
        for e in history.history.keys():
            plt.plot(history.epoch, history.history[e], label=e)
        plt.legend()
        plt.ylim([0,2])
        plt.show()


    def build_model(self, train_data):
        model = keras.Sequential([
            keras.layers.Dense(32, activation=tf.nn.relu, 
                       input_shape=(3,20,)),
            keras.layers.Dense(32, activation=tf.nn.relu),
            keras.layers.Flatten(data_format=None),
            keras.layers.Dense(1)
        ])
        ## regression algorithm ##
        model.compile(loss='mse',
                optimizer=tf.train.RMSPropOptimizer(0.0001),
                metrics=['mae'])
        # binary crossentropy algorithm ##
        # model.compile(optimizer=tf.train.AdamOptimizer(),
        #       loss='binary_crossentropy',
        #       metrics=['accuracy'])
        ## binary crossentroy accuracy and binary_cross
        # model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
        #                loss='binary_crossentropy',
        #                metrics=['accuracy', 'binary_crossentropy'])

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
        train_tensor = tf.convert_to_tensor(data_np, np.float32)
        history = model.fit(train_tensor, train_labels, 
                                epochs=EPOCHS,
                                steps_per_epoch=1,
                                # Only validation split on numpy array
                                #validation_split=0.2, 
                                verbose=0,
                                callbacks=[early_stop, PrintDot()])

        history_dict = history.history
        print('history:', history_dict.keys())
        self.plot_history(history)

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


    def get_pair_info(self, symbol, filename):
        symbols = ['ETHBTC']#['ETHBTC', 'HOTETH', 'NANOETH']
        try:
            while 1==1:
                try:
                    f = open(filename, 'x')
                    for e in symbols:
                        data = client.get_ticker(e)
                        for key in data:
                            print("key: ", key, ":", data[key])
                            if key != 'symbol':
                                f.write(str(data[key]) + ' ')
                        f.write('\n')
                    f.write('\n')
                except:
                    f = open(filename, 'a')
                    for e in symbols:
                        data = client.get_ticker(e)
                        for key in data:
                            print("key: ", key, ":", data[key])
                            if key != 'symbol':
                                f.write(str(data[key]) + ' ')
                        f.write('\n')
                    f.write('\n')
        except:
            f.close()

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')
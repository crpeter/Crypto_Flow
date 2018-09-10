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

# Define ML imports
import tensorflow as tf
from tensorflow import keras

# Define Custom imports
from Database import Database
from Orders import Orders
from BinanceAPI import BinanceAPI

from keras.utils.data_utils import get_file

import numpy as np


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
            self.createNumPyFromFile(self.targetFile)
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

    
    def createNumPyFromFile(self, filename):
        data_bundle_count = 3
        dataLength = 10134
        train_data = np.zeros((dataLength, data_bundle_count, 20))
        try:
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

            #print(std)
            # for (e, i, o) in train_data:
            #     print('e:', e, '\ni:', i, '\no:', o, '\n')
        except:
            print('exception')

    
    def putThroughTheOleNumberCruncher(self, npArr):
        s = 0
        s+=1

    def createLabelsFromNumPy(self, np):
        s = 0
        s+=1


    def get_pair_info(self, symbol, filename):
        symbols = ['ETHBTC', 'HOTETH', 'NANOETH']
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
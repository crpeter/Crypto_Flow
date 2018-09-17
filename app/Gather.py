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

# # Define ML imports
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow import keras
# from keras.utils.data_utils import get_file
# from sklearn.cross_validation import train_test_split

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


    def get_pair_info(self, symbol, filename):
        symbols = ['ETHBTC']#['ETHBTC', 'HOTETH', 'NANOETH']
        # try:
        while 1==1:
        # try:
            # f = open(filename, 'x')
            # for e in symbols:
            #     data = client.get_ticker(e)
            #     for key in data:
            #         print("key: ", key, ":", data[key])
            #         if key != 'symbol':
            #             f.write(str(data[key]) + ' ')
            #     f.write('\n')
            #f.write('\n')
        # except:
            f = open(filename, 'a')
            for e in symbols:
                data = client.get_ticker(e)
                for key in data:
                    print("key: ", key, ":", data[key])
                    if key != 'symbol':
                        f.write(str(data[key]) + ' ')
                f.write('\n')
            #f.write('\n')
        # except:
        f.close()
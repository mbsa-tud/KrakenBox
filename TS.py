#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020

@author: krakenboxtud
"""

# suppress tensorflow CPU speedup warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import History, EarlyStopping, Callback
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout
import numpy as np

import logging

# suppress tensorflow CPU speedup warnings
logger = logging.getLogger('kraken')


class Modelfile():
    def __init__(self, sensor_id):
         
         self.sensor_id = sensor_id




class Modelstream():
    def __init__(self, sensor_id):
#        self.config = config
        self.model = load_model(os.path.join('kraken','models', 'trained_w_files',sensor_id + '_2500' + '.h5'))
        
        
    def stream_predict(self,ts):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            ts (obj): 

        Returns:
            
        """

        y_hat_batch = self.model.predict(X_test_batch)
        pass 
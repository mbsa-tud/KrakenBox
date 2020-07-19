#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020
Model for time series data

@author: krakenboxtud
"""

# suppress tensorflow CPU speedup warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import History, EarlyStopping, Callback
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation, Dropout
import numpy as np

import logging
import time

# suppress tensorflow CPU speedup warnings
logger = logging.getLogger('kraken')


class ModelOnline():
    def __init__(self, config):
         self.config = config
         
class ModelEnsemble():
    def __init__(self, config):
         self.config = config

class Modelstream():
    def __init__(self, config):
        self.config = config
        
    def save(self, tentacle):
        self.model.save(os.path.join('kraken','models','trained_w_files', tentacle.sensor_id+'_'.join(time.ctime().split())+'.h5'))
    
    def load(self, tentacle_id='R2S1'):
        """
        prepared model:
            100sec with sample time 0.1sec
            2500sec with sample time 0.1sec, trained with 250 l_b
            2500sec with sampel time 0.5sec, trained with 50 l_b
        """
        self.model = load_model(os.path.join('kraken','models',
                                             'trained_w_files',
                                             str(self.config.sample_t)+'_'+str(self.config.l_b)+'_'+str(self.config.l_f),
                                             tentacle_id + '.h5'))
        
    

    def train_arima(self, tentacle):
        pass
    
    def train_rf(self, tentacle):
        pass
    
    def train_xgboost(self, tentacle):
        pass
    
    def train_mlp(self, tentacle):
        """
        Objective is to train a model with simple enough capacity to make predictions within 100ms
        """
        cbs = [History(), EarlyStopping(monitor='val_loss',
               patience=self.config.patience,
               min_delta=self.config.min_delta,
               verbose=0)]
        self.model = Sequential()
        
        self.model.add(Dense(
                self.config.layers[0],
                input_shape=(None, tentacle.X_train.shape[2])))
        self.model.add(Dropout(self.config.dropout))
        
        self.model.add(Dense(
                self.config.layers[1]))
        self.model.add(Dropout(self.config.dropout))
        
        self.model.add(Dense(
                self.config.l_f))
        self.model.add(Activation('linear'))
        
        self.model.compile(loss=self.config.loss,
                           metrics=[self.config.metrics],
                           optimizer=self.config.optimizer)
        
        self.model.summary()
        #plot the model structure
        
        self.model.fit(tentacle.X_train,
                       tentacle.y_train,
                       batch_size=self.config.lstm_batch_size,
                       epochs=self.config.epochs,
                       validation_split=self.config.validation_split,
                       callbacks=cbs,
                       verbose=True)
        
        fig,ax = plt.subplots(1,2,figsize=(15,3))
        ax[0].plot(self.model.history.history['loss'],'bo',label='train_mse')
        ax[0].plot(self.model.history.history['val_loss'],'b',label='val_mse')
        ax[0].legend()
        ax[1].plot(self.model.history.history['mean_absolute_error'],label='train_mae')
        ax[1].plot(self.model.history.history['val_mean_absolute_error'],label='val_mae')
        ax[1].legend()
        fig.savefig(str(tentacle.sensor_id)+'_'+str(self.config.sample_t)+'_'+str(self.config.l_b))
        
    def train_lstm(self, tentacle):
        """
        Objective is to train a model with simple enough capacity to make predictions within 100ms
        """
        cbs = [History(), EarlyStopping(monitor='val_loss',
               patience=self.config.patience,
               min_delta=self.config.min_delta,
               verbose=0)]
        self.model = Sequential()
        
        self.model.add(LSTM(
                self.config.layers[0],
                input_shape=(None, tentacle.X_train.shape[2]),
                return_sequences=True))
        self.model.add(Dropout(self.config.dropout))
        
        self.model.add(LSTM(
                self.config.layers[1],
                return_sequences=False))
        self.model.add(Dropout(self.config.dropout))
        
        self.model.add(Dense(
                self.config.l_f))
        self.model.add(Activation('linear'))
        
        self.model.compile(loss=self.config.loss,
                           metrics=[self.config.metrics],
                           optimizer=self.config.optimizer)
        
        self.model.summary()
        #plot the model structure
        
        self.model.fit(tentacle.X_train,
                       tentacle.y_train,
                       batch_size=self.config.lstm_batch_size,
                       epochs=self.config.epochs,
                       validation_split=self.config.validation_split,
                       callbacks=cbs,
                       verbose=True)
        
        fig,ax = plt.subplots(1,2,figsize=(15,3))
        ax[0].plot(self.model.history.history['loss'],'bo',label='train_mse')
        ax[0].plot(self.model.history.history['val_loss'],'b',label='val_mse')
        ax[0].legend()
        ax[1].plot(self.model.history.history['mean_absolute_error'],label='train_mae')
        ax[1].plot(self.model.history.history['val_mean_absolute_error'],label='val_mae')
        ax[1].legend()
        fig.savefig(str(tentacle.sensor_id)+'_'+str(self.config.l_b))
        
    def stream_predict(self, X_test_batch):
        """
        Used trained LSTM model to predict test data arriving in batches.

        Args:
            X_test_batch: 

        Returns:
            
        """
        self.y_hat_batch = self.model.predict(X_test_batch)
        
# =============================================================================
# #%%
# config = Config('./kraken/config.yaml')
# m = Modelstream(config)        
# m.load(tentacle_id='R2S1')
# #%%
# %%timeit
# #time1 = time.time()
# y_hat = m.stream_predict(np.random.rand(1,m.config.l_b,1))
# #time2 = time.time()
# #print(time2-time1)
# =============================================================================


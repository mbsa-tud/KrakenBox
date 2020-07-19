#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020

@author: krakenboxtud
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import logging
import pickle

logger = logging.getLogger('kraken')

class Tentacle():
    """
    The tentacle sucks signal from the critical AP(access Point)
    prepare signals collected from the sensors of the CPS 
    and trasnsform them to the brain
    """
    def __init__(self, config, sensor_id):
        """
        Load and reshape sensor signals (y_hat and y_test)
        """
        
        self.config = config
        self.sensor_id = sensor_id
        
        self.train = None
        self.test = None
        
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.y_test = None
        
        self.y_hat = None
        
        
    def load_golden_run(self, *ts):
        """
        Load the saved injection free data from local file
        ( could be xlsx, csv, npy and ts-obj).
        
        Parameters
        ----------
        
        run: several retrieve options: 100sec and 2500sec, to verify that 
        more data can train more accuate LSTM

        Returns
        -------
        ts : series
            DESCRIPTION.

        """
      
        try:
            # use the data in the memory that has just been collected
            if self.config.train_data == 'online':
                ts = ts
                
            # Interface for the npz file
            elif self.config.train_data == '2500npz': 
                                       
                    arch = np.load(os.path.join(self.config.path_root,"data","train","online_collect.npz"))
                    ts = arch[self.sensor_id.lower()[-2:]]
                    ts = pd.Series(ts)
            
            # Interface for the pkl file
            elif self.config.train_data == '2500pkl':
                with open(os.path.join(self.config.path_root,"data","train",
                                       "{}_2500.pkl".format(self.sensor_id.lower())),'rb') as f:
                    ts = pickle.load(f)
                ts = ts.resample(str(self.config.sample_t)+'S').mean()
                ts = ts.fillna(method = 'ffill')
               
            # Interface for the xlsx file
            elif self.config.train_data == '100xlsx':
                dfTrain = pd.read_excel(os.path.join('.',"data","train","{}.xlsx".format(self.id)))  # two col dataframe,col0 is time stamp, col1 is ts
                
                # info about the df read from the excel
                dfTrain.head()
                plt.figure(figsize=(30,10))
                dfTrain.iloc[:,1].plot()
                
                # timestamp at 1st col, from the col0, make the timedelta list
                idx = np.array(dfTrain['time'])  # get the time col
                t_idx = list(map(lambda s: pd.Timedelta(s, unit='sec'), idx))    # encapseln the time into timedelta
                
                # transform the df into a ts obj
                ts = pd.Series(np.array(dfTrain.iloc[:,1]), index = t_idx)   # time series obj, index is pd.period_range
                
                # resample
                ts = ts.resample('0.1S').mean()  # sample time must be big enough, the span must be big enough to cover some points at least, otherwise NaN
                ts = ts.fillna(method = 'ffill')
                
                ts.index = list(range(len(ts)))
            
                    
            ts.head()
            ts.plot(figsize=(30,10))
                    
            return ts
            
        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")
            
            
    def reshape_data(self, arr, shuffle=True):
        """Shape raw input streams for ingestion into LSTM. 
        
        config.l_b
            specifies the sequence length of prior timesteps fed into the model 
            at each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            shuffle (bool): if data can be shuffled
        """

#        data = np.array([])
#        arr = arr.reshape(-1,1,1)
#        for i in range(len(arr) - self.config.l_b - self.config.l_f):
#            data = np.append(data, arr[i:i + self.config.l_b + self.config.l_f], axis=1)
#        data.swapaxes(0,1)
#        print(data.shape)
#        assert len(data.shape) == 3
        
        data = []
        for i in range(len(arr) - self.config.l_b - self.config.l_f):
            data.append(arr[i:i + self.config.l_b + self.config.l_f])
        data=np.array(data)
        print(data.shape)
        assert len(data.shape) == 3

        if shuffle:
            np.random.shuffle(data)
            
            
        self.X_train = data[:, :-self.config.l_f, :]
        self.y_train = data[:, -self.config.l_f:, 0]  # sensor value is at position 0
       

        
        
        
    def load_data_train(self, *ts):
       """
       Load the training data which is injection free.
       call the load_golden_run and then shape the data for training
       1 load golden run
       2 reshape data
       """
       train_ts = self.load_golden_run(*ts)
       self.train = np.array(train_ts).reshape(-1,1)
       
       logger.info('before shaping:  {}'.format(self.train.shape))
       self.reshape_data(self.train)
       logger.info('after shaping: {}{} '.format(self.X_train.shape,self.y_train.shape))
    
    
    def load_y_hat(self):
        self.y_hat = np.load(os.path.join('data', self.config.use_id, 'y_hat','{}.npy'.format(self.id)))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:22:10 2020

@author: krakenboxtud
"""

from tensorflow.keras.datasets import boston_housing
(X_train,y_train), (X_test,y_test) = boston_housing.load_data()
X_train
X_train.shape
y_train.shape


%time temp = np.append(t.train[0:60],t.train[1:61], axis=1)
%time temp = np.concatenate(t.train[0:60],t.train[1:61], axis=1)


pd.DataFrame(dict(zip(['a','b','c'],[np.random.rand(100),np.random.rand(100),np.random.rand(100)])))

%time np.hstack([np.random.rand(100,1),np.random.rand(100,1),np.random.rand(100,1)])
%time np.concatenate([np.random.rand(100,1),np.random.rand(100,1),np.random.rand(100,1)],axis=1)



l=[]
%%time
for i in range(1000):
    l.append(i)

arr=np.zeros(1000)
%%time
for i in range(1000):
    arr[i] = i

arr=np.array([])
%%time
for i in range(1000):
    arr = np.append(arr,i)




#check if there are package loss
if simulink_timestamp/self.config.sample_t - (len(self.time_track)-1) > 0:  # should be  0 VS 1, 0.5 VS 2
    logger.info('Package loss!!!')
    logger.info('simulink_timestamp at {} should collect {} data, but actually just {} data'.format(simulink_timestamp, int(simulink_timestamp*2+1), len(self.time_track)))
#check if there are package loss
logger.info('signal1: {} signal2: {} signal3: {}'.format(self.signal_track[self.tentacles[0]].shape[0],
            self.signal_track[self.tentacles[1]].shape[0],
            self.signal_track[self.tentacles[2]].shape[0]))
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

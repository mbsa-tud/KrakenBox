#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 22:59:18 2020

@author: krakenboxtud
"""
import time
import numpy as np
import matplotlib.pyplot as plt

from kb_util import animation




#plt.figure()
#plt.axis([0,10,-1,1])

#xs = [0, 0]
#ys = [1, 1]


#for i in range(100):
#    t_now = i*0.1
#    plt.plot(t_now, np.sin(t_now),'r.')
##    plt.draw()
##    time.sleep(0.01)
#    
#    plt.pause(0.0001)
    
time1 = time.time()

fig, ax = plt.subplots(6,1,figsize=(24,16))
for i in range(6):
    ax[i].set(xlim=(0,1000),ylim=(-2,2))
    
time2 = time.time()
    

print(time2-time1)

plt.ion()
animation(ax, 0.1, s1=0.1,
          s2=0.1,
          s3=0.1,
          p1=0.1,
          p2=0.1,
          p3=0.1)
plt.pause(5)
animation(ax, 0.1, s1=0.1,
          s2=0.1,
          s3=0.1,
          p1=0.1,
          p2=0.1,
          p3=0.1)
plt.pause(5)
    
def online_training():
    '''
    train a mlp with 5steps
    '''
    pass


#data_path
#{
#'nyc':
#'power':
#'ECG':
#'SWaT':
#}




from collections import defaultdict

num_items = 0
def tuple_counter():
    global num_items
    num_items+=1
    return (num_items, [])

d = defaultdict(tuple_counter)

d['a'][1].append("hello")
d['b'][1].append("world")
d
[*d]
{**d}


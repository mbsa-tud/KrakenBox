#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020

@author: krakenboxtud
"""

import socket, struct, os
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pickle

from curses import wrapper
import time, curses
from time import ctime 




    

from kraken.helpers import Config
import kraken.helpers as helpers
from kraken.channel import Channel
from kraken.modeling import Model,Modelstream
#from kraken.AD import AD


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('my.log')
formatter = logging.Formatter('%(asctime)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class KB_manager():
    def __init__(self,dataType):
        self.dataType = dataType
    
    	# -------------------------------- Initializing --------------------------------------------
    	# Create a socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        self.signal_batch1 = np.array([])
# =============================================================================
#         self.signal_batch = {}
#         self.signal_batch[1] = np.array([])
#         self.signal_batch[2] = np.array([])
#         self.signal_batch[3] = np.array([])
# =============================================================================
        self.time_batch = np.array([])
        
    def connect(self):
        # Bind the IP address and port.
        self.simulink_addr = ("192.168.178.46", 25000)
        self.xavier_eth0_addr = ("192.168.178.45", 54320)
        self.udp_socket.bind(self.xavier_eth0_addr)  # its a udp server, listening the requeset from simulink
    	
    
    def listening(self):
#        print("----Listening data from simulink udp block----")
        scr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
        
        window_height = curses.LINES - 2
        window_width = curses.COLS - 2
        win = curses.newwin(5 + window_height, window_width, 2, 4)
        win.addstr(1, 0, "KrakenBox Status:")
        win.addstr(3, 0, "Listening DemoSystem at:")
        win.addstr(5, 0, "TimeStamp:")
        win.addstr(7, 0, "Data:")
        win.addstr(9, 0, "Prediction:")
        win.addstr(11, 0, "Diff:")
        win.addstr(13, 0, "Detection:")
        
        win.addstr(1, 20, "Connected")
        
        
        
    	#print("Please open the Simulink file under the current working directory")
    	#print("The program is waiting until you run the Simulink file.")
    
    	#----------------------------------- Data Receiving ----------------------------------------
        
    	
        # Using a loop to receive data from Simulink
        # Can be modified by (simulationTime/sampleTime).
        
        
        
#        listen_start_time = time.time()
#        listen_start_time = ctime()
        
        modelstream1 = Modelstream('R2S1')
# =============================================================================
#         modelstream2 = Modelstream('R2S2')
#         modelstream3 = Modelstream('R2S3')
# =============================================================================
        old_ts_length = 260
        threshold = 0.2
        prior_idx = 0
        
        
        set_collect = np.array([])
        pred_collect = np.array([])
        
        for count in range(0,5000,1):
    		# Start to receive data from Simulink.
            recv_data = self.udp_socket.recvfrom(1024)
            
    		
    		# recv_data will return tuple, the first element is DATA, and the second is address information
            recv_msg = recv_data[0]
            send_addr = recv_data[1]

    		# Decode the data from Simulink whose type is double and return a tuple
            recv_msg_decode = struct.unpack("dddd", recv_msg)
            simulink_timestamp = round(recv_msg_decode[0],3)
            signal1 = round(recv_msg_decode[1],3)
# =============================================================================
#             signal2 = round(recv_msg_decode[2],3)
#             signal3 = round(recv_msg_decode[3],3)
# =============================================================================
	       
    		
            # Restore the signal and time for a batch:
            self.time_batch = np.append(self.time_batch, round(simulink_timestamp,3))
            self.signal_batch1 = np.append(self.signal_batch1, round(signal1,3))
# =============================================================================
#             self.signal_batch[2] = np.append(self.signal_batch[2], round(signal2,3))
#             self.signal_batch[3] = np.append(self.signal_batch[3], round(signal3,3))
# =============================================================================
            
            
            # for debugging, Print the receieved attr, time and signal1
# =============================================================================
#             print("addr: %s time: %s signal: %s" % (str(send_addr),simulink_timestamp, signal1))
# =============================================================================
# =============================================================================
#             print("addr: %s time: %s signal: %s %s %s" % (str(send_addr),simulink_timestamp, signal1, signal2, signal3))
# =============================================================================
            
            
            win.addstr(3, 20, str(send_addr))
            win.addstr(5, 20, str(simulink_timestamp))
            win.addstr(7, 20, str(signal1))
# =============================================================================
#             win.addstr(7, 30, str(signal2))
#             win.addstr(7, 40, str(signal3))
# =============================================================================
            
    		
            # Set the condition to jump out of this loop ???
            
            
            
            # encapseln into ts object
        	   # resample the signals according to the timestamp, standard is 0.1sec
            idx = self.time_batch
            t_idx = list(map(lambda s: pd.Timedelta(s, unit='s'), idx))    # encapseln the timestamp into timedelta obj
            
            ts1 = pd.Series(self.signal_batch1, index = t_idx)   # time series obj with pd.period_range as index
# =============================================================================
#             ts2_resampled = pd.Series(self.signal_batch[2], index = t_idx)
#             ts3_resampled = pd.Series(self.signal_batch[3], index = t_idx)
# =============================================================================
            
            # resample with 0.1sec
            # sample time must be big enough, the span must be big enough to cover some points at least, otherwise NaN
            ts1_resampled = ts1.resample('0.1S').mean()  
            ts1_resampled = ts1_resampled.fillna(method = 'ffill') #ts.index = list(range(len(ts))) 
            
# =============================================================================
#             ts2_resampled = ts2.resample('0.1S').mean()  
#             ts2_resampled = ts2_resampled.fillna(method = 'ffill')
#             
#             ts3_resampled = ts3.resample('0.1S').mean()  
#             ts3_resampled = ts3_resampled.fillna(method = 'ffill') 
# =============================================================================
                 
            
# =============================================================================
#             print('{} {} {}'.format(self.signal_batch[1].shape[0],self.signal_batch[2].shape[0], self.signal_batch[3].shape[0]))
# =============================================================================
            
            
            
            if ts1_resampled.values.shape[0] > old_ts_length:
                ts_updated = True
                old_ts_length = ts1_resampled.values.shape[0]
            else:
                ts_updated = False
            
            # collect the first l_s steps to initialize the ts obj
            if simulink_timestamp > 260 * 0.1 and  count > 260 and ts_updated:
                win.addstr(1, 20, '%-50s' % 'Detection')
                
# =============================================================================
#                 print('prior_idx{} count{}'.format(prior_idx, count))
# =============================================================================
                
                
                
                X_test_batch1 = ts1_resampled[prior_idx : prior_idx + 250].values.reshape(1,250,1)
                time1 = time.time()
                y_hat_batch1 = modelstream1.model.predict(X_test_batch1)
                time2 = time.time()
                logger.info('prediction takes {}'.format(time2-time1))
                
                set_value1 = round(ts1_resampled[prior_idx + 260 - 1],2)
                pred_value1 = round(y_hat_batch1[:,-1][0],2)
                residual1 = round(abs(set_value1 - pred_value1),2)
                
                win.addstr(9, 20, str(set_value1))
                win.addstr(11, 20, str(residual1))
                
                set_collect = np.append(set_collect,set_value1)
                pred_collect = np.append(pred_collect,pred_value1)
                
                
# =============================================================================
#                 # pred 2
#                 X_test_batch2 = self.signal_batch[2][prior_idx : prior_idx + 250].reshape(1,250,1)
#                 y_hat_batch2 = modelstream2.model.predict(X_test_batch2)
#                 
#                 
#                 set_value2 = round(self.signal_batch[2][prior_idx + 260 - 1],2)
#                 pred_value2 = round(y_hat_batch2[:,-1][0],2)
#                 residual2 = round(set_value2 - pred_value2,2)
# =============================================================================
# =============================================================================
#                 win.addstr(9, 30, str(set_value2))
#                 win.addstr(11, 30, str(residual2))
# =============================================================================
                
# =============================================================================
#                 # pred 3
#                 X_test_batch3 = self.signal_batch[3][prior_idx : prior_idx + 250].reshape(1,250,1)
#                 y_hat_batch3 = modelstream3.model.predict(X_test_batch3)
#                 
#                 
#                 set_value3 = round(self.signal_batch[3][prior_idx + 260 - 1],2)
#                 pred_value3 = round(y_hat_batch3[:,-1][0],2)
#                 residual3 = round(set_value3 - pred_value3,2)
# =============================================================================
# =============================================================================
#                 win.addstr(9, 40, str(set_value3))
#                 win.addstr(11, 40, str(residual3))
# =============================================================================
#                print(y_hat_batch1[:,-1][0], ts1_resampled[prior_idx + 260 - 1], residual1)
#                print('residual1 {}'.format(residual1))
# =============================================================================
#                 print('residual1 {}, residual2 {}, residual3 {}'.format(residual1, residual2, residual3))
# =============================================================================
                

                
                if residual1 > threshold:
#                    flag = '%-10s' % 'Error' # overite with 10space
                    flag = 'ERROR'
                    win.addstr(13, 20, flag,curses.color_pair(1))
#                    os.system('spd-say "anomaly detected"')
                    
                    # reuse the socket, udp communication session with the simulink model
    		           # Start to send data to Simulink.
# =============================================================================
#                     self.udp_socket.sendto(bytes(1), self.simulink_addr)
# =============================================================================
                else:
                    flag = 'OK'
# =============================================================================
#                 print(flag)
# =============================================================================
                    win.addstr(13, 20, flag,curses.color_pair(4))
                win.clrtoeol()
                    
#                y_hat = np.reshape(y_hat, (y_hat.size,))
                prior_idx += 1
                
            else:
                win.addstr(1, 20, '%-50s' % 'Waiting for collceting enough data')
                
            win.refresh()
            
            
            if abs(simulink_timestamp - 99.8) < 1e-3:
                np.savez('compare1',s = set_collect, p = pred_collect)
           
            
                
       # save the ts, for testing     
#        with open('ts.pkl','wb') as f:
#            pickle.dump(ts_resampled, f)    
#        ts.to_csv('comm.csv')
            
       def training_mode(self, training_mode):
           '''
           get the collected data from training mode and save them as files
           Arg:
               training_mode: boolean
               
           '''
           
           
            

    		
        
        #----------------------------------- Data Sending ----------------------------------------
                
        

            
            
    	# Using a loop to send data to Simulink
    #	while count < 101: # Can be modified by (simulationTime/sampleTime).
    #		
    #		count += 1
    
    	
        
        
    def view_signal(self, visualize=True):
        
        # Create a path to save figure:
    	path = ''
    
    	# ------------------------------------ Visualization ----------------------------------------------- 
    	# Set the time axis, 10 is the simulation end time that can be modified by user.
    	if visualize:
    		index = list(np.linspace(0, 10, (len(self.data_collect))))
    		plt.plot(index, self.data_collect)
    		plt.title("Signal Received from Simulink")
    		plt.xlabel("Time")
    		plt.ylabel("Received Data")
#    		plt.savefig(os.path.join(path, 'data_figure.png'), dpi=600)
#    		print("Close the figure to restart.")
    		plt.show()







if __name__ == "__main__":
    manager = KB_manager('stream')
    manager.connect()
    manager.listening()
    manager.udp_socket.close()
    
    
    


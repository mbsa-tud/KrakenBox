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


from kraken.config import Config
from kraken.tentacle import Tentacle
from kraken.brain import Modelstream
#from kraken.AD import AD


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler('my.log')
formatter = logging.Formatter('%(asctime)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)







#%%

class KB_manager():
    def __init__(self, dataType, config):
        self.dataType = dataType
        self.config = config
#        self.config = {'old_ts_length':260}
    
    	# -------------------------------- Initializing --------------------------------------------
    	# Create a socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        
        self.tentacles = ['R2S1', 'R2S2', 'R2S3']
        self.signal_track = dict(zip(self.tentacles, [np.array([]),np.array([]),np.array([])]))
        self.set_collect = dict(zip(self.tentacles, [np.array([]),np.array([]),np.array([])]))
        self.pred_collect = dict(zip(self.tentacles, [np.array([]),np.array([]),np.array([])]))
        self.time_batch = np.array([])
        
        
        
        self.prior_idx = 0
        
    def prepare_brain(self):
        """
          get the subbrain for the responding tentacle ready
          opt1: load the local .h5file
          opt2: online training 
        """
#        self.modelstream = {sensor:model for senor in sensor_list for model in [Modelstream('R2S1'), Modelstream('R2S2'), Modelstream('R2S3')]}
#        self.modelstream = dict(zip(self.sensor_list, list(Modelstream(sensor) for sensor in self.sensor_list)))
        m1 = Modelstream(self.config)
        m1.load(tentacle_id='R2S1')
        
        m2 = Modelstream(self.config)
        m2.load(tentacle_id='R2S2')
        
        m3 = Modelstream(self.config)
        m3.load(tentacle_id='R2S3')
        
        self.sub_brains = dict(zip(self.tentacles, [m1, m2, m3]))
        
        
    def init_layout(self):
        # establish the curse and the initialized canvas
        scr = curses.initscr()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
        
        window_height = curses.LINES - 2
        window_width = curses.COLS - 2
        self.win = curses.newwin(5 + window_height, window_width, 2, 4)
        self.win.addstr(1, 0, "KrakenBox Status:")
        self.win.addstr(3, 0, "Listening DemoSystem at:")
        self.win.addstr(5, 0, "TimeStamp:")
        self.win.addstr(7, 0, "Data:")
        self.win.addstr(9, 0, "Prediction:")
        self.win.addstr(11, 0, "Diff:")
        self.win.addstr(13, 0, "Detection:")
        
        self.win.addstr(1, 20, "Connected")    
        
    def connect(self, simulink_addr = ("192.168.178.46", 25000), xavier_eth0_addr = ("129.69.81.73", 54320)):
        # Bind the IP address and port.
        self.udp_socket.bind(xavier_eth0_addr)  # its a udp server, listening the requeset from simulink
    
    
    def listening(self):
        
        
        def flash_detect(tentacle_id='R2S1'):
            """
            display the predtion and residual on the canvas window
            """
            X_test_batch = self.signal_track[tentacle_id][self.prior_idx : self.prior_idx + self.config.l_b].reshape(1,self.config.l_b,1)
            time1 = time.time()
            y_hat_batch = self.sub_brains[tentacle_id].model.predict(X_test_batch)
            time2 = time.time()
            logger.info('Signal {} prediction takes {}'.format(tentacle_id, time2-time1))
            
            set_value = round(self.signal_track[tentacle_id][self.prior_idx + self.config.l_b + self.config.l_f - 1],2)
            pred_value = round(y_hat_batch[:,-1][0],2)
            residual = round(set_value - pred_value,2)
            
            self.win.addstr(9, 10*(int(tentacle_id[-1])+1), str(set_value))
            self.win.addstr(11, 10*(int(tentacle_id[-1])+1), str(residual))
            
            self.set_collect[tentacle_id] = np.append(self.set_collect, set_value)
            self.pred_collect[tentacle_id] = np.append(self.pred_collect, pred_value)
            
            if residual > self.config['threshold']:
        #                    flag = '%-10s' % 'Error' # overite with 10space
                flag = 'ERROR'
                self. win.addstr(13, 10*(int(tentacle_id[-1])+1), flag, curses.color_pair(1))
        #                    os.system('spd-say "anomaly detected"')
                
                # reuse the socket, udp communication session with the simulink model
        		           # Start to send data to Simulink.
        #                    self.udp_socket.sendto(bytes(1), self.simulink_addr)
            else:
                flag = 'OK'
                self.win.addstr(13, 10*(int(tentacle_id[-1])+1), flag, curses.color_pair(4))
            self.win.clrtoeol()
                
    #        y_hat = np.reshape(y_hat, (y_hat.size,))
            self.prior_idx += 1
            
    #        print(y_hat_batch1[:,-1][0], ts1_resampled[prior_idx + 260 - 1], residual1)
    #        print('residual1 {}, residual2 {}, residual3 {}'.format(residual1, residual2, residual3))
    
    
        
    
    	#----------------------------------- Data Receiving ----------------------------------------
        
        # Using a loop to receive data from Simulink, Can be modified by (simulationTime/sampleTime).
        print("----Listening data from simulink udp block----")
        print("Please open the Simulink file under the demo system working directory")
        print("The program is waiting until you run the Simulink file.")
        logger.info("listen start at {}".format(time.ctime())) #time.time()
        
        
        
        
        for count in range(0, 5000, 1): # Set the condition to jump out of this loop ???
    		# Start to receive data from Simulink.
            recv_data = self.udp_socket.recvfrom(1024)
                		
    		# recv_data will return tuple, the first element is DATA, and the second is address information
            recv_msg = recv_data[0]
            send_addr = recv_data[1]

    		# Decode the data from Simulink whose type is double and return a tuple
            recv_msg_decode = struct.unpack("dddd", recv_msg)
            simulink_timestamp = round(recv_msg_decode[0],3)
                		
            # Restore the signal and time for a batch:
            self.time_batch = np.append(self.time_batch, round(simulink_timestamp,3))
            self.signal_track[self.tentacles[0]] = np.append(self.signal_track[self.tentacles[0]], round(recv_msg_decode[1],3))
            self.signal_track[self.tentacles[1]] = np.append(self.signal_track[self.tentacles[1]], round(recv_msg_decode[2],3))
            self.signal_track[self.tentacles[2]] = np.append(self.signal_track[self.tentacles[2]], round(recv_msg_decode[3],3))
            
            #check if there are package loss
            if simulink_timestamp*10 - len(self.time_batch) > 2:
                logger.warning('simulink_timestamp-{}   time_batch-{}'.format(simulink_timestamp*10, len(self.time_batch)))
            
            
            # debugging, Print the receieved attr, time and signal1
#            logger.info("addr: %s time: %s signal: %s %s %s" % (str(send_addr), simulink_timestamp, signal1, signal2, signal3))
            
            # prediction only after warm up (collect the first l_s steps to initialize the ts obj)
            if simulink_timestamp > (self.config.l_b + self.config.l_f) * self.config.sample_t: #and count > self.config.l_b + self.config.l_f
                
                self.win.addstr(1, 20, '%-50s' % 'Detection')
                
                for tentacle_id in self.tentacles:
                    flash_detect(tentacle_id)
                
                logger.info('prior_idx{} count{}'.format(self.prior_idx, count))
                                
            else:
                self.win.addstr(1, 20, '%-50s' % 'Waiting for collceting enough data')
                
            self.win.refresh()
            
            
            self.win.addstr(3, 20, str(send_addr))
            self.win.addstr(5, 20, str(simulink_timestamp))
            self.win.addstr(7, 20, str(round(recv_msg_decode[1],3)))
            self.win.addstr(7, 30, str(round(recv_msg_decode[2],3)))
            self.win.addstr(7, 40, str(round(recv_msg_decode[3],3)))
                    
            #check if there are package loss
            logger.info('{} {} {}'.format(self.signal_track[self.tentacles[0]].shape[0], self.signal_track[self.tentacles[1]].shape[0], self.signal_track[self.tentacles[2]].shape[0]))
            

            
        
    
    def save_ts(self, ts_resampled, pkl_name='ts.pkl', csv_name='comm.csv'):
        """
        save the ts(obj and csv), for visualization/testing
        """
        with open(pkl_name,'wb') as f:
            pickle.dump(ts_resampled, f)    
        ts_resampled.to_csv(csv_name)
        
        
        
    def training_mode(self, training_mode):
       '''
       get the collected data from training mode and save them as files
       Arg:
           training_mode: boolean
           
       '''
       if abs(self.simulink_timestamp - 99.8) < 1e-3:
                np.savez('compare1',s = set_collect, p = pred_collect)
           
   

            
    def arr2ts(self, signal_nb=1):
        """
        encapseln into ts object, Index of series: simulink_time->pd.time_delta
    	    resample the signals according to the timestamp, standard is 0.1sec
        """
        idx = self.time_batch
        t_idx = list(map(lambda s: pd.Timedelta(s, unit='s'), idx))    # encapseln the timestamp into timedelta obj
        ts = pd.Series(self.signal_track[signal_nb], index = t_idx)   # time series obj with pd.period_range as index
        # resample with 0.1sec
        # sample time must be big enough, the span must be big enough to cover some points at least, otherwise NaN value appears
        ts_resampled = ts.resample('0.1S').mean()  
        ts_resampled = ts_resampled.fillna(method = 'ffill') #ts.index = list(range(len(ts))) 
        return ts_resampled
            
    def check_resample(self):
        """
        keep focus on a ts
        if stashing signals while resampling at the same time, e.g. 10sec->resample->1sec, then 9 of these signals will be ignored 
        this function is used, after resampling, to check if there are new values added into the ts
        """
        if ts_resampled.values.shape[0] > self.old_ts_length:
            ts_updated = True
            self.old_ts_length = ts_resampled.values.shape[0]
        else:
            ts_updated = False
        return ts_updated
    
    

    		
        
        #----------------------------------- Data Sending ----------------------------------------                
        
    	# Using a loop to send data to Simulink
    #	while count < 101: # Can be modified by (simulationTime/sampleTime).
    #		
    #		count += 1
    
    	
        
        
    def view_ts_signal(self, visualize=True):
        
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
#            plt.savefig(os.path.join(path, 'data_figure.png'), dpi=600)
#            print("Close the figure to restart.")
            plt.show()
        else:
            pd.Series(self.data_collect).describe()
        







if __name__ == "__main__":
    config = Config('./kraken/config.yaml')
    manager = KB_manager('stream',config)
    manager.prepare_brain()
    manager.init_layout()
    manager.connect()
    manager.listening()
    manager.udp_socket.close()
    
# =============================================================================
# #%%
# config = Config('./kraken/config.yaml')
# t = Tentacle(config,'R2S3')
# t.load_data_train()
# m = Modelstream(config)
# m.train_lstm(t)
# =============================================================================


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020

@author: krakenboxtud
"""

import socket, struct, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from curses import wrapper
import time, curses
import kb_gpio as alarmer
from kb_util import detect_plot, aggregate_training_data

from kraken.config import Config
from kraken.tentacle import Tentacle
from kraken.brain import ModelStream

import logging
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

handler = logging.FileHandler('kraken.log')
formatter = logging.Formatter('%(asctime)s  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class KB_manager():
    def __init__(self, dataType, config):
        self.dataType = dataType
        self.config = config
#        self.config = {'old_ts_length':260}
    
    	# -------------------------------- Initializing --------------------------------------------
    	# Create a socket
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
        
        #to do, save it in 4column np or pd
        self.tentacles = ['R2S1', 'R2S2', 'R2S3']
        
        self.steps = int(self.config.sim_t/self.config.sample_t)
        self.time_track = np.zeros(self.steps)
        self.signal_track = dict(zip(self.tentacles, [np.zeros(self.steps),np.zeros(self.steps),np.zeros(self.steps)]))
        
        self.set_collect = dict(zip(self.tentacles, [np.zeros(self.steps),np.zeros(self.steps),np.zeros(self.steps)]))
        self.pred_collect = dict(zip(self.tentacles, [np.zeros(self.steps),np.zeros(self.steps),np.zeros(self.steps)]))
        
        
        self.prior_idx = 0
        
        
#        if self.config.animation:    
#            fig, self.ax = plt.subplots(6,1,figsize=(24,16))
#            for i in range(6):
#                self.ax[i].set(xlim=(0,100),ylim=(-2,2))
        
#        animation(self.ax, 0.1, s1=0.1,
#                          s2=0.1,
#                          s3=0.1,
#                          p1=0.1,
#                          p2=0.1,
#                          p3=0.1)
#        plt.pause(0.0001)
         
        
    def prepare_brain(self):
        """
          get the subbrain for the responding tentacle ready
          opt1: load the local .h5file
          opt2: online training 
        """
#        self.ModelStream = {sensor:model for senor in sensor_list for model in [ModelStream('R2S1'), ModelStream('R2S2'), ModelStream('R2S3')]}
#        self.ModelStream = dict(zip(self.sensor_list, list(ModelStream(sensor) for sensor in self.sensor_list)))
        m1 = ModelStream(self.config)
        m1.load(tentacle_id='R2S1')
        
        m2 = ModelStream(self.config)
        m2.load(tentacle_id='R2S2')
        
        m3 = ModelStream(self.config)
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
        self.win.refresh()
        
    def connect(self):
        # Bind the IP address and port.
        simulink_addr = (self.config.simulink_eth0, 25000)
        xavier_eth0_addr = (self.config.xavier_eth0, 54320)
        self.udp_socket.bind(xavier_eth0_addr)  # its a udp server, listening the requeset from simulink
    
    
    
    
    def recv_track(self, count):
        """
        receieve data from 
        
        Return:
            send_addr:str
            simulink_timestamp
            signal1
            signal2
            signal3
            int(recv_msg_decode[4])
            
        """
    # Start to receive data from Simulink.
        recv_data = self.udp_socket.recvfrom(1024)
            		
		# recv_data will return tuple, the first element is DATA, and the second is address information
        recv_msg = recv_data[0]
        send_addr = recv_data[1]

		# Decode the data from Simulink whose type is double and return a tuple
        recv_msg_decode = struct.unpack("ddddd", recv_msg)
        simulink_timestamp = round(recv_msg_decode[0],3)
        signal1 = round(recv_msg_decode[1],3)
        signal2 = round(recv_msg_decode[2],3)
        signal3 = round(recv_msg_decode[3],3)
          
        #----------------------------------- Track ----------------------------------------  		
        # Restore the signal and time for a batch:
#        self.time_track = np.append(self.time_track, round(simulink_timestamp,3))
#        self.signal_track[self.tentacles[0]] = np.append(self.signal_track[self.tentacles[0]], signal1)
#        self.signal_track[self.tentacles[1]] = np.append(self.signal_track[self.tentacles[1]], signal2)
#        self.signal_track[self.tentacles[2]] = np.append(self.signal_track[self.tentacles[2]], signal3)
        
        self.time_track[count] = round(simulink_timestamp,3)
        self.signal_track[self.tentacles[0]][count] = signal1
        self.signal_track[self.tentacles[1]][count] = signal2
        self.signal_track[self.tentacles[2]][count] = signal3
        
        return send_addr, simulink_timestamp, signal1, signal2, signal3, int(recv_msg_decode[4])
        
        # debugging, Print the receieved attr, time and signal
        logger.info("addr: %s time: %s signal: %s %s %s" % (str(send_addr), simulink_timestamp, signal1, signal2, signal3))
    
    
        
    
    def detecting_mode(self):
        def flash_signals(send_addr, simulink_timestamp, *signal):
            self.win.addstr(3, 20, str(send_addr))
            self.win.addstr(5, 20, str(simulink_timestamp))
            self.win.addstr(7, 20, str(signal[0]))
            self.win.addstr(7, 30, str(signal[1]))
            self.win.addstr(7, 40, str(signal[2]))
            
        def flash_detection(tentacle_id, pred_value, residual, flag, flag_c):
            self.win.addstr(9, 10*(int(tentacle_id[-1])+1), str(pred_value))
            self.win.addstr(11, 10*(int(tentacle_id[-1])+1), str(residual))
            self.win.addstr(13, 10*(int(tentacle_id[-1])+1), flag, curses.color_pair(flag_c))
            self.win.clrtoeol()
        
        def pred_detect(tentacle_id, count):
            """
            display the predtion and residual on the canvas window
            Args:
                tentacle_id: str, use it to get the very model for the tentacle
                count: int,  use it as index to get the set and prediction at current time
            """
            #prior_idx+warmup60=count(now)
            #not [self.prior_idx : self.prior_idx + self.config.l_b], -> l_f steps before
            #but [- self.config.l_b]
            X_test_batch = self.signal_track[tentacle_id]\
            [self.prior_idx : self.prior_idx + self.config.l_b].reshape(1, self.config.l_b, 1) # critical if package loss
#            time1 = time.time()
            y_hat_batch = self.sub_brains[tentacle_id].model.predict(X_test_batch)
#            time2 = time.time()
#            logger.info('Signal {} prediction takes {}'.format(tentacle_id, time2-time1))
            
            set_value = round(self.signal_track[tentacle_id][count-self.config.take_point], 2)  # last point: [-1] VS self.prior_idx + self.config.l_b + self.config.l_f -1
            pred_value = round(y_hat_batch[0, self.config.l_f-self.config.take_point], 2)  # y_hat.shape is (10,1)
            residual = round(set_value - pred_value,2)
            
            if abs(residual) >= self.config.threshold[tentacle_id]:
                flag, flag_c = 'ERROR',1 # flag = '%-10s' % 'Error' # overite with 10space
                alarmer.red_alarm()
      
            else:
                flag,flag_c = 'OK',4
                alarmer.green_normal()
                
            logger.info('flag:{} residual:{} set_value:{} pred_value:{}'.format(flag, residual, set_value, pred_value))
            return set_value, pred_value, residual, flag, flag_c
                
    #        y_hat = np.reshape(y_hat, (y_hat.size,))
            
    #        print(y_hat_batch1[:,-1][0], ts1_resampled[prior_idx + 260 - 1], residual1)
    #        print('residual1 {}, residual2 {}, residual3 {}'.format(residual1, residual2, residual3))
    

            
#        plt.ion()
        # Using a loop to receive data from Simulink, Can be modified by (simulationTime/sampleTime).
        logger.info("listen start at {}".format(time.ctime())) 
        
        # first prediction always slow,  do it before the loop
        self.sub_brains['R2S1'].model.predict(np.random.random(50).reshape(1,50,1))
        self.sub_brains['R2S2'].model.predict(np.random.random(50).reshape(1,50,1))
        self.sub_brains['R2S3'].model.predict(np.random.random(50).reshape(1,50,1))
        for count in range(0, 10000, 1): 
            #----------------------------------- Data Receiving ----------------------------------------
            time_start = time.time()
            send_addr, simulink_timestamp,  signal1, signal2, signal3, _ = self.recv_track(count)
            logger.info('\n')
            logger.info('timestamp:{}'.format(simulink_timestamp))
            
            time_rec = time.time()
            
            #----------------------------------- Processing ----------------------------------------
            # Prediction, only after warm up (collect the first l_s steps to initialize the ts obj)
            # Detection, add the flag
            if simulink_timestamp > (self.config.l_b + self.config.l_f) * self.config.sample_t: #and count > self.config.l_b + self.config.l_f
                win_1_20 = 'Detection' 
                
#                for tentacle_id in self.tentacles:
#                    flash_detect(tentacle_id)

                
                set_value1, pred_value1, residual1, flag1, flag_c1 = pred_detect('R2S1',count)
                set_value2, pred_value2, residual2, flag2, flag_c2 = pred_detect('R2S2',count)
                set_value3, pred_value3, residual3, flag3, flag_c3 = pred_detect('R2S3',count)
                self.prior_idx += 1                 
                
#                logger.info('prior_idx: {} + warming steps: {} => count: {}'.format(self.prior_idx, self.config.l_b+self.config.l_f, count))
                
                
                # use pio and new incoming data to plot, 3sec delay at start time
#                if self.config.animation == "animation":
#                    animation(self.ax, count*0.1, s1=set_value1,
#                              s2=set_value2,
#                              s3=set_value3,
#                              p1=pred_value1,
#                              p2=pred_value2,
#                              p3=pred_value3)
            else:
                win_1_20 = 'Waiting for collceting enough data'
                
                set_value1, pred_value1, residual1, flag1, flag_c1 = signal1,signal1,0,'Ok',4
                set_value2, pred_value2, residual2, flag2, flag_c2 = signal2,signal2,0,'Ok',4
                set_value3, pred_value3, residual3, flag3, flag_c3 = signal3,signal3,0,'Ok',4
                
                
                
            #stack data into the trace
#            self.set_collect[tentacle_id] = np.append(self.set_collect[tentacle_id], set_value)
#            self.pred_collect[tentacle_id] = np.append(self.pred_collect[tentacle_id], pred_value)
            self.set_collect['R2S1'][count] = set_value1
            self.pred_collect['R2S1'][count] = pred_value1
            
            self.set_collect['R2S2'][count] = set_value2
            self.pred_collect['R2S2'][count] = pred_value2
            
            self.set_collect['R2S3'][count] = set_value3
            self.pred_collect['R2S3'][count] = pred_value3
            
#            if self.config.animation =="plot_cla":
#                    plot_cla(self.ax, count*0.1, s1=self.set_collect['R2S1'],
#                              s2=self.set_collect['R2S2'],
#                              s3=self.set_collect['R2S3'],
#                              p1=self.pred_collect['R2S1'],
#                              p2=self.pred_collect['R2S2'],
#                              p3= self.pred_collect['R2S3'])
            
                
            time_process = time.time()    
            
            #----------------------------------- Output ----------------------------------------    
#            if (flag1 == 'ERROR' or flag2  == 'ERROR' or flag3  == 'ERROR'):
            self.win.addstr(1, 20, '%-50s' % win_1_20)    
            flash_signals(send_addr, simulink_timestamp,  signal1, signal2, signal3)
            flash_detection('R2S1', pred_value1, residual1, flag1, flag_c1)
            flash_detection('R2S2', pred_value2, residual2, flag2, flag_c2)
            flash_detection('R2S3', pred_value3, residual3, flag3, flag_c3)
            self.win.refresh()
        
            time_output = time.time()
        
            logger.info('loop:{} time_rec:{} time_process:{} time_output:{}'.format(time_output-time_start,
                        time_rec-time_start,
                        time_process-time_rec,
                        time_output-time_process))
            # Set the condition to jump out of this loop
            if simulink_timestamp > self.config.sim_t-0.2: # #abs(count-100/self.config.sample_t) < 1e-6
                break

            
        
    def detect_protoc(self):
        """
        makes plots for detection with set and pred in 1st plot and res in 2nd plot
        
        """
        
        
        protoc_path = './test_protoc/' +'_'.join(time.ctime().split())
        
        # save the set value and pred value
        s1=self.set_collect['R2S1']
        s2=self.set_collect['R2S2']
        s3=self.set_collect['R2S3']
        p1=self.pred_collect['R2S1']
        p2=self.pred_collect['R2S2']
        p3=self.pred_collect['R2S3']
        
        
#        improvement: the way of storation: 
#        1. dict of arr   -> dataframe
#        3. arr with 6 column
        
        
#        self.collect = {}
        
        
#        np.savez(protoc_path+'.npz',
#                  s1=s1,
#                  s2=s2,
#                  s3=s3,
#                  p1=p1,
#                  p2=p2,
#                  p3=p3)
        
        detect_plot(protoc_path,s1,s2,s3,p1,p2,p3)
        
       
    def mixed_mode(self, update_traing_data=True, update_trained_model=True):
        '''
        not completely implemeted
        
        '''
        def dataArgument():
            """
            Update the training data, use aggregation
            Aggregation: add the new collected data into the training data, then update the model
            """
           
            aggregate_training_data('/home/krakenboxtud/Desktop/kraken_box/data/online_collect.npz',
                                    '/home/krakenboxtud/Desktop/kraken_box/data/new.npz')
            
            
            
    
        
    def training_mode(self, save_training_data=True):
        '''
        Get the collected data through the udp socket 
        save collected data as files
        Arg:
                   
        '''
        def update_brain(with_file=True):
            #-------------------------------------train times series prediction models---------------------------------
            for tentacle in self.tentacles:
                print('start training model for: ', tentacle)
                t = Tentacle(self.config, tentacle)
                
                if with_file:
                    # train model using ts withing the file
                    t.load_data_train()
                    
                else:
                    # train model using ts withing the memory
                    ts = pd.Series(self.signal_track[tentacle], index = self.time_track)
                    t.load_data_train(run='online', ts=ts)
                
                
                
                m = ModelStream(config)
                m.train_lstm(t)
                m.save(t)
                print('finish training model for: ', tentacle)
                
                
                
        
        def arch():
            traing_arch = 'traing_arch_'+'_'.join(time.ctime().split())
            print('saving collected data as: ', traing_arch)
            np.savez(traing_arch+'.npz',
                     t=self.time_track,
                     s1=self.signal_track[self.tentacles[0]],
                     s2=self.signal_track[self.tentacles[1]],
                     s3=self.signal_track[self.tentacles[2]])
            
       
            
        def collecting():
            #-------------------------------------collecting training data---------------------------------
            #       if abs(self.simulink_timestamp - 99.8) < 1e-3:
            #                np.savez('compare1',s = set_collect, p = pred_collect)
            for count in range(0, int(1e6), 1): 
                send_addr, simulink_timestamp,  signal1, signal2, signal3,_  = self.recv_track(count)
                print(send_addr, simulink_timestamp,  signal1, signal2, signal3)
                
                # Set the condition to jump out of this loop
                if count >= self.config.sim_t/self.config.sample_t -1 : #abs(count-3600/self.config.sample_t) < 1e-6
                    break
            else:
                logger.debug("simulation stop at count ", count)
     
        
        collecting()
        if save_training_data:
            arch()
        update_brain()
       
            




if __name__ == "__main__":
    print('Configuring...')
    config = Config('./kraken/config.yaml')
    print('establishing comunication interface...')
    print('preparing the log...')
    manager = KB_manager('stream',config)
    
    print('preparing brain...')
    manager.prepare_brain()
    print('brain prepared')
   
    manager.connect()
    print('start simulation please')
    
    # take a 0.1sec to check the mode
#    _,_,_,_,_, recv_mode = manager.recv_track(0)
#    print(recv_mode)
    recv_mode=1
    
    alarmer.setup()
    
    if recv_mode > 0:  # config.mode =='Testing'
        manager.init_layout()
        manager.detecting_mode()
        manager.udp_socket.close()
        
        manager.detect_protoc()
    
    elif recv_mode == 0: # config.mode == 'Training',
        manager.training_mode()
        alarmer.yellow_training()
        
    alarmer.destroy()
    
# =============================================================================
# #%% manually train the model
# config = Config('./kraken/config.yaml')
# t = Tentacle(config,'R2S3')
# t.load_data_train()
# m = ModelStream(config)
# m.train_lstm(t)
# =============================================================================


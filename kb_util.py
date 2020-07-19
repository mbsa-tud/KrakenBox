#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 21:26:31 2020

@author: krakenboxtud
"""
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

#    config = Config("./kraken/config.yaml")
    
#    if not config.train or not config.predict:  # not train means loading model, not predict means loading prediction
#        if not os.path.isdir('data/%s' %config.use_id):
#            raise ValueError("Run ID {} is not valid. If loading prior models or predictions, must provide valid ID.".format(_id))

    paths = ['data', 'data/%s' %_id, 'data/logs',
             'data/%s/models' %_id, 'data/%s/smoothed_errors' %_id,'data/%s/raw_errors' %_id, 'data/%s/y_hat' %_id,'data/%s/pdf_errors' %_id,
             'data/%s/inj_data' %_id, 'data/%s/inj_pic' %_id,
             'results/%s' %_id
             ]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

def setup_logging():
    '''Configure logging object to track parameter settings, training, and evaluation.
    
    Args:
        config(obj): Global object specifying system runtime params.

    Returns:
        logger (obj): Logging object
        _id (str): Unique identifier generated from datetime for storing data/models/results
    '''

    logger = logging.getLogger('kraken')
    logger.setLevel(logging.INFO)

    stdout = logging.StreamHandler(sys.stdout)
    stdout.setLevel(logging.INFO)
    logger.addHandler(stdout)

    return logger


def set_log():
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    
    handler = logging.FileHandler('./log/kraken')
    formatter = logging.Formatter('%(asctime)s  %(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger


def animate(ax, t_now, **data):
    '''
    use animation
    Args:
        data
        expected:
        s1, s2, s3, p1, p2, p3: value at this time step
    '''
    
    
    def update_points(num):
        line.set_ydata()
        return line,
    
    ani = animation.FuncAnimation(fig=fig, func=update_points, frames=100, init_func=a,interval=20,blit=True)
    
    
    for sensor in [0,1,2]:
        ax[sensor*2].plot(t_now, data['s'+ str(1+sensor)],'k.', ms=1)
        ax[sensor*2].plot(t_now, data['p'+ str(1+sensor)],'b.', ms=1)
        
        e = data['s'+ str(1+sensor)]-data['p'+ str(1+sensor)]
        ax[sensor*2+1].plot(t_now,e,'r.', ms=1)
        
        if abs(e) > 0.3:
            ax[sensor*2+1].axvline(x=t_now,c='red', alpha =0.3)
    plt.pause(1e-10)



def plot_cla(ax, t_now, **data):
    '''
    use cla, with the appended array
    Args:
        data
        expected:
        s1 -> set_collect['R2S1']
    '''
    
    for sensor in [0,1,2]:
        ax[sensor].cla()
        ax[sensor*2].cla()
        
    for sensor in [0,1,2]:
        
        ax[sensor*2].plot(data['s'+ str(1+sensor)],'k.', ms=1)
        ax[sensor*2].plot(data['p'+ str(1+sensor)],'b.', ms=1)
        
        e = data['s'+ str(1+sensor)]-data['p'+ str(1+sensor)]
        ax[sensor*2+1].plot(e,'r.', ms=1)
             
        for e in np.where(abs(e)>0.3)[0]:
            ax[sensor*2+1].axvline(x=e,c='red', alpha =0.3)
    
    plt.pause(1e-10)

def interact(ax, t_now, **data):
    '''
    use plt.ion, interactiveky plot, just with the datapoint at the exact time
    Args:
        data
        expected:
        s1, s2, s3, p1, p2, p3: value at this time step
    '''
    
    
    for sensor in [0,1,2]:
        ax[sensor*2].plot(t_now, data['s'+ str(1+sensor)],'k.', ms=1)
        ax[sensor*2].plot(t_now, data['p'+ str(1+sensor)],'b.', ms=1)
        
        e = data['s'+ str(1+sensor)]-data['p'+ str(1+sensor)]
        ax[sensor*2+1].plot(t_now,e,'r.', ms=1)
        
        if abs(e) > 0.3:
            ax[sensor*2+1].axvline(x=t_now,c='red', alpha =0.3)
    plt.pause(1e-10)
    

def detect_plot(protoc_name, s1, s2, s3, p1, p2, p3):
    '''
    Args:
        s1, s2, s3, p1, p2, p3: np array
    '''
    
    # plot the set value and pred value
    fig,ax = plt.subplots(6,1,figsize=(24,16))
    
    ax[0].plot(s1,'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's1')
    ax[0].plot(p1,'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p1')
    ax[0].legend()    
    ax[1].plot(s1-p1,'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    ax[1].axhline(y=0.3,color='k')
    ax[1].axhline(y=-0.3,color='k')
    for e in np.where(abs(s1-p1)>0.3)[0]:
        ax[1].axvline(x=e,c='red', alpha =0.3)
    
    ax[2].plot(s2,'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's2')
    ax[2].plot(p2,'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p2')
    ax[2].legend()    
    ax[3].plot(s2-p2,'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    ax[3].axhline(y=0.3,color='k')
    ax[3].axhline(y=-0.3,color='k')
    for e in np.where(abs(s2-p2)>0.3)[0]:
        ax[3].axvline(x=e,c='red', alpha =0.3)
    
    ax[4].plot(s3,'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's3')
    ax[4].plot(p3,'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p3')
    ax[4].legend()    
    ax[5].plot(s3-p3,'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    ax[5].axhline(y=0.3,color='k')
    ax[5].axhline(y=-0.3,color='k')
    for e in np.where(abs(s3-p3)>0.3)[0]:
        ax[5].axvline(x=e,c='red', alpha =0.3)
        
    fig.savefig(protoc_name)
    
    
    

    
def aggregate_training_data(current_training_file, new_collected_file):
    '''
    Arg
        current_training_data: npz file path
        new_collected: npz path
        
    Return
        current_training_data
    '''
    
    # rename the current_training_data
    shutil.copy(current_training_file, '/previous/'+current_training_file)
    
     # open the old trainng data
    current_training_data = np.load(current_training_file)
    new_collected_data = np.load(new_collected_file)
    
    # update the new collected data to the old one
    for k_old, v_old in current_training_data.items():  #not dict, numpy.lib.npyio.NpzFile
        for k_new, v_new in new_collected_data.items():
            if k_old == k_new:
                current_training_data[k_new] = np.append(current_training_data[k_old], new_collected_data[k_new])
    
    np.savez(current_training_file, current_training_data)                
#    return current_training_data
    
def save_ts(ts_resampled, pkl_name='ts.pkl', csv_name='comm.csv'):
    """
    save the ts(obj and csv), for visualization/testing
    """
    with open(pkl_name,'wb') as f:
        pickle.dump(ts_resampled, f)    
    ts_resampled.to_csv(csv_name)
    
    
def send_flag2simulink(udp_socket):
    """
    reuse the socket, udp communication session with the simulink model
    """
    #Start to send data to Simulink.
    udp_socket.sendto(bytes(1), self.simulink_addr)
    
def compare_s_p():
    """
    Read the achieved time series data
    Plot the s(set) and p(prediction)
    
    """

    arch = np.load(self.protoc_name+'.npz')
    fig,ax = plt.subplots(6,1,figsize=(8,12))
    ax[0].plot(arch['s1'],'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's1')
    ax[0].plot(arch['p1'],'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p1')
    ax[0].legend()    
    ax[1].plot(arch['s1']-arch['p1'],'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    
    ax[2].plot(arch['s2'],'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's2')
    ax[2].plot(arch['p2'],'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p2')
    ax[2].legend()    
    ax[3].plot(arch['s2']-arch['p2'],'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    
    ax[4].plot(arch['s3'],'k',ls='--',lw=1,marker='*',mec='k',mew=1,mfc='k',ms=2,label = 's3')
    ax[4].plot(arch['p3'],'b',ls='--',lw=1,marker='*',mec='b',mew=1,mfc='b',ms=2,label = 'p3')
    ax[4].legend()    
    ax[5].plot(arch['s3']-arch['p3'],'r',ls='--',lw=1,marker='*',mec='r',mew=1,mfc='r',ms=2)
    
def alarm():
    os.system('spd-say "error detected"')
    
def arr2ts(signal_id='R2S1'):
    """
    encapseln into ts object, Index of series: simulink_time->pd.time_delta
	    resample the signals according to the timestamp, standard is 0.1sec
    Arg:
        signal_nb: 
    """
    idx = self.time_track
    t_idx = list(map(lambda s: pd.Timedelta(s, unit='s'), idx))    # encapseln the timestamp into timedelta obj
    ts = pd.Series(self.signal_track[signal_nb], index = t_idx)   # time series obj with pd.period_range as index
    # resample with 0.1sec
    # sample time must be big enough, the span must be big enough to cover some points at least, otherwise NaN value appears
    ts_resampled = ts.resample('0.1S').mean()  
    ts_resampled = ts_resampled.fillna(method = 'ffill') #ts.index = list(range(len(ts))) 
    return ts_resampled
        
def check_resample():
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


        
#        print("----Listening data from simulink udp block----")
#        print("Please open the Simulink file under the demo system working directory")
#        print("The program is waiting until you run the Simulink file.")
    



def back_to_simulink():
    pass
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
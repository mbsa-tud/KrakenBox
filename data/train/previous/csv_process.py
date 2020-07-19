import pandas as pd
import numpy as np

import scipy.io as sio
#import mat4py


import pickle

"""

load the golden run data in xlsx form or csv form saved locally,
transform them into ts obj
"""


def mat2xls():
    data = sio.loadmat('R2S1_2500sec.mat',squeeze_me=True,struct_as_record=False)
        # mat = hdf5storage.loadmat('R2S1_2500sec.mat')
        
        # with h5py.File('R2S1_2500sec.mat','r') as f:
        #     fkey = f.keys()
        
        # data = mat4py.loadmat('R2S1_2500sec.mat')
        
        
        
        # dfexcel = pd.read_excel("{}.xlsx".format('R2S1_2500'))  # only 1048574 data






def csv2pkl(time_csv, *ts_csv):
    '''
    In: 1.csv 2.csv....n.csv   time.csv
    Out: 1.pkl, 2.pkl,...n.pkl
    '''
    
    idx = np.array(pd.read_csv(time_csv)).reshape(-1)
    t_idx = list(map(lambda s: pd.Timedelta(s, unit='sec'), idx))
    
    for ts in ts_csv:
        t_np = np.array(pd.read_csv(ts)).reshape(-1)
        t_s = pd.Series(t_np,index = t_idx)
        
        t_s = t_s.resample('0.1S').mean()
        t_s = t_s.fillna(method='ffill')
        
        with open('r2s{}_2500.pkl'.format(ts[0]),'wb') as f:
            pickle.dump(ts, f)
        






if __name__ == '__main__':
    f_name = ['../2500sec/'+f for f in ('time.csv', '1.csv', '2.csv', '3.csv')]
    csv2pkl(tuple(f_name))
   
    
    

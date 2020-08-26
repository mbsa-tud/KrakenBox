import pandas as pd
import numpy as np

import scipy.io as sio
import mat4py

import h5py
import hdf5storage


import pickle

"""

load the golden run data in xlsx form or csv form saved locally,
transform them into ts obj
"""


if __name__ == '__main__':
   
    
    data = sio.loadmat('R2S1_2500sec.mat',squeeze_me=True,struct_as_record=False)
    # mat = hdf5storage.loadmat('R2S1_2500sec.mat')
    
    # with h5py.File('R2S1_2500sec.mat','r') as f:
    #     fkey = f.keys()
    
    # data = mat4py.loadmat('R2S1_2500sec.mat')
    
    
    
    # dfexcel = pd.read_excel("{}.xlsx".format('R2S1_2500'))  # only 1048574 data
    
    
    r2s1 = np.array(pd.read_csv("{}.csv".format('1'))).reshape(-1)
    r2s2 = np.array(pd.read_csv("{}.csv".format('2'))).reshape(-1)
    r2s3 = np.array(pd.read_csv("{}.csv".format('3'))).reshape(-1)
    idx = np.array(pd.read_csv("{}.csv".format('time'))).reshape(-1)
    t_idx = list(map(lambda s: pd.Timedelta(s, unit='sec'), idx))
    
    r2s1 = pd.Series(r2s1,index = t_idx)
    r2s1 = r2s1.resample('0.1S').mean()
    r2s1 = r2s1.fillna(method='ffill')
    with open('r2s1_2500.pkl','wb') as f:
        pickle.dump(r2s1, f)
        
    r2s2 = pd.Series(r2s2,index = t_idx)
    r2s2 = r2s2.resample('0.1S').mean()
    r2s2 = r2s2.fillna(method='ffill')
    with open('r2s2_2500.pkl','wb') as f:
        pickle.dump(r2s2, f)
        
    r2s3 = pd.Series(r2s3,index = t_idx)
    r2s3 = r2s3.resample('0.1S').mean()
    r2s3 = r2s3.fillna(method='ffill')
    with open('r2s3_2500.pkl','wb') as f:
        pickle.dump(r2s3, f)

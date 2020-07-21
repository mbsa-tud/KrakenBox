import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import os
import logging
import pickle

logger = logging.getLogger('kraken')


class Channel:
    def __init__(self, config, chan_id):
        """
        File interface.
        Load and reshape channel values (predicted and actual).

        Args:
            config (obj): Config object containing parameters for processing
            chan_id (str): channel id

        Attributes:
            id (str): channel id
            config (obj): see Args
            X_train (arr): training inputs with dimensions
                [timesteps, l_s, input dimensions)
            X_test (arr): test inputs with dimensions
                [timesteps, l_s, input dimensions)
            y_train (arr): actual channel training values with dimensions
                [timesteps, n_predictions, 1)
            y_test (arr): actual channel test values with dimensions
                [timesteps, n_predictions, 1)
            train (arr): train data loaded from .npy file
            test(arr): test data loaded from .npy file
        """

        self.id = chan_id
        
        self.config = config
        
        self.train = None
        self.test = None
        
        self.X_train = None
        self.y_train = None
        
        self.X_test = None
        self.y_test = None
        self.y_hat = None
        

    def shape_data(self, arr, train=True):
        """Shape raw input streams for ingestion into LSTM. 
        config.l_s
            specifies the sequence length of prior timesteps fed into the model 
            at each timestep t.

        Args:
            arr (np array): array of input streams with
                dimensions [timesteps, 1, input dimensions]
            train (bool): If shaping training data, this indicates
                data can be shuffled
        """

        data = []
        for i in range(len(arr) - self.config.l_s - self.config.n_predictions):
            data.append(arr[i:i + self.config.l_s + self.config.n_predictions])
        data = np.array(data)

        assert len(data.shape) == 3

        if train:
            np.random.shuffle(data)
            self.X_train = data[:, :-self.config.n_predictions, :]
            self.y_train = data[:, -self.config.n_predictions:, 0]  # sensor value is at position 0
        else:
            self.X_test = data[:, :-self.config.n_predictions, :]
            self.y_test = data[:, -self.config.n_predictions:, 0]  # sensor value is at position 0




    def load_golden_run(self, need_info=False, retrieve='2500sec'):
        """
        Load the saved injection free data from local file(xlsx and ts-obj).
        
        
        
        Parameters
        ----------
        need_info: plot the data
        
        retrieve: two retrieve options: 100sec and 2500sec, to verify that 
        more data can train more accuate LSTM

        Returns
        -------
        ts : series
            DESCRIPTION.

        """
      
        try:
            # Interface for the xlsx file
            if retrieve == '100sec':
                dfTrain = pd.read_excel(os.path.join('.',"data","train","{}.xlsx".format(self.id)))  # two col dataframe,col0 is time stamp, col1 is ts
                
                # info about the df read from the excel
                if need_info:
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
                
                
                # info about the ts
                if need_info:
                    ts.head()
                    plt.figure(figsize=(30,10))
                    plt.plot(ts)
            
            
        
            if retrieve == '2500sec':
                with open(os.path.join('.',"data","train","{}_2500.pkl".format(self.id.lower())),'rb') as f:
                    ts = pickle.load(f)
                    
                # info about the ts
                if need_info:
                    ts.head()
                    plt.figure(figsize=(30,10))
                    plt.plot(ts)
                    
            return ts
            
        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

    def load_data_train(self,need_info=False):
       """
       Load the training data which is injection free.
       call the load_golden_run and then shape the data for training
       1 load golden run
       2 shape data
       """
       try:
           train_ts = self.load_golden_run(need_info)
           self.train = np.array(train_ts).reshape(-1,1)
           logger.info('before shaping:  {}'.format(self.train.shape))
           
       except FileNotFoundError as e:
           logger.critical(e)
           logger.critical("Source data not found, may need to add data to repo: <link>")
           
       self.shape_data(self.train)
       logger.info('after shaping: {}{} '.format(self.X_train.shape,self.y_train.shape))
    
        
    def load_data_robot_test(self, chan_id):
        '''
        load one injected ts, in form .npy file
        then shape it for testing
        '''
        
        #parse information from all the filenames in a folder, link the information to the file
        
#        s = [f.split('.')[0] for f in os.listdir('/data/test/inj')]
#        p = re.compile(r'\_')
#        (m,ev,ef) = p.split(s)
#        self.inject = [{'method':m,'event':ev,'effect':ef} for  ]
        
        # self.test = np.load(os.path.join('data',self.config.use_id,'inj_data','{}{}_{}_{}_{}_{}.npy'.format(config.use_id, anomaly, event, effect, magnitude, times)))
        self.test = np.load(os.path.join('data',self.config.use_id,'inj_data','{}.npy'.format(self.id)))
        self.test = self.test.reshape(-1,1)
        self.shape_data(self.test,train=False)
        
        self.id = chan_id
    
    

            
            
        
            
                
    
    def load_data(self):
        """
        back up, not useful by now
        Load train and test data from local npy.
        """
        try:
            self.train = np.load(os.path.join("data", "train", "{}.npy".format(self.id)))
            self.test = np.load(os.path.join("data", "test", "{}.npy".format(self.id)))
            logger.info('before shaping:  {}{}'.format(self.train.shape,self.test.shape))

        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")

        self.shape_data(self.train)
        self.shape_data(self.test, train=False)
        
        logger.info('after shaping: {}{}{}{} '.format(self.X_train.shape,self.y_train.shape,self.X_test.shape,self.y_test.shape))
            
    

        
        
        
    def load_data_robot_train(self):
        """
        back up, not useful by now, currently not functioning
        Load train data from local excel.
        """
        try:
            #read the xlsx, transform in np
            dfTrain = pd.read_excel(os.path.join("data","train","{}.xlsx".format(self.id)))
            npTrain = np.array(dfTrain.iloc[:,1]).reshape(-1,1)  # make it 2 dim
            numPoints = len(npTrain)  #130,000
            
            #downsampling, evert 13 points
            self.train = np.array([npTrain[int(numPoints/10000*i)] for i in range(10000)]).reshape(-1,1)  
            logger.info('before shaping:  {}'.format(self.train.shape))
            
        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")
            
        self.shape_data(self.train)
        
        logger.info('after shaping: {}{} '.format(self.X_train.shape,self.y_train.shape))
        
        
        
        
    def load_data_10times(self, anomaly, effect, magnitude):
        '''
        back up, not useful by now, currently not functioning
        
         #1round = n times, during 1 round is all elements fixed, e.g. effect=0.1, magnitude = 0.1
        cd F:\code\py\pyKraken\kraken\kraken-master\kraken-master\data\2020-03-05_18.49.12\inj_data
        glob.glob("R2S1offset_3_0.1_0.1_*.npy")
        '''
        
        round_name = self.id  + anomaly + '_' + effect + '_' + magnitude + '_'
        round_list = glob.glob(round_name + '*.npy')
        
        
        
        
        
        
        
    def prepare_injected_data(self):
        """
        back up, not useful by now, currently not functioning
        
        """
        try:
            self.test = {}
            
            for inj in self.inject:
                dfTest = pd.read_excel(os.path.join("data","test","{}_{}_{}_{}.xlsx".format(self.id, inj['method'],inj['event'],inj['effect']))) #R2S1_offset_0.30_1.xlsx
                npTest = np.array(dfTest.iloc[:,1]).reshape(-1,1)
                numPoints = len(npTest)
                
                
                self.test["{}_{}_{}_{}".format(self.id, self.event, self.effect)] = np.array([npTest[int(numPoints/10000*i)] for i in range(10000)]).reshape(-1,1) #resampling(downsamplinng), index= 0.0001-1 * numPoints
                logger.info('before shaping:  {}'.format(self.test.shape))
            
        except FileNotFoundError as e:
            logger.critical(e)
            logger.critical("Source data not found, may need to add data to repo: <link>")
            
        self.shape_data(self.test,train=False)
        logger.info('after shaping: {}{} '.format(self.X_test.shape,self.y_test.shape))
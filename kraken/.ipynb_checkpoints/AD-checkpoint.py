import os
import numpy as np
import pandas as pd
from datetime import datetime as dt
import logging

from kraken.helpers import Config
from kraken.errors import Errors
import kraken.helpers as helpers
from kraken.channel import Channel
from kraken.modeling import Model

logger = helpers.setup_logging()

class AD():
    def __init__(self, labels_path = '', result_path = '', use_id='', config_path = 'config.yaml'):
        
        self.labels_path = labels_path
        self.results = []
        self.results_df = None
        self.chan_df = None
        
        self.result_tracker = {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0}
        
        self.config = Config(config_path)
        self.y_hat = None
        
        self.id = use_id
#        self.id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
#        if not self.config.predict and self.config.use_id:
#            self.id = self.config.use_id
#        else:
#            self.id = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
            
        helpers.make_dirs(self.id)
        
        self.result_path = result_path
        
         # add logging FileHandler based on ID
        hdlr = logging.FileHandler('data/logs/%s.log' % self.id)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        
        
#        if self.result_path:
#            self.chan_df = pd.read_csv(labels_path)
#        else:
#            chan_ids = [x.split('.')[0] for x in os.listdir('data/test/')]
#            self.chan_df = pd.DataFrame({"chan_id": chan_ids})
        self.chan_df = pd.read_csv(labels_path)
            
        logger.info("{} channels found for processing."
                    .format(len(self.chan_df)))
        
    
    
    

    
    #select a group of channels for one experiment
    def select_chans(self,chan='R2S1',anomaly='offset', effect=[], magnitude=[]):
        """
        select channels for preparation of respective experiments
        ad.chan_df, the channel for prediction and detection
        
        """
        
#        select chan_id
#        chan_ids = set(ad.chan_df['chan_id']) #{'R2S1','R2S2'}
#        for chan in chan_ids:
        self.chan_df = self.chan_df[self.chan_df['chan_id'] == chan]
        self.chan_df = self.chan_df[self.chan_df['anomaly'] == anomaly]
        
        # select magnitude
        mag_l = []
        for mag in magnitude:
            mag_l.append(self.chan_df[self.chan_df['magnitude'] == mag])
        self.chan_df = pd.concat(mag_l)
        
        # select effect
        eff_l = []
        for eff in effect:
            eff_l.append(self.chan_df[self.chan_df['effect'] == eff])
        self.chan_df = pd.concat(mag_l)
            
            
        
        
    
    def evaluate_sequences(self,errors,label_row):
        result_row = {
                'false_positives': 0 ,
                'false_negatives': 0,
                'true_positives': 0,
                'fp_sequences':[],
                'tp_sequences':[],
                'num_true_anoms': 0}
        
        matched_true_seqs = []
        
        label_row['anomaly_sequences'] = eval(label_row['anomaly_sequences'])  # eval() arg is a string   [[2000,2050],[3000,3050],[5000,5050]]
        result_row['num_true_anoms'] += len(label_row['anomaly_sequences'])
        
        
        
        if len(errors.E_seq) == 0: # detect zero errors
            result_row['false_negatives'] = result_row['num_true_anoms']
            
        else: # compare label_row['anomaly_sequences']  with errors.E_seq
            true_indices_grouped = [list(range(e[0],e[1]+1)) for e in label_row['anomaly_sequences']]  # [[2000-2050],[3000-3050],[5000-5050]]
            true_indices_flat = set([i for group in true_indices_grouped  for i in group])   #  {2000-2050,3000-3050,5000-5050}
            
            for e_seq in errors.E_seq:   # [2010-2060]
                i_anom_predicted = set(range(e_seq[0], e_seq[1]+1))   #{2010-2060}
                
                matched_indices = list(i_anom_predicted & true_indices_flat)  #intersection of current testing e_seq and the labeled err seq, [2010-2050]
                valid = True if len(matched_indices) > 0 else False
                
                if valid:
                    result_row['tp_sequences'].append(e_seq)   
                    
                    true_seq_index = [i for i in range(len(true_indices_grouped)) if     # i->1,2,3,  true_seq_index = 1
                                      len(np.intersect1d(list(i_anom_predicted), true_indices_grouped[i])) > 0]
                    
                    if not true_seq_index[0] in matched_true_seqs:
                        matched_true_seqs.append(true_seq_index[0])
                        result_row['true_positives'] +=1
                else:
                    result_row['fp_sequences'].append([e_seq[0],e_seq[1]])
                    result_row['false_positives'] +=1
            result_row["false_negatives"]  = len(np.delete(label_row['anomaly_sequences'], matched_true_seqs, axis = 0)) #delete the matched_true_seqs rows in the [[2000-2050],[3000-3050],[5000-5050]] 
            
            
        for key, value in result_row.items():
            if key in self.result_tracker:
                self.result_tracker[key] += result_row[key]

        return result_row
    
       
                
                
                
                
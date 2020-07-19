#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 11:22:58 2020

@author: krakenboxtud
"""
import logging
import yaml
import json
import sys
import os
import numpy as np

logger = logging.getLogger('kraken')
sys.path.append('../kraken')


class Config:
    """
    Loads parameters from config.yaml into global object

    """

    def __init__(self, path_to_config):

        self.path_to_config = path_to_config

        if os.path.isfile(path_to_config):
            pass
        else:
            self.path_to_config = '../{}'.format(self.path_to_config)
        
        # read the yaml into dict
        with open(self.path_to_config, "r") as f:
            self.dict = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # add the yaml info to attr
        for k, v in self.dict.items():
            setattr(self, k, v)
            
                
    def config_inject(self):
        #injection configuration
        self.range_sensor_id  = ["R2S1","R2S2","R2S3"]
        self.range_method = ['offset']
        self.range_event = [1]
        self.range_effect = np.arange(2,3,1)
        self.range_magnitude = [round(m,1) for m in np.linspace(0.1,1,10)]
        self.length_circle = 100
        self.nb_circle = len(self.range_magnitude) if self.dict['element']=='magnitude' else len(self.range_effect)
        self.inj_experiments = len(self.range_sensor_id)*len(self.range_method)* self.length_circle*self.nb_circle
        
        # run time configuration
        self.current_ts = self.range_sensor_id[2]

    def build_group_lookup(self, path_to_groupings):

        channel_group_lookup = {}

        with open(path_to_groupings, "r") as f:
            groupings = json.loads(f.read())

            for subsystem in groupings.keys():
                for subgroup in groupings[subsystem].keys():
                    for chan in groupings[subsystem][subgroup]:
                        channel_group_lookup[chan["key"]] = {}
                        channel_group_lookup[chan["key"]]["subsystem"] = subsystem
                        channel_group_lookup[chan["key"]]["subgroup"] = subgroup

        return channel_group_lookup





    

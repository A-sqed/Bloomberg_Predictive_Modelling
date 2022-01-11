################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu 
################################################################################

# WIP: Current DF return begins 1-month after start date/ need to decrease
#   for shorter analysis windows

import datetime
import json
import logging
import operator
import os
import pathlib
import sys
import time

import numpy as np
import streamlit as st
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from tqdm import tqdm

path = pathlib.Path(__file__).parent.absolute()
logger = logging.getLogger('_preprocess_xlsx')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(str(path)+'\\logs\\_preprocess.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
os.environ['NUMEXPR_MAX_THREADS'] = '16'
################################################################################
# Pre-Processing of XLSX Into Pandas Dataframe
################################################################################


class _preprocess_xlsx:

    def __init__(self,
                xlsx_file,
                target_col,
                forecast_list=[1, 3, 7, 15, 30],
                momentum_list = [],
                split_percentage = .20,
                sequential = False,
                momentum_X_days = [5, 10, 15],
                momentum_Y_days = 30,
                ):
        
        logger.info(" Preprocessing, using XLSX: {} and target(s): {}".format(xlsx_file, target_col))
       
        try:
            assert (pathlib.Path(xlsx_file)).is_file()
        except Exception as e:
            logger.debug(" Missing XLSX File")

        self.df = pd.read_excel(xlsx_file)
        self.label_encoder = preprocessing.LabelEncoder()
        self.target_col = target_col
        self.forecast_list = forecast_list
        self.momentum_list = momentum_list
        self.split_percentage = split_percentage
        self.sequential = sequential
        self.momentum_X_days = momentum_X_days
        self.momentum_Y_days = momentum_Y_days
   
        self._add_custom_features()
        
        self.complete_data = self.df.dropna().copy()

        self.X = self.complete_data.drop([ self.target_col,'Dates'], axis=1)
        
        self.feature_cols = self.X.columns

        self.Y = self.complete_data[ self.target_col]

        logger.debug(" Splitting Test and Training Data")

        self.X_train, self.X_test, self.Y_train, self.Y_test = \
        train_test_split(self.X,
                        self.Y,
                        test_size=self.split_percentage,
                        shuffle=self.sequential)
        
        self._find_entropy_of_feature(self.Y)

        # Encode Target
        logger.debug(" Encoding target training and test data")
        self.Y_encoded = self.label_encoder.fit_transform(self.Y)
        self.Y_train_encoded = self.label_encoder.fit_transform(self.Y_train)
        self.Y_test_encoded = self.label_encoder.fit_transform(self.Y_test)
        
################################################################################
# Class Methods 
################################################################################

    # Find - (sum_i)( p_i * log_2 (p_i) ) for each i
    def _find_entropy_of_feature(self, df_target_col):
        target_counts = df_target_col.value_counts().astype(float).values
        total = df_target_col.count()  
        probas = target_counts/total
        entropy_components = probas * np.log2(probas)
        entropy = (- entropy_components.sum())
        logger.info(" Entropy of target feature is {}".format(entropy))
        return entropy

    # H(target) - H(target | info > thresh) - H(target | info <= thresh)
    def _information_gain(self, info_column, target_col, threshold=.5):

        data_above_thresh = self.df[self.df[info_column] > threshold]
        data_below_thresh = self.df[self.df[info_column] <= threshold]
        
        entropy_target_col = self._find_entropy_of_feature(self.df[target_col])
        entropy_above = self._find_entropy_of_feature(data_above_thresh[target_col])
        entropy_below = self._find_entropy_of_feature(data_below_thresh[target_col])

        ct_above = data_above_thresh.shape[0]
        ct_below = data_below_thresh.shape[0]
        tot = float(self.df.shape[0])
        IG =  entropy_target_col - entropy_above*ct_above/tot - entropy_below*ct_below/tot
        logger.info(" IG of {} and {} at threshold {} is {}").format(info_column,
                                                              target_col,
                                                              threshold,
                                                              IG)
        return IG

    def best_threshold(self, info_column, target_col, criteria=_information_gain):
        maximum_ig = 0
        maximum_threshold = 0

        for thresh in self.df[info_column]:
            IG = criteria(info_column, target_col, thresh)
            if IG > maximum_ig:
                maximum_ig = IG
                maximum_threshold = thresh
                
        return (maximum_threshold, maximum_ig)

################################################################################
# Customize Import Data For Bloomberg 
################################################################################

    def _add_custom_features(self):
        try:
            self.df['EARN_DOWN'] = self.df['EARN_DOWN'].astype(np.float16)
        except ValueError as e:
            logger.debug(" EARN_DOWN Not Included in XLSX")
            print(e)
            return
        
        try:
            self.df['EARN_UP'] = self.df['EARN_UP'].astype(np.float16)
        except ValueError as e:
            logger.debug(" EARN_UP Not Included in XLSX")
            print(e)
            return

        # Add new momentum features
        self._add_momentum(self.momentum_list, 
                          self.momentum_X_days, 
                          self.momentum_Y_days)

    def _add_momentum(self, momentum_list, momentum_X_days, momentum_Y_days):
        
        logger.info(" momentum_list: {}".format(momentum_list))
        
        if not momentum_list: 
            return
        
        if momentum_X_days == None: momentum_X_days == self.momentum_X_days
        if momentum_Y_days == None: momentum_Y_days == self.momentum_Y_days
        
        else:
            for item in momentum_list:
                for win in momentum_X_days:
                    new_item = str(item) + "_" + str(win) + "day_rolling_average"
                    self.df[new_item] =  \
                    self.df[item].rolling(window=win).mean() -  \
                    self.df[item].rolling(window=momentum_Y_days).mean() /  \
                    self.df[item].rolling(window=momentum_Y_days).mean()
                    logger.info(" Adding new col for {}".format(new_item))

     # Add column to df with net change from day to dh in future
    def _change_over_days(self, dh=None):
        if dh == None:
            for dh in self.forecast_list:

                logger.debug(" Processing for {} days ahead".format(dh))
                self.target_change = '{}_{}_Day_Change'.format(self.target_col, dh)
                self.df[str(self.target_change)] = \
                    self.df[self.target_col] - \
                    self.df[self.target_col].shift(dh)
        else:
            for d in dh:

                logger.debug(" Processing for {} days ahead".format(d))
                self.target_change = '{}_{}_Day_Change'.format(self.target_col, d)
                self.df[str(self.target_change)] = \
                    self.df[self.target_col] - \
                    self.df[self.target_col].shift(d)
                    
################################################################################
# Returns
################################################################################    
    
    # Add column to df with actual value of target in days ahead 
    def _return_data_with_dh_actuals(self, days_ahead = None, target = None):
        max_forecast = days_ahead      
        if max_forecast == None: max_forecast  = max(self.forecast_list)
        days_to_go = list(range(1,max_forecast+1))
        data_dict, X_dict, Y_dict = {}, {}, {}
        
        for dh in days_to_go:
            
            temp_data = self.complete_data.copy()
            
            # Add predicative column for days ahead (d)
            forecast_name = '{0}_{1}D_Ahead_Actual'.format(self.target_col, dh)
            #logger.info("Adding {0} ".format(forecast_name))
            
            temp_data[forecast_name] = temp_data[self.target_col].shift(dh)
            temp_data = temp_data.dropna()
            Y_dict[dh] = temp_data[[forecast_name]]
            
            data_dict[dh] = temp_data
            
            temp_data = temp_data.drop([forecast_name, 'Dates'], axis=1)       
            X_dict[dh] = temp_data
            
        return data_dict, X_dict, Y_dict

    def _set_feature_names(self, new_features):
        self.feature_cols = new_features

    def _set_target_col(self, target_col):
        self.target_col = target_col

    def _print_target_col(self):
        print(tabulate(self.target_col))
        
    def _return_target_col(self):
        return self.target_col

    def _return_feature_names(self):
        return self.feature_cols
    
    def _return_complete_data(self):
        return self.complete_data

    def _return_test_and_train_data(self):
        return self.X_train, self.X_test, self.Y_train, self.Y_test

    def _return_Y_encoded(self):
        return self.Y_encoded, self.Y_train_encoded, self.Y_test_encoded

    def _return_dataframe(self):
        return self.complete_data
    
    def _return_X_Y_dataframe(self):
        return self.X, self.Y
    
    def _return_forecast_list(self):
        return self.forecast_list

    def __str__(self):
        return print(tabulate(self.df, headers='keys', tablefmt='psql'))
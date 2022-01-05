################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu 
################################################################################

import datetime, os
import logging
import random
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import pylab as pl
import seaborn as sns
from tqdm import tqdm
from sklearn import linear_model, metrics, preprocessing
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sktime.forecasting.model_selection import SingleWindowSplitter
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from xgboost import XGBRegressor, plot_importance, plot_tree
path = os.path.dirname(__file__)
import _preprocessing
path = os.path.dirname(__file__)

#Debug and logger
warnings.filterwarnings('ignore')
logger = logging.getLogger('_pred_power')
logger.setLevel(logging.INFO)
#handler = logging.FileHandler(path+'/model.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


################################################################################
# Fit and Predict a Chosen Model 
################################################################################

class _build_model:
    def __init__(self, 
                pipeline,
                model_name='XGBoost',
                estimators = 1000,
                random_state = random.seed(),
                max_forecast=30):
        
        try:
            if model_name != None: pass
        except ValueError:
            self.logger.debug(" Must specify a model type")

        self.pipeline = pipeline
        self.timeseries_splits = 5
        self.forecast_horizon = max(pipeline._return_forecast_list())
        self.scaler = MinMaxScaler(feature_range=(0,1))
        
        # Return Processesed Data 
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            pipeline._return_test_and_train_data()
        self.Y_encoded, self.Y_train_encoded, self.Y_test_encoded = \
            pipeline._return_Y_encoded() 
        self.X_df, self.Y_df = pipeline._return_X_Y_dataframes()
        
        logger.info(" Selecting model {}".format(model_name))
        
        # Default Models
        self.models_available = {'CART':DecisionTreeRegressor(
                                                random_state=random_state),
                                 'XGBoost':XGBRegressor(
                                                n_estimators=estimators, 
                                                random_state=random_state),
                                 'AdaBoostClassifier':AdaBoostClassifier(
                                                n_estimators= 30, 
                                                learning_rate = 0.50,
                                                random_state= random_state),
                                 'LogisticRegression':linear_model.LogisticRegression(),
                                 'Quadratic Regression':make_pipeline(
                                                PolynomialFeatures(3), 
                                                Ridge()),    
                                 'KNeighborsRegressor':KNeighborsRegressor(
                                                n_neighbors=2)                       
                                }
        self.model = self.models_available.get(model_name)      
        tss = TimeSeriesSplit(n_splits=self.timeseries_splits).split(self.X_df)
           
        logger.info(" Fitting model: \n {}".format(self.model))
        self.model.fit(self.X_train, self.Y_train)

        # Predict
        logger.info(" Predicting with model: \n {}".format(self.model))
        self.model_preds = self.model.predict(self.X_test)
        self.scores = cross_val_score(self.model, self.X_df, self.Y_df, cv=tss)
        logger.info(" Mean cross-validataion Accuracy: %0.2f (+/- %0.3f)" % (self.scores.mean(), 
                                                                             self.scores.std()))
        
        self.features_over_time_dict = {}
        
        #_print_accuracy(model_name, self.Y_df, self.model_preds)

################################################################################
# Class Methods 
################################################################################
    def predictive_power(self, forecast_range=30):
        
        all_data, X_data, Y_data = self.pipeline._return_data_with_dh_actuals(forecast_range)
                
        pipeline_target = self.pipeline._return_target_col()
        
        feats = pd.DataFrame(data=X_data[forecast_range])

        predictors_df = pps.predictors(feats, y=pipeline_target)
  
        predictors_df = predictors_df[predictors_df['ppscore'] > 0.5]
  
        f, ax = plt.subplots(figsize=(16, 5))
        ax.set_title("Predicative Power for {0} at {1} Days".format(pipeline_target, forecast_range))
        sns.barplot(data=predictors_df, y="x", x="ppscore",palette="rocket")
        f.show() 
                 
    def _feature_importance(self, forecast_range=30, plot=True):

        pipeline_target = self.pipeline._return_target_col()
        
        all_data, X_data, Y_data = \
            self.pipeline._return_data_with_dh_actuals(days_ahead=forecast_range,
                                                       target = pipeline_target)
            
        for day in range(1,(forecast_range+1)):

            X_feature_cols = X_data[day].columns

            # Scale and add back to df with column names
            X_scaled = self.scaler.fit_transform(X_data[day])
            X_scaled = pd.DataFrame(X_scaled, columns=X_feature_cols)
            
            logger.info(" Fitting Scaled Model: Day {}".format(day))
            model_scaled = self.model.fit(X_scaled, Y_data[day])
            importances = model_scaled.feature_importances_
            
            feats = {}
            feats_model_by_day = {}
            threshold = 0.05

            if day not in self.features_over_time_dict.keys():
                self.features_over_time_dict[day] = \
                    { feature : None for feature in X_feature_cols }

            for feature, importance in zip(X_feature_cols, importances):
                
                if importance > threshold:
                    feats[feature] = importance
                                
                if self.features_over_time_dict[day][feature] == None:
                    self.features_over_time_dict[day][feature] = [importance]
                
                else:
                    self.features_over_time_dict[day][feature].append(importance)
        
            feats = sorted(feats.items(), key=lambda x: x[1],  reverse=True) 
            feats = dict(feats)    
            feats_model_by_day[day] = feats
            
            if plot == True:
                if day == forecast_range:
                    for target, feature in feats_model_by_day.items():
                        width = 1
                        keys = feature.keys()
                        values = feature.values()
                        if target == day:
                            f, ax = plt.subplots(figsize=(16, 5))
                            ax.set_title("Feature Importance for {0} Day Forecast: {1}".format(target, pipeline_target))
                            sns.barplot(y=list(keys), x=list(values), palette="rocket")
                            f.show()     

    def _feature_importance_over_time(self, plot=True, forecast_range=30, usefulness_threshold=0.2):
        
        pipeline_target = self.pipeline._return_target_col()
        
        if not self.features_over_time_dict:
            self._feature_importance(forecast_range, plot=False)
        
        list_of_days_to_forecast = list(range(1,forecast_range+1))
        df = pd.DataFrame()
        column_names = []
        
        for day in list_of_days_to_forecast:
            if day == 1:
                df = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                column_names = list(df.columns)
            else:
                feature_dict = pd.DataFrame.from_dict(self.features_over_time_dict[day])
                df = df.append(feature_dict, ignore_index = True)
        df['day'] =  list_of_days_to_forecast      

        remove_list = []
        
        for feat in column_names: 
            usefulness = df[feat].max()

            if usefulness < usefulness_threshold:
                logger.info("feat: {0}, usseful-max: {1}".format(feat, usefulness))
                df.drop([feat], axis=1)
                remove_list.append(feat)

        for x in remove_list:
            column_names.remove(x)

        sns.set_palette(sns.color_palette("rocket"))
        f, ax = plt.subplots(figsize=(14, 6))
        for feat in column_names:
            sns.lineplot(data=df, 
                        x='day', 
                        y=df[feat], 
                        dashes=False).set_title('{0} Feature Importance By Time'.format(pipeline_target))
        sns.set_style("whitegrid")
        ax.grid(True)
        ax.set(xlabel='Days Out', ylabel='Predictive Importance')
        ax.set(xticks= list(range(1,forecast_range+1)))
        ax.legend(column_names)
        if plot:
            f.show()        

    def _return_preds(self):
        return self.model_preds

    def _return_model(self):
        return self.model
    
################################################################################
# WIP
################################################################################
"""
    def _print_accuracy(model_name, Y_test_encoded, model_preds):
        
        if model_name == 'Quadratic Regression': 
            accuracy = metrics.mean_squared_error(Y_test_encoded, 
                                                            model_preds)
            
            logger.info("Accuracy of model {}: {:.2%}".format(model_name, 
                                                        accuracy))
            return '{:.2%}'.format(accuracy)
        
        else:
            accuracy = metrics.accuracy_score(Y_test_encoded, 
                                            model_preds)
            
            logger.info("Accuracy of model {}: {:.2%}".format(model_name, 
                                                        accuracy))
            return '{:.2%}'.format(accuracy)

"""
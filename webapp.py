################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu 
################################################################################
import os

import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)
import datetime
import logging
import pathlib
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import pylab as pl
import seaborn as sns
from PIL import Image

import _models
import _preprocessing

session_state = st.session_state

path = pathlib.Path(__file__).parent.absolute()
os.environ['NUMEXPR_MAX_THREADS'] = '16'
banner = Image.open(str(path)+'\\_img\\arrow_logo.png')
st.sidebar.image(banner, caption='ML Predictions From Bloomberg Data Pulls', width=200,)
pd.set_option('display.max_columns', None) 
global counter
counter = 0
pipeline = None
complete_data = None
session_state.model_chooser = None

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_info = logging.getLogger(__name__)
log_info.setLevel(logging.INFO)
handler = logging.FileHandler(str(path)+'/logs/_main.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

################################################################################
#  Helper Functions
################################################################################
session_state.model_loaded = False

def _max_width_():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

# Set invert colors for for fixed income 
def color_arrow(val):
    color = None
    if val == u"\u2191":
        color = 'green'  
    elif val == u"\u2193":
        color = 'red'
    return 'background-color: %s' % color

# Use pipeline and model selection to run results 
def run_model(model_name):
    my_bar = st.progress(0)    
    st.session_state.model =  _models._build_model(st.session_state.pipeline, model_name)
    my_bar.progress(50)
    
    #Get the predictions and soprt by date
    st.session_state.model_results = st.session_state.model._return_preds()
    #st.session_state.model_results = st.session_state.model_results[np.argsort( st.session_state.model_results[:, 1])]
  
    st.session_state.model.predictive_power()
    my_bar.progress(60)
    st.session_state.model._feature_importance()
    my_bar.progress(70)
    st.session_state.model._feature_importance_over_time(forecast_range=30)
    my_bar.progress(90)
    st.session_state.metrics = st.session_state.model._return_mean_error_metrics()
    my_bar.progress(100)
    


################################################################################
# Side Bar - File and Date Chooser  
################################################################################

st.sidebar.title("To Begin:")


file_buffer = st.sidebar.file_uploader("Upload new Excel file")

# Default set to debug file 
target_feature = st.sidebar.text_input("Target Feature", 
                                       value="LF98TRUU_Index_OAS", 
                                       type="default")

momentum_list = st.sidebar.text_input("Add Momentum Parameters", 
                                       value="LF98TRUU_Index_OAS,LUACTRUU_Index_OAS", 
                                       type="default")


date_range = st.sidebar.date_input("Select a date range", 
                                   [datetime.date(2012, 8, 1), 
                                    datetime.date(2020, 7, 30)] )



################################################################################
# Main Page
################################################################################

st.title("Decision Support System for Bloomberg Analytics")

session_state.model_chooser = st.selectbox('Which Model Would You Like to Use?',
                             (['XGBoost']))

st.write('You selected:', session_state.model_chooser)



################################################################################
# If Data File -> Pre-process raw data 
################################################################################
if st.sidebar.button('Load Data '):
    if file_buffer:
        my_bar = st.progress(0)
        log.info(" Preprocessing Data File")
        my_bar.progress(20)
        momentum_list = list(momentum_list.split(","))
        st.session_state.pipeline = _preprocessing._preprocess_xlsx(file_buffer,
                                                   target_feature,
                                                   momentum_list = momentum_list)
        my_bar.progress(60)
        data = st.session_state.pipeline._return_dataframe()
        st.write('Current Dataframe Below:')
        st.write(data)
        st.session_state.pipeline_built = True
        my_bar.progress(80)
        # Set dates in the dataframe 
        st.session_state.data['Dates'] = pd.to_datetime(session_state.data['Dates']).dt.date
        st.session_state.data = session_state.data[(session_state.data['Dates'] >= date_range[0]) & 
                                                (session_state.data['Dates'] <= date_range[1])]
        my_bar.progress(100)
        
    else:
        st.sidebar.info('Please select a file')

################################################################################
# If Pipeline Exists -> Train Model 
################################################################################

# If pipeline built and model selected build predictions 
if st.sidebar.button('Train Model'):
    if session_state.pipeline_built and session_state.model_chooser:
        log.info(" Training Model: {}".format(session_state.model_chooser))
        
        # _build_model and return it to sessions state
        run_model(session_state.model_chooser)
        session_state.model_loaded = True
        if session_state.model_chooser == 'XGBoost':
            session_state.regression = True
    else:
        st.sidebar.info('Please load data and select a model')

data = None
my_bar = None

################################################################################
# Main 
################################################################################      
        
if session_state.model_loaded:
    page_selection = st.sidebar.radio("Page", 
                                      ('Main', 
                                       'Feature Importance', 
                                       'Model Performance'), 
                                      0)

    if page_selection == 'Main':
        
        if not my_bar:
            my_bar = st.progress(0)
        
        st.header("Historical Data:")
        
        histo_table = st.checkbox("Display Historical Data Table?", False)
        
        if histo_table:
            st.write(session_state.model_results)
        
        feature_columns = list(session_state.data.columns)
        
        options = st.multiselect('View Historical Indices',
                                feature_columns,
                                target_feature)
        my_bar.progress(40)
        if session_state.regression:
            st.header("Predictions and Forecasts")
            MAE, MSE, RMSE = session_state.metrics
            st.subheader("MAE: {:.4f}, MSE: {:.4f}, RMSE: {:.4f}".format(MAE, MSE, RMSE))
            features = Image.open(str(path)+'\\_img\\predictive_power.png')
            st.image(features, caption='Predictive Power of Features', width=1000)
            my_bar.progress(60)
            features = Image.open(str(path)+'\\_img\\feats_importance.png')
            st.image(features, caption='Features Importance For Forecast Period', width=1000)
            my_bar.progress(70)
            features = Image.open(str(path)+'\\_img\\feats_importance_over_time.png')
            st.image(features, caption='Features Importance Over Time', width=1000)
            
        my_bar.progress(90)
                
        st.write(st.session_state.model_results)

        my_bar.progress(100)
        
################################################################################
# Feature Importance Page 
################################################################################      
        
    if page_selection == 'Feature Importance':
        my_bar = st.progress(0)
        feature_description = pd.read_csv('feature_description.csv')
        st.write(feature_description)
        day_display = st.selectbox('Which Model?', session_state.days_selected)
        my_bar.progress(20)
        if day_display:
            session_state.model_result[day_display]['feature_plot']
        my_bar.progress(100)

################################################################################
# Model Importance Page 
################################################################################           
        
    if page_selection == 'Model Performance':
        my_bar = st.progress(0)
        day_display = st.selectbox('Which Model?', session_state.days_selected, 1)
        my_bar.progress(20)
        if day_display:
            session_state.model_result[day_display]['ROCplot']
        my_bar.progress(100)

else:
    st.info('Please select a file and train the model to begin')

_max_width_()

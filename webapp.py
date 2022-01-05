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
st.sidebar.image(banner, caption='CDX Trade Predictions', width=300)
pd.set_option('display.max_columns', None) 
global counter
counter = 0
pipeline = None
complete_data = None
model_chooser = None

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_info = logging.getLogger(__name__)
log_info.setLevel(logging.INFO)
handler = logging.FileHandler(str(path)+'/logs/_main.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

################################################################################
#  
################################################################################

st.markdown('<style>body{background-color: White;}</style>',unsafe_allow_html=True)

# Set max width of window
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
    model_results =  _models._build_model(session_state.data, model_name)
    return model_results

################################################################################
# Side Bar - File and Date Chooser  
################################################################################

st.sidebar.title("To Begin:")
session_state.model_loaded = False

file_buffer = st.sidebar.file_uploader("Upload new Excel file")

# Default set to debug file 
target_feature = st.sidebar.text_input("Target Feature", 
                                       value="LF98TRUU_Index_OAS", 
                                       type="default")

momentum_list = st.sidebar.text_input("Add Momentum Parameters", 
                                       value="LF98TRUU_Index_OAS, LUACTRUU_Index_OAS", 
                                       type="default")


date_range = st.sidebar.date_input("Select a date range", 
                                   [datetime.date(2012, 8, 1), 
                                    datetime.date(2020, 7, 30)] )



################################################################################
# Main Page
################################################################################

st.title("Decision Support System for CDX Trading")

model_chooser = st.selectbox('Which Model Would You Like to Use?',
                             (['XGBoost']))

st.write('You selected:', model_chooser)

################################################################################
# If Data File -> Pre-process raw data 
################################################################################
if st.sidebar.button('Load Data'):
    if file_buffer:
        my_bar = st.progress(0)
        log.info(" Preprocessing Data File")
        my_bar.progress(20)
        
        pipeline = _preprocessing._preprocess_xlsx(file_buffer,
                                                   target_feature,
                                                   momentum_list.split("delimiter"))
        my_bar.progress(60)
        session_state.pipeline = pipeline
        session_state.data = session_state.pipeline._return_xlsx_dataframe()
        my_bar.progress(80)
        # Set dates in the dataframe 
        session_state.data['Dates'] = pd.to_datetime(session_state.data['Dates']).dt.date
        session_state.data = session_state.data[(session_state.data['Dates'] >= date_range[0]) & 
                                                (session_state.data['Dates'] <= date_range[1])]
        my_bar.progress(100)
        
    else:
        st.sidebar.info('Please select a file')

################################################################################
# If Pipeline Exists -> Train Model 
################################################################################

# If pipeline built and model selected build predictions 
if st.sidebar.button('Train Model'):
    if pipeline and model_chooser:
        log.info(" Training Model: {}".format(model_chooser))
        
        # _build_model and return it to sessions state
        session_state.model_result = run_model(model_chooser)
        session_state.model_loaded = True
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
        st.header("CDX Historical Data:")
        
        histo_table = st.checkbox("Display Historical Data Table?", False)
        if histo_table:
            st.write(session_state.data.sort_values('Dates', ascending=False).head(200).set_index('Dates'))
        
        feature_columns = list(session_state.data.columns)
        
        options = st.multiselect('View Historical Indices',
                                feature_columns,
                                target_feature)
        
        df_2target = pd.melt(session_state.data, id_vars=['Dates'], value_vars=options)
        df_2target.columns = ['Dates','Index','Value']

        fig = px.line(df_2target, x="Dates", y="Value", color='Index', width=1400)
        st.plotly_chart(fig)

        my_bar.progress(60)
        st.header("Predictions and Forecasts")
        display_data = session_state.model_result['final_result'].head(200)
        for ds in session_state.days_selected:
            display_data['CDX_HY_Pred_{}D'.format(ds)] = display_data['CDX_HY_UpNext_{}Day'.format(ds)].map(
                lambda x: (u"\u2191" if x else u"\u2193") )
            display_data['CDX_IG_Pred_{}D'.format(ds)] = display_data['CDX_IG_UpNext_{}Day'.format(ds)].map(
                lambda x: (u"\u2191" if x else u"\u2193") )

        def check_name(col_name):
            if col_name in ['CDX_HY','CDX_IG', 'Dates']:
                return True
            else:
                for prefix in ['CDX_HY_Pred','CDX_IG_Pred']:
                    if col_name.startswith(prefix):
                        return True
            return False
        keep_cols = [col for col in display_data.columns if check_name(col)]
        display_data = display_data[keep_cols]

        my_bar.progress(80)
        
        def get_trade_positions(latest_df):
            HY_pos = IG_pos = "<font color='red'>**SHORT**</font>"
            HY_explanation = IG_explanation = "DECREASE"
            if (latest_df['CDX_HY_Pred_30D'] == u"\u2191") and (latest_df['CDX_HY_Pred_60D'] == u"\u2191"):
                HY_pos = "<font color='red'>**LONG**</font>, based on our predicted price increase in longer term"
                HY_explanation = 'INCREASE'
            if (latest_df['CDX_IG_Pred_30D'] == u"\u2191") and (latest_df['CDX_IG_Pred_60D'] == u"\u2191"):
                IG_pos = "<font color='red'>**LONG**</font>, based on our predicted price increase in longer term"
                IG_explanation = 'INCREASE'
            return HY_pos, IG_pos, HY_explanation, IG_explanation
        
        
        HY_pos, IG_pos, HY_explanation, IG_explanation = get_trade_positions(display_data.iloc[0])
        
        st.markdown("Model recommends taking {} position on CDX HY, based on our predicted price {} in longer term".format(HY_pos, HY_explanation),
                   unsafe_allow_html=True)
        st.markdown("Model recommends taking {} position on CDX IG, based on our predicted price {} in longer term".format(IG_pos, IG_explanation),
                   unsafe_allow_html=True)
        s = display_data.style.applymap(color_arrow)
        s
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

################################################################################
# Author: Adrian Adduci
# Email: FAA2160@columbia.edu 
################################################################################
from itertools import filterfalse
import os

import streamlit as st

st.set_option('deprecation.showfileUploaderEncoding', False)
import datetime
import logging
import pathlib
import pandas as pd
from PIL import Image
import plotly.express as px
import _models
import _preprocessing
from streamlit import session_state

path = pathlib.Path(__file__).parent.absolute()
os.environ['NUMEXPR_MAX_THREADS'] = '16'
banner = Image.open(str(path)+'\\_img\\arrow_logo.png')
st.sidebar.image(banner, caption='ML Predictions From Bloomberg Data Pulls', width=200,)
pd.set_option('display.max_columns', None) 



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log_info = logging.getLogger(__name__)
log_info.setLevel(logging.INFO)
handler = logging.FileHandler(str(path)+'/logs/_main.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

################################################################################
#  Helper Functions
################################################################################

def _max_width_():
    max_width_str = f"max-width: 1700px;"
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
    if 'model' not in session_state:
        log.info(" Building the current model...")    
        session_state.model =  _models._build_model(session_state.pipeline, model_name)
    my_bar.progress(50)
    session_state.model_results = session_state.model._return_preds()
    session_state.model.predictive_power()
    my_bar.progress(60)
    session_state.model._feature_importance()
    my_bar.progress(70)
    session_state.model._feature_importance_over_time(forecast_range=30)
    my_bar.progress(90)
    session_state.metrics = session_state.model._return_mean_error_metrics()
    my_bar.progress(100)
    
################################################################################
# Main Page
################################################################################
my_bar = None

st.title("Decision Support System for Bloomberg Analytics")

if 'data' in session_state:
    st.info('Currently Selected XLSX File: {}'.format(session_state.file_buffer.name))

else:
    st.info('Please load a properly formatted XLSX data file')
    
if 'model_type' not in session_state:
    st.info('Please select a model type and train')

elif 'model' not in session_state and session_state.model_type != '':
    st.warning('Please train the model to continue')

elif 'model' not in session_state and session_state.model_type == '': 
    st.info('Please select a model type and train')
            
else:
    st.info('Currently Selected Model: {}'.format(session_state.model_type))

if 'model' in session_state:
    
    page_selection = st.sidebar.radio("Page", 
                                      ('Historic Data',
                                       'Model Analsys'), 
                                      index=0)

    if page_selection == 'Main':
        
        if not my_bar:
            my_bar = st.progress(0)
        
        st.header("Historical Data:")
        
        if 'data' in session_state:
            session_state.feature_columns = list(session_state.data.columns)
            my_bar.progress(20)
            options = st.multiselect('View Historical Indices',
                                    session_state.feature_columns,
                                    session_state.target_feature)
            
            df_2target = pd.melt(session_state.data, id_vars=['Dates'], value_vars=options)
            df_2target.columns = ['Dates','Target','Value']

            fig = px.line(df_2target, x="Dates", y="Value", color='Target', width=1100)
            st.plotly_chart(fig)
        
        my_bar.progress(40)
        
        session_state.histo_table = st.checkbox("Display Historical Data Table?", False)
        
        if session_state.histo_table:
            st.write(session_state.data.sort_values('Dates', ascending=False).set_index('Dates'))
        my_bar.progress(80)
        
        
        if 'regression' in session_state:
            st.header("Feature Importance & Model Analysis")
            MAE, MSE, RMSE = session_state.metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", MAE)
            col2.metric("Mean Square Error", MSE,)
            col3.metric("Root Mean Square Error", RMSE)
            
            features = Image.open(str(path)+'\\_img\\predictive_power.png')
            st.image(features, caption='Predictive Power of Features', width=1000)
            my_bar.progress(60)
            features = Image.open(str(path)+'\\_img\\feats_importance.png')
            st.image(features, caption='Features Importance For Forecast Period', width=1000)
            my_bar.progress(70)
            features = Image.open(str(path)+'\\_img\\feats_importance_over_time.png')
            st.image(features, caption='Features Importance Over Time', width=1000)             

        my_bar.progress(100)
        

################################################################################
# Side Bar - File and Date Chooser  
################################################################################
st.sidebar.subheader("Reset: ")

if st.sidebar.button('Reset All'):
    for key in st.session_state.keys():
        del st.session_state[key]
    
    if 'pipeline' in session_state:
        del st.session_state['pipeline']
        
    st.experimental_rerun() 


st.sidebar.subheader("Load Data & Train Model:")


session_state.file_buffer = st.sidebar.file_uploader("Upload new Excel file")

# Default set to debug file 
session_state.target_feature = st.sidebar.text_input("Target Feature", 
                                       value="LF98TRUU_Index_OAS", 
                                       type="default")


session_state.momentum_list = st.sidebar.text_input("Add Momentum Parameters", 
                                       value="LF98TRUU_Index_OAS,LUACTRUU_Index_OAS", 
                                       type="default")
st.sidebar.caption("Enter column names seperated by a comma (,) ")

date_range = st.sidebar.date_input("Select a date range", 
                                   [datetime.date(2012, 8, 8), 
                                    datetime.date(2020, 7, 31)] )

################################################################################
# If Data File -> Pre-process raw data 
################################################################################
if st.sidebar.button('Load Data '):
    
    if 'file_buffer' in session_state:
        my_bar = st.progress(0)
        log.info(" Preprocessing Data File")
        my_bar.progress(20)
        session_state.momentum_list = list(session_state.momentum_list.split(","))
        session_state.pipeline = _preprocessing._preprocess_xlsx(session_state.file_buffer,
                                                   session_state.target_feature,
                                                   momentum_list = session_state.momentum_list)
        my_bar.progress(60)
        
        session_state.data  = session_state.pipeline._return_dataframe()
        
        if 'data' in session_state:
            st.write('Current Dataframe Below:')
            st.dataframe(session_state.data)
                
        my_bar.progress(80)
        
        # Set dates in the dataframe 
        session_state.data['Dates'] = pd.to_datetime(session_state.data['Dates']).dt.date
        
        # WIP - Autosize date range based on XLSX 
        session_state.data = session_state.data[(session_state.data['Dates'] >= date_range[0]) & 
                                                (session_state.data['Dates'] <= date_range[1])]
        my_bar.progress(100)
        
    else:
        st.sidebar.info('Please select a file')

################################################################################
# If Pipeline Exists -> Train Model 
################################################################################
session_state.model_type = \
    st.sidebar.selectbox('Which Model Would You Like to Use?', (['','XGBoost']))

# If pipeline built and model selected build predictions 
if st.sidebar.button('Train Model'):
    
    if session_state.model_type == '':
        st.warning('Must choose a model to continue')  
        
    else:
        if 'data' and 'model' not in session_state:
            
            log.info(" Training Model: {}".format(session_state.model_type))
            
            # _build_model and return it to sessions state
            run_model(session_state.model_type)
            
            if session_state.model_type == 'XGBoost':
                session_state.regression = True
        
        elif 'data' not in session_state and 'model' in session_state:
            st.sidebar.info('Please load data to train')
        
        elif 'model_type' not in session_state:
               st.sidebar.info('Please choose a model to continue') 
        else:
            st.sidebar.info('Please load data to train')



_max_width_()

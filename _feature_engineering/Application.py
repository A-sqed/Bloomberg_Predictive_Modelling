"""
Run as a streamlit application:
https://docs.streamlit.io/en/stable/getting_started.html

Libraries: streamlit, matplotlib, sklearn, 

Author: Adrian Adduci | adrian.adduci@gmail.com

"""
import sys, time, datetime
import streamlit as st
from streamlit import session_state
import numpy as np
import pandas as pd
import pylab as pl
import logging

# ML Libraries
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA

# Graphics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns
from PIL import Image

# Globals and Settings
global _COUNTER
_COUNTER = 0
final_data = None

# Adaboost Settings
AB_estimators = 30 
AB_learning_rate = 0.5
AB_random_state = 1

test_set_size = .25 # as a percentage 
shuffle_flag = False # test/train data 


logging.basicConfig(
    filename="test.log", 
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s")

#Return rolling average for window1 / window2
def momentum_calc(csv_data, category_name, window1, window2):
    return  csv_data[category_name].rolling(window=window1).mean() - \
            csv_data[category_name].rolling(window=window2).mean() / \
            csv_data[category_name].rolling(window=window2).mean()

# Import CSV, insert momentum calculations 
def process_CSV(csv_data, days_ahead):

    logging.debug(
        "Preprocessing data recieved from {} for {} days ahead".format(
        csv_data, 
        days_ahead))

    csv_data['EARN_DOWN'] = csv_data['EARN_DOWN'].astype(np.float16)
    csv_data['EARN_UP'] = csv_data['EARN_DOWN'].astype(np.float16)
    csv_data['CDX_HY_momentum'] = (csv_data['CDX_HY_10D_AVG'] - csv_data['CDX_HY_30D_AVG']) / csv_data['CDX_HY_30D_AVG']
    csv_data['CDX_IG_momentum'] = (csv_data['CDX_IG_10D_AVG'] - csv_data['CDX_IG_30D_AVG']) / csv_data['CDX_IG_30D_AVG']
    csv_data['GOLD_momentum'] = momentum_calc(csv_data, 'GOLD', 10, 30)

    final_data = csv_data.dropna().copy()

    logging.debug("Data Import Complete")


################################################################################
# Run Predictions
################################################################################

def total_prediction(csv_data, days_ahead):
   
    model_results = {}

    for dh in days_ahead:
        model_results[dh] = {}
        logging.debug("==={} Day===".format(dh))

        HY_colname = 'CDX_HY_UpNext_{}Day'.format(dh)
        IG_colname = 'CDX_IG_UpNext_{}Day'.format(dh)
        
        csv_data[HY_colname] = csv_data['CDX_HY'].shift(-dh) > csv_data['CDX_HY']
        csv_data[IG_colname] = csv_data['CDX_IG'].shift(-dh) > csv_data['CDX_IG']
        complete_data = csv_data.dropna().copy()
        X_complete_data= complete_data.drop([HY_colname, IG_colname,'Dates'], axis=1)
        feature_cols = X_complete_data.columns
        
        Y_HY_raw_data = complete_data[HY_colname]
        Y_IG_raw_data = complete_data[IG_colname]

        logging.debug("Splitting Test and Training Data")

        # Create Test / Train Data 
        X_HY_train_data, X_HY_test_data, Y_HY_train_data, Y_HY_test_data = \
        train_test_split(X_complete_data,
                        Y_HY_raw_data, 
                        test_size=test_set_size, 
                        shuffle=shuffle_flag)
                                                                                            
        X_IG_train, X_IG_test_data, Y_IG_train_data, Y_IG_test_data = \
        train_test_split(X_complete_data, 
                        Y_IG_raw_data, 
                        test_size=test_set_size, 
                        shuffle=shuffle_flag)
        
        logging.debug("Encoding target data with labels")

        # Encode Target
        label_encode = preprocessing.LabelEncoder()
        Y_HY_raw_data_encoded = label_encode.fit_transform(Y_HY_raw_data)
        Y_IG_raw_data_encoded = label_encode.fit_transform(Y_IG_raw_data)
        Y_HY_train_data_encoded = label_encode.fit_transform(Y_HY_train_data)
        Y_IG_train_data_encoded = label_encode.fit_transform(Y_IG_train_data)

        logging.debug("Training the HY models (1st iteration, AdaBoost)")

        AB_model_HY = AdaBoostClassifier(n_estimators=AB_estimators, 
                                        learning_rate = AB_learning_rate,
                                        random_state=AB_random_state).fit(X_HY_train_data, 
                                                                          Y_HY_train_data_encoded)

        logging.debug("Training the IG models (1st iteration, AdaBoost)")

        AB_model_IG = AdaBoostClassifier(n_estimators=AB_estimators, 
                                        learning_rate = AB_learning_rate,
                                        random_state=AB_random_state).fit(X_IG_train, 
                                                                          Y_IG_train_data_encoded)
        
        Y_HY_test_data_encoded = label_encode.fit_transform(Y_HY_test_data)
        Y_IG_test_data_encoded = label_encode.fit_transform(Y_IG_test_data)

        Y_HY_pred = AB_model_HY.predict(X_HY_test_data)
        Y_IG_pred = AB_model_IG.predict(X_IG_test_data)
        
        accuracy_HY = metrics.accuracy_score(Y_HY_test_data_encoded, Y_HY_pred)
        logging.debug('AdaBoost: HY Accuracy {:.2%}'.format(accuracy_HY))

        accuracy_IG = metrics.accuracy_score(Y_IG_test_data_encoded, Y_IG_pred)
        logging.debug('AdaBoost: IG Accuracy {:.2%}'.format(accuracy_IG))
        
        #feature_plot
        fig_feature, axes = plt.subplots(nrows=1, ncols=2)
        feat_impt_HY = pd.DataFrame(AB_model_HY.feature_importances_, columns=['Feature Importance'], index= X_complete_data.columns)
        feat_impt_HY = feat_impt_HY.sort_values('Feature Importance', ascending=True)
        feature_plot_HY = feat_impt_HY.plot(kind='barh', title='Feature Importance HY',figsize=(20, 10), ax=axes[0])
        feat_impt_HY = feat_impt_HY[-20:]
        
        feat_impt_IG = pd.DataFrame(AB_model_IG.feature_importances_, columns=['Feature Importance'], index= X_complete_data.columns)
        feat_impt_IG = feat_impt_IG.sort_values('Feature Importance', ascending=True)
        feature_plot_IG = feat_impt_IG.plot(kind='barh', title='Feature Importance IG',figsize=(20, 10), ax=axes[1])
        feat_impt_IG = feat_impt_IG[-20:]
        fig_feature.tight_layout(pad=3.0)

        #HY_area_roc = roc_auc_score(Y_HY_test_data_encoded , Y_HY_pred)
        #IG_area_roc = roc_auc_score(Y_IG_test_data_encoded , Y_IG_pred)
        
        fig_roc, axes = plt.subplots(nrows=2, ncols=2,  figsize=(15, 7))
        HY_ROCplot = plot_roc_curve(AB_model_HY, X_HY_test_data, Y_HY_test_data_encoded, ax=axes[0,0])
        axes[0,0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,0].title.set_text("CDX HY Prediction ROC [Accuracy: {}]".format(accuracy_HY))
        
        IG_ROCplot = plot_roc_curve(AB_model_IG, X_IG_test_data, Y_IG_test_data_encoded, ax=axes[0,1])
        axes[0,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0,1].title.set_text("CDX IG Prediction ROC [Accuracy: {}]".format(accuracy_IG))
        
        HY_PRplot = plot_precision_recall_curve(AB_model_HY, X_HY_test_data, Y_HY_test_data_encoded, ax=axes[1,0])
        axes[1,0].title.set_text("CDX HY Prediction Precision-Recall Curve")
        
        IG_PRplot = plot_precision_recall_curve(AB_model_IG, X_IG_test_data, Y_IG_test_data_encoded, ax=axes[1,1])
        axes[1,1].title.set_text("CDX IG Prediction Precision-Recall Curve")
        fig_roc.tight_layout(pad=3.0)

        logging.debug("Training the HY models (2nd iteration, AdaBoost)")
        AB_model_HY = AdaBoostClassifier(n_estimators=30, learning_rate = 0.5,
                                         random_state=1).fit(X_complete_data, Y_HY_raw_data_encoded)

        logging.debug("Training the IG models (2nd iteration, AdaBoost)")

        AB_model_IG = AdaBoostClassifier(n_estimators=30, learning_rate = 0.5,
                                         random_state=1).fit(X_complete_data, Y_IG_raw_data_encoded)
        
        final_data[HY_colname] = AB_model_HY.predict(final_data[feature_cols])
        final_data[IG_colname] = AB_model_IG.predict(final_data[feature_cols])
        
        model_results[dh] = {'ROCplot':fig_roc, 'feature_plot':fig_feature}
        
    model_results['final_result'] = final_data.sort_values('Dates', ascending=False).set_index('Dates')

    return model_results


################################################################################
# Application Window Settings
################################################################################
st.set_option('deprecation.showfileUploaderEncoding', False)

#banner = Image.open('columbia_banner.jpeg')

#st.sidebar.image(banner, 
#                 caption='Index Forecasting', 
#                 width=300)
pd.set_option('display.max_columns', None) 


################################################################################
# Helpers
################################################################################

# Set max width of streamlit window 
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

# Define arrow color for credit market (green = down)
def color_arrow(val):
    color = None
    if val == u"\u2191":
        color = 'green'  
    elif val == u"\u2193":
        color = 'red'
    return 'background-color: %s' % color

def check_name(col_name):
    if col_name in ['CDX_HY','CDX_IG', 'Dates']:
        return True
    else:
        for prefix in ['CDX_HY_Pred','CDX_IG_Pred']:
            if col_name.startswith(prefix):
                return True
    return False

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

_max_width_()

st.sidebar.title("To Begin:")
file_buffer = st.sidebar.file_uploader("Upload new Excel file")

# Need to add a date range import from CSV function 
date_range = st.sidebar.date_input("Select a date range", 
                        [datetime.date(2012, 8, 1), datetime.date(2020, 7, 30)])

st.title("Decision Support System ")
my_bar = None

session_state.model_loaded = False

# Sidebar for loading data and executing model 
if st.sidebar.button('Train Model'):

    if file_buffer:
        
        # Read in CSV file into current session_state
        my_bar = st.progress(0)
        
        session_state.data = pd.read_excel(file_buffer)
        
        # Add dates to session_state in dataframe datime format 
        session_state.data['Dates'] = \
                            pd.to_datetime(session_state.data['Dates']).dt.date
        
        # Remove all dates outside of analysis period
        session_state.data = \
                            session_state.data[
                                (session_state.data['Dates'] >= date_range[0]) & 
                                (session_state.data['Dates'] <= date_range[1])]
        my_bar.progress(20)
        
        # Options for forecasting periods in days 
        session_state.days_selected = [1,7,30,60]
        
        # Forecast the model 
        session_state.model_result = \
            total_prediction(session_state.data, session_state.days_selected)
        my_bar.progress(40)
        session_state.model_loaded = True
    
    # Else need the input file
    else:
        st.sidebar.info('Please select a file')

# If model has been successfully been trained     
if session_state.model_loaded:

    # Options for main window views
    page_selection = st.sidebar.radio(
        "Page", 
            ('Main',
            'Feature Importance', 
            'Model Performance'), 0)

    # Main page options
    if page_selection == 'Main':
       
       if not my_bar: 
           my_bar = st.progress(0)

       st.header("Historical Data:")
       histo_table = st.checkbox("Display Historical Data Table?", False)

       # Build table of historical data if option true               
       if histo_table:
           st.write(
               session_state.data.sort_values(
                   'Dates', ascending=False).head(200).set_index('Dates'))
        
       # Model Features 
       feature_columns = list(session_state.data.columns)
        
       # Default view to the CDX Index 
       options = st.multiselect('View Historical Indices',
                                feature_columns,
                                 ['CDX_HY','CDX_IG'])
        
       df_2target = pd.melt(session_state.data, 
                            id_vars=['Dates'], 
                            value_vars=options)
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


       keep_cols = [col for col in display_data.columns if check_name(col)]
       display_data = display_data[keep_cols]

       my_bar.progress(80)
        
       HY_pos, IG_pos, HY_explanation, IG_explanation = get_trade_positions(display_data.iloc[0])
        
       st.markdown("Model recommends taking {} position on CDX HY, based on our predicted price {} in longer term".format(HY_pos, HY_explanation),
                   unsafe_allow_html=True)
       st.markdown("Model recommends taking {} position on CDX IG, based on our predicted price {} in longer term".format(IG_pos, IG_explanation),
                   unsafe_allow_html=True)
       s = display_data.style.applymap(color_arrow)
       my_bar.progress(100)
    
    if page_selection == 'Feature Importance':
        my_bar = st.progress(0)
        feature_description = pd.read_csv('feature_description.csv')
        st.write(feature_description)
        day_display = st.selectbox('Which Model?', session_state.days_selected)
        my_bar.progress(20)
        if day_display:
            session_state.model_result[day_display]['feature_plot']
        my_bar.progress(100)
    if page_selection == 'Model Performance':
        my_bar = st.progress(0)
        day_display = st.selectbox('Which Model?', session_state.days_selected, 1)
        my_bar.progress(20)
        if day_display:
            session_state.model_result[day_display]['ROCplot']
        my_bar.progress(100)
else:
    st.info('Please select a file and train the model to begin')

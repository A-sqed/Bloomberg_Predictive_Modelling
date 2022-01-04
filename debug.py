#%%
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import datetime
import sklearn as sk
import pylab as pl
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn import preprocessing
from statsmodels.tsa.arima_model import ARIMA
import sys
import time
from PIL import Image
import _preprocessing, _models


################################################################################

target_feature = 'LF98TRUU_Index_OAS'
momentum_list = ['LF98TRUU_Index_OAS', 'LUACTRUU_Index_OAS']
file_buffer =  './data/Economic_Data_2020_08_01.xlsx'

pipeline = _preprocessing._preprocess_xlsx(file_buffer,
                                           target_feature,
                                           momentum_list = momentum_list
                                           )

new_model = _models._build_model(pipeline, model_name='XGB')

#works!
#new_model.predictive_power()
#new_model._feature_importance()
new_model._feature_importance_over_time(forecast_range=30)

# %%

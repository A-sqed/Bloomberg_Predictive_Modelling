#%%
import streamlit as st
st.set_option('deprecation.showfileUploaderEncoding', False)
import _preprocessing, _models


################################################################################

target_feature = 'LF98TRUU_Index_OAS'
momentum_list = ['LF98TRUU_Index_OAS', 'LUACTRUU_Index_OAS']
file_buffer =  './data/Economic_Data_2020_08_01.xlsx'

pipeline = _preprocessing._preprocess_xlsx(file_buffer,
                                           target_feature,
                                           momentum_list = momentum_list
                                           )

new_model = _models._build_model(pipeline, model_name='XGBoost')

#works!
new_model.predictive_power()
#new_model._feature_importance()
#new_model._feature_importance_over_time(forecast_range=30)
#new_model._return_mean_error_metrics()

# Needs a classifier not binary model
#new_model._return_roc_and_precision_recall_curves()

# %%

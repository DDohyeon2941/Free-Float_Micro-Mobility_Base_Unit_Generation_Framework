# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:33:01 2024

@author: dohyeon
"""

import pandas as pd
import numpy as np

new_temp_df = pd.read_csv(r'minneapolis_prediction_performance_500m_prop_all_sep_ovl_0812_rev.csv')
temp_df = pd.read_csv(r'minneapolis_prediction_performance_250m_750m_0812.csv', index_col=0)


temp_df.groupby(['metric','group']).std(ddof=0)['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].to_csv(r'minneapolis_prediction_performance_250m_750m_groupby_std_0812_rev.csv')

new_temp_df.groupby(['metric','group','type']).std(ddof=0)['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].to_csv(r'minneapolis_prediction_performance_500m_prop_all_sep_ovl_groupby_std_0812_rev.csv')




new_temp_df.groupby(['metric','group','type']).mean()['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].round(4)

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:33:01 2024

@author: dohyeon
"""

import user_utils as uu
import pandas as pd
import numpy as np

temp_df = pd.read_csv(r'kansas_prediction_performance_250m_750m_0812.csv', index_col=0)

temp_df.groupby(['metric','group']).mean()['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].to_csv(r'kansas_prediction_performance_250m_750m_groupby_0812_rev.csv')


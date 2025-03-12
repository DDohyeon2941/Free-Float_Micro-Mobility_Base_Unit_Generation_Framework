# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 18:33:01 2024

@author: dohyeon
"""

import user_utils as uu
import pandas as pd
import numpy as np

new_temp_df = pd.read_csv(r'kansas_prediction_performance_500m_prop_all_sep_ovl_0812_rev.csv')
temp_df = pd.read_csv(r'kansas_prediction_performance_250m_750m_0812.csv', index_col=0)

temp_df.groupby(['metric','group']).std(ddof=0)['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].to_csv(r'kansas_prediction_performance_250m_750m_groupby_std_0812_rev.csv')

new_temp_df.groupby(['metric','group','type']).std(ddof=0)['score'].unstack(0)[['rmse','mae','mape','smape','smape0']].to_csv(r'kansas_prediction_performance_500m_prop_all_sep_ovl_groupby_std_0812_rev.csv')


new_temp_df.groupby(['metric','group','type']).mean()['score'].unstack(0)[['rmse','mae','mape','smape','smape0']]


#%%


import math

def calculate_std(data):
    # 1. 데이터의 평균을 구합니다.
    mean = sum(data) / len(data)
    
    # 2. 각 데이터 포인트에서 평균을 빼고 그 차이를 제곱합니다.
    squared_diffs = [(x - mean) ** 2 for x in data]
    
    # 3. 제곱된 값들의 평균을 구합니다.
    variance = sum(squared_diffs) / len(squared_diffs)
    
    # 4. 그 평균의 제곱근을 취합니다.
    std_dev = math.sqrt(variance)
    
    return std_dev


std_dev = calculate_std(np.array([0.84994117, 0.86341106, 0.8814945 , 0.84148585, 0.85267676]))
print("표준편차:", std_dev)
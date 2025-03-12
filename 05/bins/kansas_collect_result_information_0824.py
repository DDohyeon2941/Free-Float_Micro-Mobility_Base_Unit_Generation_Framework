# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np

import user_utils as uu

date_df = pd.read_csv(r'kansas_test_target_date_0820.csv')

def merge_real_pred_y(real_y, pred_y):
    time_len = real_y.shape[1]
    temp_list = []

    for xx in range(pred_y.shape[0]):

        temp_df = pd.DataFrame({'grid_idx':np.ones(time_len)*xx,'time_idx':date_df['0'], 'real':real_y[xx,:], 'pred':pred_y[xx,:]})
        temp_list.append(temp_df)

    return pd.concat(temp_list).reset_index(drop=True)

def merge_real_pred_y_ovl(real_y, pred_y, ovl_index):
    time_len = real_y.shape[1]
    temp_list = []

    for xx in range(pred_y.shape[0]):

        temp_df = pd.DataFrame({'grid_idx':np.ones(time_len)*xx,'time_idx':date_df['0'], 'real':real_y[xx,:], 'pred':pred_y[xx,:]})
        if xx in ovl_index:
            temp_df.loc[:,'ovl']=1
        else:
            temp_df.loc[:,'ovl']=0
        temp_list.append(temp_df)

    return pd.concat(temp_list).reset_index(drop=True)


def merge_real_pred_y_ovl_prop(real_y, pred_y, ovl_index, base_unit_index):
    time_len = real_y.shape[1]
    temp_list = []

    for xx in range(pred_y.shape[0]):

        temp_df = pd.DataFrame({'grid_idx':np.ones(time_len)*xx,'time_idx':date_df['0'], 'real':real_y[xx,:], 'pred':pred_y[xx,:]})

        if xx in ovl_index:
            temp_df.loc[:,'ovl']=1
        else:
            temp_df.loc[:,'ovl']=0
        temp_df.loc[:,'n_grid'] = len(base_unit_index[xx])
        temp_list.append(temp_df)

    return pd.concat(temp_list).reset_index(drop=True)

#%%

temp_obj1 = uu.load_gpickle(r'kansas_real_pred_y_fixed_250m_0824.pickle')
temp_obj2 = uu.load_gpickle(r'kansas_real_pred_y_fixed_750m_0824.pickle')
temp_obj3 = uu.load_gpickle(r'kansas_real_pred_y_fixed_0824.pickle')
temp_obj4 = uu.load_gpickle(r'kansas_real_pred_y_prop_0824.pickle')

#%%


result_250m = merge_real_pred_y(temp_obj1['real_y'], temp_obj1['pred_y'])
result_750m = merge_real_pred_y(temp_obj2['real_y'], temp_obj2['pred_y'])

result_250m.to_csv(r'kansas_250m_574_0824.csv',index=False)
result_750m.to_csv(r'kansas_750m_147_0824.csv',index=False)

#%%


base_unit_index_pkl = uu.load_gpickle(r'kansas_overlapped_index_fixed_prop_0809.pickle')


result_500m = merge_real_pred_y_ovl(temp_obj3['real_y'], temp_obj3['pred_y'], base_unit_index_pkl['fixed_grid_index'])



result_500m.to_csv(r'kansas_500m_262_0824.csv',index=False)

#%%


obj2 = uu.load_gpickle(r'kansas_prop_45c_237f_0809.pickle')

result_prop = merge_real_pred_y_ovl_prop(temp_obj4['real_y'], temp_obj4['pred_y'], base_unit_index_pkl['prop_grid_index'], obj2)


result_prop.to_csv(r'kansas_prop_282_0824.csv',index=False)







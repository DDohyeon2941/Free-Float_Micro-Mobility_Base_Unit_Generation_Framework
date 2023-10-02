# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 13:32:49 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import itertools
import pendulum
from sklearn.preprocessing import MinMaxScaler
import data_preproceess_fixed_indi_train_test_250m_0615 as pu

import user_utils as uu



def split_train_vali(train_info, vali_days):
    train_days = np.arange(1,22)
    print(train_days)
    print(vali_days)

    train_info.loc[:,'day'] = [xx.day for xx in train_info.date]
    new_train_df = train_info.loc[train_info.day.isin(train_days)].reset_index(drop=True)
    vali_df = train_info.loc[train_info.day.isin(vali_days)].reset_index(drop=True)
    return new_train_df.drop(columns='day'), vali_df.drop(columns='day')


def get_splited_df_scaler(train_info, vali_days):
    """train과 vali에 대해서 타겟값 정규화 진행"""
    target_col= 'demand'
    tr1, va1 = split_train_vali(train_info, vali_days)

    scaler=MinMaxScaler()
    
    scaler.fit(tr1[target_col].values.reshape(-1,1))
    
    tr1.loc[:, 'scaled_y'] = scaler.transform(tr1[target_col].values.reshape(-1,1)).squeeze()
    va1.loc[:, 'scaled_y'] = scaler.transform(va1[target_col].values.reshape(-1,1)).squeeze()
    return tr1, va1, scaler


def conv_tup_to_dic(uni_tup):
    """모든 작업이 끝난 결과를 딕셔너리형태로 리"""
    return {'scaler':uni_tup[2], 'train_dataset':uni_tup[0], 'valid_dataset':uni_tup[1]}

if __name__ == '__main__':

    temp_pkl = pd.read_pickle(r'roughly_filtered_dataset_250m_0615.pkl')
    train_df = pu.prep_base_df(temp_pkl, True)
    
    
    train_vali_dic = {}
    
    train_vali_dic[1] = conv_tup_to_dic(get_splited_df_scaler(train_df, [22, 23]))
    train_vali_dic[2] = conv_tup_to_dic(get_splited_df_scaler(train_df, [24, 25]))
    train_vali_dic[3] = conv_tup_to_dic(get_splited_df_scaler(train_df, [26, 27]))
    train_vali_dic[4] = conv_tup_to_dic(get_splited_df_scaler(train_df, [28, 29]))
    train_vali_dic[5] = conv_tup_to_dic(get_splited_df_scaler(train_df, [30, 31]))


    #%%
    uu.save_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_0615.pickle', train_vali_dic)
    

    
    
    
    
    
    
    







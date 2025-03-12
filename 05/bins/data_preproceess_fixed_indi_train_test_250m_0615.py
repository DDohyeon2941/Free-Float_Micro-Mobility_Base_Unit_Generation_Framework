# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 15:19:41 2023

@author: dohyeon
"""


import pandas as pd
import numpy as np
from datetime import date, datetime
import itertools
import pendulum
from sklearn.preprocessing import MinMaxScaler
import user_utils as uu

def get_mask_hour(uni_hour):

    if uni_hour in [0,1,2,3,4,5]: return 0
    elif uni_hour in [6,7,8,9,10,11]: return 1
    elif uni_hour in [12,13,14,15,16,17]: return 2
    elif uni_hour in [18,19,20,21,22,23]: return 3


def get_base_df(is_train=True):
    """실험기간내 최소 time step을 인덱스로 하는 df 생성"""
    if is_train:
        sdate = date(2019,7,1)   # start date
        edate = datetime(2019,7,31,23,59)   # end date
    else:
        sdate = date(2019,8,1)   # start date
        edate = datetime(2019,8,31,23,59)   # end date

    t_idx =  pd.date_range(sdate,edate,freq='1h')
    temp_df = pd.DataFrame(index= t_idx , data=np.zeros(len(t_idx)))
    temp_df.loc[:, 'date'] = [xx.date() for xx in temp_df.index]
    temp_df.loc[:, 'hour'] = [xx.hour for xx in temp_df.index]
    temp_df = temp_df.reset_index(drop=True)
    new_temp_df = temp_df.set_index(['date','hour'])

    return new_temp_df

def get_demand_df(all_info, is_train=True):
    """실험기간내 최소 time step별 대여량 df 생성"""
    SEL_COLS = ['start_date','hour','start_grid']
    if is_train:
        train_info = all_info.loc[(all_info.start_month==7)].reset_index(drop=True)
    else:
        train_info = all_info.loc[(all_info.start_month==8)].reset_index(drop=True)

    demand_df = train_info.groupby(SEL_COLS).count()['year'].unstack(fill_value=0)
    return demand_df

def prep_base_df(all_info, is_train=True):
    """전처리 메인"""

    new_temp_df = get_base_df(is_train)
    demand_df = get_demand_df(all_info, is_train)
    n_units = demand_df.shape[1]


    test1 = pd.DataFrame(index=new_temp_df.index,
                         columns=np.arange(n_units), data=0)
    test1.loc[demand_df.index,test1.columns] = demand_df.values

    test2 = test1.stack().reset_index()

    test2 = test2.rename(columns={0:'demand'})
    doweek = [pendulum.parse(str(xx)).day_of_week for xx in  test2.date]
    qohour = [get_mask_hour(xx) for xx in test2.hour]

    mask_key = list(itertools.product([0,1,2,3,4,5,6], [0,1,2,3]))
    mask_dic = dict(zip(mask_key, np.arange(28)))

    mask_wm = [mask_dic[(xx,yy)] for xx,yy in zip(doweek, qohour)]
    test2.loc[:, 'mask_wq'] = mask_wm

    test2['demand'] = np.log(test2['demand']+1)
    return test2
#%%
if __name__ == '__main__':
    temp_pkl = pd.read_pickle(r'roughly_filtered_dataset_250m_0615.pkl')

    train_df = prep_base_df(temp_pkl, True)
    test_df = prep_base_df(temp_pkl, False)
    
    target_col = 'demand'
    #%%
    scaler=MinMaxScaler()
    
    scaler.fit(train_df[target_col].values.reshape(-1,1))
    
    train_df.loc[:, 'scaled_y'] = scaler.transform(train_df[target_col].values.reshape(-1,1)).squeeze()
    test_df.loc[:, 'scaled_y'] = scaler.transform(test_df[target_col].values.reshape(-1,1)).squeeze()
    
    uu.save_gpickle(r'kansas_fixed_train_test_scaler_dataset_250m_0615.pickle', {'scaler':scaler, 'train_dataset':train_df, 'test_dataset':test_df})
    










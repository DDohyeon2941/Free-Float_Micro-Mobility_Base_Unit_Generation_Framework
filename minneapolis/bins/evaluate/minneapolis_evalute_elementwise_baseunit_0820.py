# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_utils as uu
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae




def plain_average(real_y_2d, pred_y_2d):
    s_rmse = mse(real_y_2d.T, pred_y_2d.T, multioutput='raw_values')**0.5
    s_mae = mae(real_y_2d.T, pred_y_2d.T, multioutput='raw_values')
    s_mape = get_mape_2d(real_y_2d, pred_y_2d)
    s_smape = get_smape_2d(real_y_2d, pred_y_2d)
    s_smape1 = get_smape_2d_no_zero(real_y_2d, pred_y_2d)

    return s_rmse.mean().round(4), s_mae.mean().round(4), np.nanmean(s_mape).round(4), s_smape.mean().round(4), s_smape1.mean().round(4)



def get_metric_2d(real_y_2d, pred_y_2d):
    s_rmse = mse(real_y_2d.T, pred_y_2d.T, multioutput='raw_values')**0.5
    s_mae = mae(real_y_2d.T, pred_y_2d.T, multioutput='raw_values')
    s_mape = get_mape_2d(real_y_2d, pred_y_2d)
    s_smape = get_smape_2d(real_y_2d, pred_y_2d)
    s_smape1 = get_smape_2d_no_zero(real_y_2d, pred_y_2d)

    return np.vstack((s_rmse, s_mae, s_mape, s_smape, s_smape1)).T

def get_smape1(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum((np.abs(pred_y-real_y)) / (((real_y+pred_y)/2)+1e-14))/real_y.shape[0]

def get_smape_no_zero(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    bzi = real_y>0
    return np.sum((np.abs(pred_y[bzi]-real_y[bzi])) / (((real_y[bzi]+pred_y[bzi])/2)+1e-14))/real_y[bzi].shape[0]


def get_smape_2d(real_y_2d, pred_y_2d):

    smape_li = []
    for uni_row in range(real_y_2d.shape[0]):
        smape_li.append(get_smape1(real_y_2d[uni_row,:], pred_y_2d[uni_row,:]))
    return np.array(smape_li)

def get_smape_2d_no_zero(real_y_2d, pred_y_2d):

    smape_li = []
    for uni_row in range(real_y_2d.shape[0]):
        try:
            smape_li.append(get_smape_no_zero(real_y_2d[uni_row,:], pred_y_2d[uni_row,:]))
        except:
            smape_li.append(np.nan)
    return np.array(smape_li)



def get_mape(real_y, pred_y):
    return mape(real_y[np.where(real_y>0)[0]],pred_y[np.where(real_y>0)[0]])

def get_mape_2d(real_y_2d, pred_y_2d):

    mape_li = []
    for uni_row in range(real_y_2d.shape[0]):
        try:
            mape_li.append(get_mape(real_y_2d[uni_row,:], pred_y_2d[uni_row,:]))
        except: mape_li.append(np.nan)

    return np.array(mape_li)



#import ipdb
def get_metrics(final_y1, final_pred):

    s_mse = mse(final_y1, final_pred)**0.5
    s_mae = mae(final_y1, final_pred)
    s_smape = get_smape1(final_y1, final_pred)
    s_mape = mape(final_y1[np.where(final_y1>0)[0]],  final_pred[np.where(final_y1>0)[0]])
    #ipdb.set_trace()
    return [s_mse, s_mae, s_mape, s_smape]

def get_metrics1(final_y1, final_pred):

    s_mse = mse(final_y1, final_pred)**0.5
    s_mae = mae(final_y1, final_pred)
    s_smape = get_smape1(final_y1, final_pred)
    #s_mape = mape(final_y1[np.where(final_y1>0)[0]],  final_pred[np.where(final_y1>0)[0]])
    #ipdb.set_trace()
    return [s_mse, s_mae, s_smape]

#%%

temp_obj = uu.load_gpickle(r'minneapolis_real_pred_y_prop_0820.pickle')
temp_obj1 = uu.load_gpickle(r'minneapolis_real_pred_y_fixed_0820.pickle')

temp_obj2 = uu.load_gpickle(r'minneapolis_fixed_prop_index_0809.pickle')


#%%

all_pred_y_prop, all_real_y_prop = temp_obj['pred_y'], temp_obj['real_y']
all_pred_y_fixed, all_real_y_fixed = temp_obj1['pred_y'], temp_obj1['real_y']


#%% 조건별 metric 2d 행렬 만들기

metric_2d_prop = get_metric_2d(all_real_y_prop, all_pred_y_prop)
metric_2d_fixed = get_metric_2d(all_real_y_fixed, all_pred_y_fixed)

#%% 조건별 weight 1d 만들기

def calculate_weight(real_y_2d, is_normalized=False):
    NUM_BASE_UNIT = real_y_2d.shape[0]
    argsort_real_1d = np.argsort(real_y_2d.sum(axis=1))
    rank_value = NUM_BASE_UNIT - np.array([np.where(argsort_real_1d==xx)[0][0] for xx in np.arange(NUM_BASE_UNIT)])
    output = (0.5 + (1/(1+rank_value)))

    if is_normalized: return output / np.sum(output)
    else: return output


def calculate_weight1(real_y_2d):
    real_1d = np.sum(real_y_2d, axis=1)
    return float(0.5) + (real_1d / np.sum(real_1d))

def nor_weight(weight_1d):
    return weight_1d/ np.sum(weight_1d)

def weighted_mape(metric_1d, weight_1d):
    # metric, weight 전체를 불러온다
    # metric값이 nan인 경우를 찾는다
    # metric 값을 인덱싱 한다.weight 값을 인덱싱 한다.
    # weight 값을 대상으로 normalize 한다.
    # @ 한다

    non_nan_idx = np.where(np.isnan(metric_1d) != True)[0]
    return (metric_1d[non_nan_idx] @ nor_weight(weight_1d[non_nan_idx]))/np.sum(nor_weight(weight_1d[non_nan_idx]))


#weight_prop = calculate_weight(temp_obj['real_y'], False)
#weight_fixed = calculate_weight(temp_obj1['real_y'], False)


weight_prop1 = calculate_weight1(temp_obj['real_y'])
weight_fixed1 = calculate_weight1(temp_obj1['real_y'])


weight_fixed1.shape

#%% 샘플 구분에 따른 인덱스 구하기

fixed_idx, prop_idx = temp_obj2['bigger_ovl_idx'], temp_obj2['prop_ovl_idx']


sep_fixed_idx, sep_prop_idx = temp_obj2['bigger_sep_idx'], temp_obj2['prop_sep_idx']

#%%
"""weighted summation 방식으로, 모든 metric에 대해서 한번에 산출하기"""


def weighted_multi_metric(metric_2d, weight_1d):
    w_rmse = (metric_2d[:,0] @ weight_1d)/np.sum(weight_1d)
    w_mae = (metric_2d[:,1] @ weight_1d)/np.sum(weight_1d)
    w_mape = weighted_mape(metric_2d[:,2], weight_1d)
    w_smape = (metric_2d[:,3] @ weight_1d)/np.sum(weight_1d)
    w_smape1 = weighted_mape(metric_2d[:,4], weight_1d)

    return np.array([w_rmse, w_mae, w_mape, w_smape, w_smape1]).round(4)

"""모든 base unit 구분으로, 모든 metric에 대해서 한번에 산출하기"""

def weighted_multi_metric_level(metric_2d, weight_1d, ovl_idx, sep_idx):
    all_metrics = weighted_multi_metric(metric_2d, weight_1d)
    ovl_metrics = weighted_multi_metric(metric_2d[ovl_idx,:], weight_1d[ovl_idx])
    sep_metrics = weighted_multi_metric(metric_2d[sep_idx,:], weight_1d[sep_idx])

    return np.vstack((all_metrics, ovl_metrics, sep_metrics))


def uniform_multi_metric_level(metric_2d, ovl_idx, sep_idx):
    all_metrics = np.nanmean(metric_2d, axis=0)
    ovl_metrics = np.nanmean(metric_2d[ovl_idx,:], axis=0)
    sep_metrics = np.nanmean(metric_2d[sep_idx,:], axis=0)

    return np.vstack((all_metrics, ovl_metrics, sep_metrics)).round(4)

uniform_prop_score_df=pd.DataFrame(data=uniform_multi_metric_level(metric_2d_prop, prop_idx, sep_prop_idx),
columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])

uniform_fixed_score_df=pd.DataFrame(
    data=uniform_multi_metric_level(metric_2d_fixed, fixed_idx, sep_fixed_idx),
    columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])



#%%

prop_score_df = pd.DataFrame(data=weighted_multi_metric_level(metric_2d_prop,
                            weight_prop1,
                            prop_idx,
                            sep_prop_idx), columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])


fixed_score_df = pd.DataFrame(data=weighted_multi_metric_level(metric_2d_fixed,
                            weight_fixed1,\
                            fixed_idx,\
                            sep_fixed_idx), columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])
#%%


prop_score_df = prop_score_df.reset_index()
prop_score_df.loc[:,'cond'] = 'prop'

fixed_score_df = fixed_score_df.reset_index()
fixed_score_df.loc[:,'cond'] = 'fixed'


#(pd.concat([prop_score_df, fixed_score_df]).groupby(['index','cond']).mean()).to_csv(r'minneapolis_weighted_index_cond_0820.csv')

#%%



uniform_prop_score_df = uniform_prop_score_df.reset_index()
uniform_prop_score_df.loc[:,'cond'] = 'prop'

uniform_fixed_score_df = uniform_fixed_score_df.reset_index()
uniform_fixed_score_df.loc[:,'cond'] = 'fixed'

#(pd.concat([uniform_prop_score_df,uniform_fixed_score_df]).groupby(['index','cond']).mean()).to_csv(r'minneapolis_uniform_index_cond_0820.csv')





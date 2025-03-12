# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:57:16 2024

@author: dohyeon
"""

import user_utils as uu
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error as mape, mean_absolute_error as mae, mean_squared_error as mse
import matplotlib.pyplot as plt
from scipy import stats

save_date = '0812'
save_date1 = '0812'

def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum(np.nan_to_num((np.abs(pred_y-real_y)) / ((real_y+pred_y)/2),0))/real_y.shape[0]



def find_both_non_zero_idx(real_y, pred_y):
    return np.setdiff1d(np.arange(real_y.shape[0]), np.intersect1d(np.where(real_y==0)[0], np.where(pred_y==0)[0]))


#%%

m = '250m'

result_dic_rmse = {'prop':[], 'fixed':[]}
result_dic_mae = {'prop':[], 'fixed':[]}
result_dic_mape = {'prop':[], 'fixed':[]}
result_dic_smape = {'prop':[], 'fixed':[]}
result_dic_smape0 = {'prop':[], 'fixed':[]}


for subset_name in [0,1,2,3,4]:
    
    
    temp_obj = uu.load_gpickle(r'kansas_real_pred_y_prop_%s_%s.pickle'%(subset_name, save_date1))
    temp_obj1 = uu.load_gpickle(r'kansas_real_pred_y_fixed_%s_%s_%s.pickle'%(m, subset_name, save_date))

    
    temp_df = pd.DataFrame(data={'real':temp_obj['real_y'].reshape(-1,1).squeeze(), 'pred':temp_obj['pred_y'].reshape(-1,1).squeeze()})
    
    temp_df1 = pd.DataFrame(data={'real':temp_obj1['real_y'].reshape(-1,1).squeeze(), 'pred':temp_obj1['pred_y'].reshape(-1,1).squeeze()})


    #rmse
    result_dic_rmse['prop'].append(mse(temp_df['real'].values, temp_df['pred'].values)**0.5)
    result_dic_rmse['fixed'].append(mse(temp_df1['real'].values, temp_df1['pred'].values)**0.5)


    #mae
    result_dic_mae['prop'].append(mae(temp_df['real'].values, temp_df['pred'].values))
    result_dic_mae['fixed'].append(mae(temp_df1['real'].values, temp_df1['pred'].values))

    #mape
    result_dic_mape['prop'].append(mape(temp_df.loc[temp_df['real']>0]['real'].values, temp_df.loc[temp_df['real']>0]['pred'].values))
    result_dic_mape['fixed'].append(mape(temp_df1.loc[temp_df1['real']>0]['real'].values, temp_df1.loc[temp_df1['real']>0]['pred'].values))

    #smape
    result_dic_smape['prop'].append(get_smape(temp_df['real'].values[find_both_non_zero_idx(temp_df['real'].values, temp_df['pred'].values)], temp_df['pred'].values[find_both_non_zero_idx(temp_df['real'].values, temp_df['pred'].values)]))
    result_dic_smape['fixed'].append(get_smape(temp_df1['real'].values[find_both_non_zero_idx(temp_df1['real'].values, temp_df1['pred'].values)], temp_df1['pred'].values[find_both_non_zero_idx(temp_df1['real'].values, temp_df1['pred'].values)]))

    #smape0
    result_dic_smape0['prop'].append(get_smape(temp_df.loc[temp_df['real']>0]['real'].values, temp_df.loc[temp_df['real']>0]['pred'].values))
    result_dic_smape0['fixed'].append(get_smape(temp_df1.loc[temp_df1['real']>0]['real'].values, temp_df1.loc[temp_df1['real']>0]['pred'].values))




rmse_df = pd.concat([pd.DataFrame(data={'metric':['rmse',]*10, 'try':np.repeat(np.arange(5),2),'group':['%s'%(m),]*10}), pd.DataFrame(data=result_dic_rmse).stack().reset_index()[['level_1',0]]], axis=1)
rmse_df.columns = ['metric','try','group','type','score']


mae_df = pd.concat([pd.DataFrame(data={'metric':['mae',]*10, 'try':np.repeat(np.arange(5),2),'group':['%s'%(m),]*10}), pd.DataFrame(data=result_dic_mae).stack().reset_index()[['level_1',0]]], axis=1)
mae_df.columns = ['metric','try','group','type','score']


mape_df = pd.concat([pd.DataFrame(data={'metric':['mape',]*10, 'try':np.repeat(np.arange(5),2),'group':['%s'%(m),]*10}), pd.DataFrame(data=result_dic_mape).stack().reset_index()[['level_1',0]]], axis=1)
mape_df.columns = ['metric','try','group','type','score']


smape_df = pd.concat([pd.DataFrame(data={'metric':['smape',]*10, 'try':np.repeat(np.arange(5),2),'group':['%s'%(m),]*10}), pd.DataFrame(data=result_dic_smape).stack().reset_index()[['level_1',0]]], axis=1)
smape_df.columns = ['metric','try','group','type','score']

smape0_df = pd.concat([pd.DataFrame(data={'metric':['smape0',]*10, 'try':np.repeat(np.arange(5),2),'group':['%s'%(m),]*10}), pd.DataFrame(data=result_dic_smape0).stack().reset_index()[['level_1',0]]], axis=1)
smape0_df.columns = ['metric','try','group','type','score']



print(stats.ttest_rel(result_dic_rmse['fixed'], result_dic_rmse['prop']))
print(stats.ttest_rel(result_dic_mae['fixed'], result_dic_mae['prop']))
print(stats.ttest_rel(result_dic_mape['fixed'], result_dic_mape['prop']))
print(stats.ttest_rel(result_dic_smape['fixed'], result_dic_smape['prop']))
print(stats.ttest_rel(result_dic_smape0['fixed'], result_dic_smape0['prop']))

#final_df.groupby(['group','metric','type']).mean()['score'].loc['750m'].unstack()






final_df = pd.concat([rmse_df, mae_df, mape_df, smape_df, smape0_df]).reset_index(drop=True)

#%%
new_final_df = final_df.loc[final_df['type']=='fixed'][['metric','try','group','score']]
#final_df.to_csv(r'kansas_prediction_performance_750m_prop_all_sep_ovl_0812_groupby.csv')


#%%

pd.concat([final_df.loc[final_df['type']=='fixed'][['metric','try','group','score']], new_final_df]).reset_index(drop=True).to_csv(r'kansas_prediction_performance_250m_750m_0812.csv')

























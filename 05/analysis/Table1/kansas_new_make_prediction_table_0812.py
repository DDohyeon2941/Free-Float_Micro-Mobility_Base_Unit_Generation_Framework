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

result_dic_rmse = {'prop':[], 'fixed':[]}
result_dic_mae = {'prop':[], 'fixed':[]}
result_dic_mape = {'prop':[], 'fixed':[]}
result_dic_smape = {'prop':[], 'fixed':[]}
result_dic_smape0 = {'prop':[], 'fixed':[]}


for subset_name in [0,1,2,3,4]:
    
    
    temp_obj = uu.load_gpickle(r'kansas_real_pred_y_prop_%s_%s.pickle'%(subset_name, save_date1))
    temp_obj1 = uu.load_gpickle(r'kansas_real_pred_y_fixed_%s_%s.pickle'%(subset_name, save_date))

    
    temp_df = pd.DataFrame(data={'real':temp_obj['real_y'].reshape(-1,1).squeeze(), 'pred':temp_obj['pred_y'].reshape(-1,1).squeeze()})
    
    temp_df1 = pd.DataFrame(data={'real':temp_obj1['real_y'].reshape(-1,1).squeeze(), 'pred':temp_obj1['pred_y'].reshape(-1,1).squeeze()})

    print(np.sum(temp_df.pred.values[np.where(temp_df.real.values==0)[0]]==0)  / (temp_df.real.values == 0).sum(),
          np.sum(temp_df1.pred.values[np.where(temp_df1.real.values==0)[0]]==0)  / (temp_df1.real.values == 0).sum())
    print('-'*50)
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




rmse_df = pd.concat([pd.DataFrame(data={'metric':['rmse',]*10, 'try':np.repeat(np.arange(5),2),'group':['all',]*10}), pd.DataFrame(data=result_dic_rmse).stack().reset_index()[['level_1',0]]], axis=1)
rmse_df.columns = ['metric','try','group','type','score']


mae_df = pd.concat([pd.DataFrame(data={'metric':['mae',]*10, 'try':np.repeat(np.arange(5),2),'group':['all',]*10}), pd.DataFrame(data=result_dic_mae).stack().reset_index()[['level_1',0]]], axis=1)
mae_df.columns = ['metric','try','group','type','score']


mape_df = pd.concat([pd.DataFrame(data={'metric':['mape',]*10, 'try':np.repeat(np.arange(5),2),'group':['all',]*10}), pd.DataFrame(data=result_dic_mape).stack().reset_index()[['level_1',0]]], axis=1)
mape_df.columns = ['metric','try','group','type','score']


smape_df = pd.concat([pd.DataFrame(data={'metric':['smape',]*10, 'try':np.repeat(np.arange(5),2),'group':['all',]*10}), pd.DataFrame(data=result_dic_smape).stack().reset_index()[['level_1',0]]], axis=1)
smape_df.columns = ['metric','try','group','type','score']

smape0_df = pd.concat([pd.DataFrame(data={'metric':['smape0',]*10, 'try':np.repeat(np.arange(5),2),'group':['all',]*10}), pd.DataFrame(data=result_dic_smape0).stack().reset_index()[['level_1',0]]], axis=1)
smape0_df.columns = ['metric','try','group','type','score']



print(stats.ttest_rel(result_dic_rmse['fixed'], result_dic_rmse['prop']))
print(stats.ttest_rel(result_dic_mae['fixed'], result_dic_mae['prop']))
print(stats.ttest_rel(result_dic_mape['fixed'], result_dic_mape['prop']))
print(stats.ttest_rel(result_dic_smape['fixed'], result_dic_smape['prop']))
print(stats.ttest_rel(result_dic_smape0['fixed'], result_dic_smape0['prop']))



#%%

index_obj = uu.load_gpickle(r'kansas_new_overlapped_index_fixed_prop_0812.pickle')


prop_sep_idx = index_obj['prop_separate_index']
prop_ovl_idx = index_obj['prop_overlapped_index']

fixed_sep_idx = index_obj['fixed_separate_index']
fixed_ovl_idx = index_obj['fixed_overlapped_index']


real_arr= temp_obj['real_y'][prop_sep_idx,:].reshape(-1,1).squeeze()
pred_arr= temp_obj['pred_y'][prop_sep_idx,:].reshape(-1,1).squeeze()
real_arr1= temp_obj1['real_y'][fixed_sep_idx,:].reshape(-1,1).squeeze()
pred_arr1= temp_obj1['pred_y'][fixed_sep_idx,:].reshape(-1,1).squeeze()



ovl_real_arr= temp_obj['real_y'][prop_ovl_idx,:].reshape(-1,1).squeeze()
ovl_pred_arr= temp_obj['pred_y'][prop_ovl_idx,:].reshape(-1,1).squeeze()
ovl_real_arr1= temp_obj1['real_y'][fixed_ovl_idx,:].reshape(-1,1).squeeze()
ovl_pred_arr1= temp_obj1['pred_y'][fixed_ovl_idx,:].reshape(-1,1).squeeze()


#%%



sep_result_dic_rmse = {'prop':[], 'fixed':[]}
ovl_result_dic_rmse = {'prop':[], 'fixed':[]}

sep_result_dic_mae = {'prop':[], 'fixed':[]}
ovl_result_dic_mae = {'prop':[], 'fixed':[]}

sep_result_dic_mape = {'prop':[], 'fixed':[]}
ovl_result_dic_mape = {'prop':[], 'fixed':[]}

sep_result_dic_smape = {'prop':[], 'fixed':[]}
ovl_result_dic_smape = {'prop':[], 'fixed':[]}

sep_result_dic_smape0 = {'prop':[], 'fixed':[]}
ovl_result_dic_smape0 = {'prop':[], 'fixed':[]}



for subset_name in [0,1,2,3,4]:
    
    
    temp_obj = uu.load_gpickle(r'kansas_real_pred_y_prop_%s_%s.pickle'%(subset_name, save_date1))
    temp_obj1 = uu.load_gpickle(r'kansas_real_pred_y_fixed_%s_%s.pickle'%(subset_name, save_date))


    real_arr= temp_obj['real_y'][prop_sep_idx,:].reshape(-1,1).squeeze()
    pred_arr= temp_obj['pred_y'][prop_sep_idx,:].reshape(-1,1).squeeze()
    real_arr1= temp_obj1['real_y'][fixed_sep_idx,:].reshape(-1,1).squeeze()
    pred_arr1= temp_obj1['pred_y'][fixed_sep_idx,:].reshape(-1,1).squeeze()
    
    
    
    ovl_real_arr= temp_obj['real_y'][prop_ovl_idx,:].reshape(-1,1).squeeze()
    ovl_pred_arr= temp_obj['pred_y'][prop_ovl_idx,:].reshape(-1,1).squeeze()
    ovl_real_arr1= temp_obj1['real_y'][fixed_ovl_idx,:].reshape(-1,1).squeeze()
    ovl_pred_arr1= temp_obj1['pred_y'][fixed_ovl_idx,:].reshape(-1,1).squeeze()
    

    #rmse
    sep_result_dic_rmse['prop'].append(mse(real_arr, pred_arr)**0.5)
    sep_result_dic_rmse['fixed'].append(mse(real_arr1, pred_arr1)**0.5)
    ovl_result_dic_rmse['prop'].append(mse(ovl_real_arr, ovl_pred_arr)**0.5)
    ovl_result_dic_rmse['fixed'].append(mse(ovl_real_arr1, ovl_pred_arr1)**0.5)


    #mae
    sep_result_dic_mae['prop'].append(mae(real_arr, pred_arr))
    sep_result_dic_mae['fixed'].append(mae(real_arr1, pred_arr1))
    ovl_result_dic_mae['prop'].append(mae(ovl_real_arr, ovl_pred_arr))
    ovl_result_dic_mae['fixed'].append(mae(ovl_real_arr1, ovl_pred_arr1))


    #mape
    sep_result_dic_mape['prop'].append(mape(real_arr[real_arr>0], pred_arr[real_arr>0]))
    sep_result_dic_mape['fixed'].append(mape(real_arr1[real_arr1>0], pred_arr1[real_arr1>0]))
    ovl_result_dic_mape['prop'].append(mape(ovl_real_arr[ovl_real_arr>0], ovl_pred_arr[ovl_real_arr>0]))
    ovl_result_dic_mape['fixed'].append(mape(ovl_real_arr1[ovl_real_arr1>0], ovl_pred_arr1[ovl_real_arr1>0]))

    #smape
    sep_result_dic_smape['prop'].append(get_smape(real_arr[find_both_non_zero_idx(real_arr, pred_arr)], pred_arr[find_both_non_zero_idx(real_arr, pred_arr)]))
    sep_result_dic_smape['fixed'].append(get_smape(real_arr1[find_both_non_zero_idx(real_arr1, pred_arr1)], pred_arr1[find_both_non_zero_idx(real_arr1, pred_arr1)]))
    ovl_result_dic_smape['prop'].append(get_smape(ovl_real_arr[find_both_non_zero_idx(ovl_real_arr, ovl_pred_arr)], ovl_pred_arr[find_both_non_zero_idx(ovl_real_arr, ovl_pred_arr)]))
    ovl_result_dic_smape['fixed'].append(get_smape(ovl_real_arr1[find_both_non_zero_idx(ovl_real_arr1, ovl_pred_arr1)], ovl_pred_arr1[find_both_non_zero_idx(ovl_real_arr1, ovl_pred_arr1)]))


    #smape0
    sep_result_dic_smape0['prop'].append(get_smape(real_arr[real_arr>0], pred_arr[real_arr>0]))
    sep_result_dic_smape0['fixed'].append(get_smape(real_arr1[real_arr1>0], pred_arr1[real_arr1>0]))
    ovl_result_dic_smape0['prop'].append(get_smape(ovl_real_arr[ovl_real_arr>0], ovl_pred_arr[ovl_real_arr>0]))
    ovl_result_dic_smape0['fixed'].append(get_smape(ovl_real_arr1[ovl_real_arr1>0], ovl_pred_arr1[ovl_real_arr1>0]))
    #print((real_arr==0).sum()/ real_arr.shape[0], (real_arr1==0).sum()/ real_arr1.shape[0])
    print(np.sum(pred_arr[np.where(real_arr==0)[0]]==0)  / pred_arr[np.where(real_arr==0)[0]].shape[0],
          np.sum(pred_arr1[np.where(real_arr1==0)[0]]==0)  / pred_arr1[np.where(real_arr1==0)[0]].shape[0])

    print(np.sum(ovl_pred_arr[np.where(ovl_real_arr==0)[0]]==0)  / ovl_pred_arr[np.where(ovl_real_arr==0)[0]].shape[0],
          np.sum(ovl_pred_arr1[np.where(ovl_real_arr1==0)[0]]==0)  / ovl_pred_arr1[np.where(ovl_real_arr1==0)[0]].shape[0])
    print('-'*100)



#rmse
sep_rmse_df = pd.concat([pd.DataFrame(data={'metric':['rmse',]*10, 'try':np.repeat(np.arange(5),2),'group':['sep',]*10}), pd.DataFrame(data=sep_result_dic_rmse).stack().reset_index()[['level_1',0]]], axis=1)
sep_rmse_df.columns = ['metric','try','group','type','score']

ovl_rmse_df = pd.concat([pd.DataFrame(data={'metric':['rmse',]*10, 'try':np.repeat(np.arange(5),2),'group':['ovl',]*10}), pd.DataFrame(data=ovl_result_dic_rmse).stack().reset_index()[['level_1',0]]], axis=1)
ovl_rmse_df.columns = ['metric','try','group','type','score']

#mae

sep_mae_df = pd.concat([pd.DataFrame(data={'metric':['mae',]*10, 'try':np.repeat(np.arange(5),2),'group':['sep',]*10}), pd.DataFrame(data=sep_result_dic_mae).stack().reset_index()[['level_1',0]]], axis=1)
sep_mae_df.columns = ['metric','try','group','type','score']

ovl_mae_df = pd.concat([pd.DataFrame(data={'metric':['mae',]*10, 'try':np.repeat(np.arange(5),2),'group':['ovl',]*10}), pd.DataFrame(data=ovl_result_dic_mae).stack().reset_index()[['level_1',0]]], axis=1)
ovl_mae_df.columns = ['metric','try','group','type','score']


#mape

sep_mape_df = pd.concat([pd.DataFrame(data={'metric':['mape',]*10, 'try':np.repeat(np.arange(5),2),'group':['sep',]*10}), pd.DataFrame(data=sep_result_dic_mape).stack().reset_index()[['level_1',0]]], axis=1)
sep_mape_df.columns = ['metric','try','group','type','score']


ovl_mape_df = pd.concat([pd.DataFrame(data={'metric':['mape',]*10, 'try':np.repeat(np.arange(5),2),'group':['ovl',]*10}), pd.DataFrame(data=ovl_result_dic_mape).stack().reset_index()[['level_1',0]]], axis=1)
ovl_mape_df.columns = ['metric','try','group','type','score']


#smape
sep_smape_df = pd.concat([pd.DataFrame(data={'metric':['smape',]*10, 'try':np.repeat(np.arange(5),2),'group':['sep',]*10}), pd.DataFrame(data=sep_result_dic_smape).stack().reset_index()[['level_1',0]]], axis=1)
sep_smape_df.columns = ['metric','try','group','type','score']

ovl_smape_df = pd.concat([pd.DataFrame(data={'metric':['smape',]*10, 'try':np.repeat(np.arange(5),2),'group':['ovl',]*10}), pd.DataFrame(data=ovl_result_dic_smape).stack().reset_index()[['level_1',0]]], axis=1)
ovl_smape_df.columns = ['metric','try','group','type','score']


#smape0

sep_smape0_df = pd.concat([pd.DataFrame(data={'metric':['smape0',]*10, 'try':np.repeat(np.arange(5),2),'group':['sep',]*10}), pd.DataFrame(data=sep_result_dic_smape0).stack().reset_index()[['level_1',0]]], axis=1)
sep_smape0_df.columns = ['metric','try','group','type','score']

ovl_smape0_df = pd.concat([pd.DataFrame(data={'metric':['smape0',]*10, 'try':np.repeat(np.arange(5),2),'group':['ovl',]*10}), pd.DataFrame(data=ovl_result_dic_smape0).stack().reset_index()[['level_1',0]]], axis=1)
ovl_smape0_df.columns = ['metric','try','group','type','score']

### cocnat
#%%
pd.concat([rmse_df, mae_df, mape_df, smape_df, smape0_df,sep_rmse_df, ovl_rmse_df, sep_mae_df, ovl_mae_df, sep_mape_df, ovl_mape_df, sep_smape_df, ovl_smape_df,  sep_smape0_df, ovl_smape0_df]).reset_index(drop=True).to_csv(r'kansas_prediction_performance_500m_prop_all_sep_ovl_0812_rev.csv', index=False)


###final
final_df = pd.concat([rmse_df, mae_df, mape_df, smape_df, smape0_df,sep_rmse_df, ovl_rmse_df, sep_mae_df, ovl_mae_df, sep_mape_df, ovl_mape_df, sep_smape_df, ovl_smape_df,  sep_smape0_df, ovl_smape0_df]).reset_index(drop=True)

final_df.groupby(['group','metric','type']).mean()['score'].unstack(1)[['rmse','mae','mape','smape','smape0']].to_csv(r'kansas_prediction_performance_500m_prop_all_sep_ovl_0812_groupby_rev.csv')


### check statistics


print(stats.ttest_rel(sep_result_dic_rmse['fixed'], sep_result_dic_rmse['prop']))
print(stats.ttest_rel(ovl_result_dic_rmse['fixed'], ovl_result_dic_rmse['prop']))

print(stats.ttest_rel(sep_result_dic_mae['fixed'], sep_result_dic_mae['prop']))
print(stats.ttest_rel(ovl_result_dic_mae['fixed'], ovl_result_dic_mae['prop']))

print(stats.ttest_rel(sep_result_dic_mape['fixed'], sep_result_dic_mape['prop']))
print(stats.ttest_rel(ovl_result_dic_mape['fixed'], ovl_result_dic_mape['prop']))

print(stats.ttest_rel(sep_result_dic_smape['fixed'], sep_result_dic_smape['prop']))
print(stats.ttest_rel(ovl_result_dic_smape['fixed'], ovl_result_dic_smape['prop']))

print(stats.ttest_rel(sep_result_dic_smape0['fixed'], sep_result_dic_smape0['prop']))
print(stats.ttest_rel(ovl_result_dic_smape0['fixed'], ovl_result_dic_smape0['prop']))



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
import matplotlib.pyplot as plt


temp_obj = uu.load_gpickle(r'kansas_real_pred_y_prop_0824.pickle')
temp_obj1 = uu.load_gpickle(r'kansas_real_pred_y_fixed_0824.pickle')

temp_obj2 = uu.load_gpickle(r'kansas_overlapped_index_fixed_prop_0809.pickle')




#%%

all_pred_y_prop, all_real_y_prop = temp_obj['pred_y'], temp_obj['real_y']
all_pred_y_fixed, all_real_y_fixed = temp_obj1['pred_y'], temp_obj1['real_y']


#%%

plt.plot(np.sort(np.mean(all_pred_y_prop, axis=1)),c='g'), plt.plot(np.sort(np.mean(all_pred_y_fixed, axis=1)))
plt.plot(np.sort(np.mean(all_real_y_prop, axis=1)),c='g'), plt.plot(np.sort(np.mean(all_real_y_fixed, axis=1)))


#%%


crr1 = (all_real_y_fixed == 0).sum(axis=1) / all_real_y_fixed.shape[1]
crr2 = (all_real_y_prop == 0).sum(axis=1) / all_real_y_prop.shape[1]

drr1 = (all_pred_y_fixed == 0).sum(axis=1) / all_pred_y_fixed.shape[1]
drr2 = (all_pred_y_prop == 0).sum(axis=1) / all_pred_y_prop.shape[1]



crr11 = (all_real_y_fixed[sep_fixed_idx,:] == 0).sum(axis=1) / all_real_y_fixed[sep_fixed_idx,:].shape[1]
crr21 = (all_real_y_prop[sep_prop_idx,:] == 0).sum(axis=1) / all_real_y_prop[sep_prop_idx,:].shape[1]


plt.boxplot({1:crr1, 2:crr2}.values(), showfliers=True)
plt.boxplot({1:drr1, 2:drr2}.values(), showfliers=True)


plt.boxplot({1:crr1, 2:drr1, 3:crr2, 4:drr2}.values(), showfliers=True)


plt.boxplot({1:crr11, 2:crr21}.values(), showfliers=True)


def count_real_0_as_0(real_1d, pred_1d):
    return np.intersect1d(np.where(real_1d==0)[0],np.where(pred_1d==0)[0]).shape[0]


frr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed, all_pred_y_fixed)])
frr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop, all_pred_y_prop)])


plt.boxplot({1:frr1, 2:frr2}.values(), showfliers=True)








#%% 샘플 구분에 따른 인덱스 구하기

fixed_idx, prop_idx = temp_obj2['fixed_grid_index'], temp_obj2['prop_grid_index']


sep_fixed_idx, sep_prop_idx = np.setdiff1d(np.setdiff1d(np.arange(temp_obj1['real_y'].shape[0]), fixed_idx), fixed_idx), np.setdiff1d(np.arange(temp_obj['real_y'].shape[0]), prop_idx)

#%%

grr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed[sep_fixed_idx,:], all_pred_y_fixed[sep_fixed_idx,:])])
grr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop[sep_prop_idx,:], all_pred_y_prop[sep_prop_idx,:])])


plt.boxplot({1:grr1, 2:grr2}.values(), showfliers=True)




#%%


hrr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed[fixed_idx,:], all_pred_y_fixed[fixed_idx,:])])
hrr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop[prop_idx,:], all_pred_y_prop[prop_idx,:])])


plt.boxplot({1:hrr1, 2:hrr2}.values(), showfliers=True)





#%%



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


def weighted_multi_metric_level1(metric_2d, weight_1d, ovl_idx, sep_idx):
    all_idx1 = np.sort(np.append(ovl_idx, sep_idx))
    all_metrics = weighted_multi_metric(metric_2d[all_idx1,:], weight_1d[all_idx1])
    ovl_metrics = weighted_multi_metric(metric_2d[ovl_idx,:], weight_1d[ovl_idx])
    sep_metrics = weighted_multi_metric(metric_2d[sep_idx,:], weight_1d[sep_idx])

    return np.vstack((all_metrics, ovl_metrics, sep_metrics))


def uniform_multi_metric_level(metric_2d, ovl_idx, sep_idx):
    all_metrics = np.nanmean(metric_2d, axis=0)
    ovl_metrics = np.nanmean(metric_2d[ovl_idx,:], axis=0)
    sep_metrics = np.nanmean(metric_2d[sep_idx,:], axis=0)

    return np.vstack((all_metrics, ovl_metrics, sep_metrics)).round(4)

def uniform_multi_metric_level1(metric_2d, ovl_idx, sep_idx):
    all_idx1 = np.sort(np.append(ovl_idx, sep_idx))
    all_metrics = np.nanmean(metric_2d[all_idx1, :], axis=0)
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


(pd.concat([prop_score_df, fixed_score_df]).groupby(['index','cond']).mean()).to_csv(r'kansas_weighted_index_cond_0824.csv')

#%%



uniform_prop_score_df = uniform_prop_score_df.reset_index()
uniform_prop_score_df.loc[:,'cond'] = 'prop'

uniform_fixed_score_df = uniform_fixed_score_df.reset_index()
uniform_fixed_score_df.loc[:,'cond'] = 'fixed'

(pd.concat([uniform_prop_score_df,uniform_fixed_score_df]).groupby(['index','cond']).mean()).to_csv(r'kansas_uniform_index_cond_0824.csv')


#%%





































#%%



uniform_prop_score_big_df=pd.DataFrame(data=uniform_multi_metric_level(metric_2d_prop, prop_idx, sep_prop_idx[sep_prop_idx<45]),
columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])

uniform_prop_score_big_df1=pd.DataFrame(data=uniform_multi_metric_level1(metric_2d_prop, prop_idx, sep_prop_idx[sep_prop_idx<25]),
columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])



prop_score_big_df = pd.DataFrame(data=weighted_multi_metric_level1(metric_2d_prop,
                            weight_prop1,
                            prop_idx,
                            sep_prop_idx[sep_prop_idx<25]), columns=['RMSE','MAE','MAPE','SMAPE','SMAPE_zero'], index=['all','ovl','sep'])


uniform_prop_score_big_df
uniform_prop_score_big_df1

uniform_fixed_score_df
uniform_prop_score_df

prop_score_big_df

#%%
#plt.hist(np.mean(all_pred_y_fixed, axis=1)[sep_fixed_idx], alpha=0.5, bins=30)
plt.hist(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]], alpha=0.5, bins=30)

plt.hist(np.mean(all_pred_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]], alpha=0.5, bins=30)
#plt.hist(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]])

#%%
plt.boxplot({1:np.mean(all_pred_y_fixed, axis=1)[sep_fixed_idx], 2:np.mean(all_pred_y_prop, axis=1)[sep_prop_idx]}.values(),  showfliers=False)




#%%

plt.plot(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]], c='b')
plt.plot(np.mean(all_pred_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]], c='g')

#%%

plt.plot(np.mean(all_real_y_prop, axis=1)[prop_idx], c='b')
plt.plot(np.mean(all_pred_y_prop, axis=1)[prop_idx], c='g')

#%%

plt.plot(np.mean(all_real_y_prop, axis=1)[sep_prop_idx], c='b')
plt.plot(np.mean(all_pred_y_prop, axis=1)[sep_prop_idx], c='g')



#%%

plt.plot(np.mean(all_real_y_fixed, axis=1)[sep_fixed_idx], c='b')
plt.plot(np.mean(all_pred_y_fixed, axis=1)[sep_fixed_idx], c='g')

#%%
metric_2d_prop[sep_prop_idx[sep_prop_idx>=25],-1]
metric_2d_fixed[sep_fixed_idx,:][:,]




metric_2d_prop[prop_idx,:][:,-1]
#%%


plt.hist(np.mean(all_real_y_prop, axis=1)[prop_idx], alpha=0.5, bins=30)

plt.hist(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>25]], alpha=0.5, bins=30)

#%%

plt.plot(np.mean(all_real_y_prop, axis=1)[prop_idx], c='b')
plt.plot(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>=25]], c='g')

#%%

plt.boxplot({1:np.mean(all_pred_y_prop, axis=1)[prop_idx], 2:np.mean(all_pred_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>=25]]}.values(),  showfliers=True)



#%%

(np.mean(all_real_y_prop, axis=1) < 1).sum()
(np.mean(all_real_y_fixed, axis=1) < 1).sum()


plt.plot(np.sort(np.mean(all_real_y_prop, axis=1)[prop_idx]), c='b')

plt.plot((np.mean(all_pred_y_prop, axis=1)[prop_idx])[np.argsort(np.mean(all_real_y_prop, axis=1)[prop_idx])], c='b', ls='--')

plt.plot((np.mean(all_pred_y_fixed, axis=1)[fixed_idx])[np.argsort(np.mean(all_real_y_fixed, axis=1)[fixed_idx])], c='r', ls='--')


#%%
plt.plot(np.sort(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>=25]]), c='g')
plt.plot((np.mean(all_pred_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>=25]])[np.argsort(np.mean(all_real_y_prop, axis=1)[sep_prop_idx[sep_prop_idx>=25]])], c='g', ls='--')


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



#%%

temp_obj = uu.load_gpickle(r'minneapolis_real_pred_y_prop_0820.pickle')
temp_obj1 = uu.load_gpickle(r'minneapolis_real_pred_y_fixed_0820.pickle')

temp_obj2 = uu.load_gpickle(r'minneapolis_fixed_prop_index_0809.pickle')


#%%

all_pred_y_prop, all_real_y_prop = temp_obj['pred_y'], temp_obj['real_y']
all_pred_y_fixed, all_real_y_fixed = temp_obj1['pred_y'], temp_obj1['real_y']

#%%


crr1 = (all_real_y_fixed == 0).sum(axis=1) / all_real_y_fixed.shape[1]
crr2 = (all_real_y_prop == 0).sum(axis=1) / all_real_y_prop.shape[1]

drr1 = (all_pred_y_fixed == 0).sum(axis=1) / all_pred_y_fixed.shape[1]
drr2 = (all_pred_y_prop == 0).sum(axis=1) / all_pred_y_prop.shape[1]

plt.boxplot({1:crr1, 2:drr1, 3:crr2, 4:drr2}.values(), showfliers=True)

#%%

plt.plot(np.sort(np.mean(all_pred_y_prop, axis=1)),c='g'), plt.plot(np.sort(np.mean(all_pred_y_fixed, axis=1)))
plt.plot(np.sort(np.mean(all_real_y_prop, axis=1)),c='g'), plt.plot(np.sort(np.mean(all_real_y_fixed, axis=1)))


#%% 샘플 구분에 따른 인덱스 구하기

fixed_idx, prop_idx = temp_obj2['bigger_ovl_idx'], temp_obj2['prop_ovl_idx']


sep_fixed_idx, sep_prop_idx = temp_obj2['bigger_sep_idx'], temp_obj2['prop_sep_idx']

#%%

def count_real_0_as_0(real_1d, pred_1d):
    return np.intersect1d(np.where(real_1d==0)[0],np.where(pred_1d==0)[0]).shape[0]


frr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed, all_pred_y_fixed)])
frr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop, all_pred_y_prop)])


plt.boxplot({1:frr1, 2:frr2}.values(), showfliers=True)

#%%

grr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed[sep_fixed_idx,:], all_pred_y_fixed[sep_fixed_idx,:])])
grr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop[sep_prop_idx,:], all_pred_y_prop[sep_prop_idx,:])])


plt.boxplot({1:grr1, 2:grr2}.values(), showfliers=True)





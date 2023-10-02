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


min_prop_obj = uu.load_gpickle(r'../Minneapolis/minneapolis_real_pred_y_prop_0820.pickle')
min_fixed_obj = uu.load_gpickle(r'../Minneapolis/minneapolis_real_pred_y_fixed_0820.pickle')
#%%

all_pred_y_prop, all_real_y_prop = temp_obj['pred_y'], temp_obj['real_y']
all_pred_y_fixed, all_real_y_fixed = temp_obj1['pred_y'], temp_obj1['real_y']


all_pred_y_prop_min, all_real_y_prop_min = min_prop_obj['pred_y'], min_prop_obj['real_y']
all_pred_y_fixed_min, all_real_y_fixed_min = min_fixed_obj['pred_y'], min_fixed_obj['real_y']


def count_real_0_as_0(real_1d, pred_1d):
    return np.intersect1d(np.where(real_1d==0)[0],np.where(pred_1d==0)[0]).shape[0]


frr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed, all_pred_y_fixed)])
frr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop, all_pred_y_prop)])


drr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed_min, all_pred_y_fixed_min)])
drr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_prop_min, all_pred_y_prop_min)])



#plt.boxplot({1:frr1, 2:frr2}.values(), showfliers=True)

#%%
from matplotlib.patches import Patch

bp=plt.boxplot({1:frr1, 2:frr2, 3:drr1, 4:drr2}.values(), positions=[1, 2, 4, 5], widths=0.6, showfliers=False)
colors = ['green', 'red']


for i, color in enumerate(colors * 2):
    plt.setp(bp['boxes'][i], color=color)
    plt.setp(bp['whiskers'][i*2], color=color)
    plt.setp(bp['whiskers'][i*2 + 1], color=color)
    plt.setp(bp['caps'][i*2], color=color)
    plt.setp(bp['caps'][i*2 + 1], color=color)
    plt.setp(bp['medians'][i], color=color)

legend_elements = [Patch(facecolor='green', edgecolor='green', label='500m'),
                   Patch(facecolor='red', edgecolor='red', label='Prop')]
plt.legend(handles=legend_elements)

# x 축 레이블 설정
plt.xticks([1.5, 4.5], ['Kansas City', 'Minneapolis'])
plt.ylabel('Correctly predicted Zero Ratio')
plt.xlabel('City')

plt.show()



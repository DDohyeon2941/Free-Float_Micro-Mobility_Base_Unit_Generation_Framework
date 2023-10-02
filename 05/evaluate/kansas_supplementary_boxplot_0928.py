# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import user_utils as uu


temp_obj_prop = uu.load_gpickle(r'../test/kansas_real_pred_y_prop_0824.pickle')
temp_obj_fixed = uu.load_gpickle(r'../test/kansas_real_pred_y_fixed_0824.pickle')


all_pred_y_prop, all_real_y_prop = temp_obj_prop['pred_y'], temp_obj_prop['real_y']
all_pred_y_fixed, all_real_y_fixed = temp_obj_fixed['pred_y'], temp_obj_fixed['real_y']





#%%




crr1 = (all_real_y_prop == 0).sum(axis=1) / all_real_y_prop.shape[1]
crr2 = (all_real_y_fixed == 0).sum(axis=1) / all_real_y_fixed.shape[1]


drr1 = (all_pred_y_prop == 0).sum(axis=1) / all_pred_y_prop.shape[1]
drr2 = (all_pred_y_fixed == 0).sum(axis=1) / all_pred_y_fixed.shape[1]


#%%



# 박스플롯 생성
bp = plt.boxplot([crr2, drr2, crr1, drr1], positions=[1, 2, 4, 5], widths=0.6, showfliers=False)

# 색상 지정
colors = ['green', 'red']


for i, color in enumerate(colors * 2):
    plt.setp(bp['boxes'][i], color=color)
    plt.setp(bp['whiskers'][i*2], color=color)
    plt.setp(bp['whiskers'][i*2 + 1], color=color)
    plt.setp(bp['caps'][i*2], color=color)
    plt.setp(bp['caps'][i*2 + 1], color=color)
    plt.setp(bp['medians'][i], color=color)

legend_elements = [Patch(facecolor='green', edgecolor='green', label='Real'),
                   Patch(facecolor='red', edgecolor='red', label='Predicted')]
plt.legend(handles=legend_elements, loc='upper right', prop={'size': 8.5})

# x 축 레이블 설정
plt.xticks([1.5, 4.5], ['500m', 'Prop'])
plt.ylabel('Proportion')
plt.xlabel('Method')

plt.show()

#%%

















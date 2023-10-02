# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import user_utils as uu




temp_obj_250m = uu.load_gpickle(r'../test/kansas_real_pred_y_fixed_250m_0824.pickle')
temp_obj_750m = uu.load_gpickle(r'../test/kansas_real_pred_y_fixed_750m_0824.pickle')
temp_obj_500m = uu.load_gpickle(r'../test/kansas_real_pred_y_fixed_0824.pickle')


all_pred_y_fixed_250m, all_real_y_fixed_250m = temp_obj_250m['pred_y'], temp_obj_250m['real_y']
all_pred_y_fixed_750m, all_real_y_fixed_750m = temp_obj_750m['pred_y'], temp_obj_750m['real_y']
all_pred_y_fixed_500m, all_real_y_fixed_500m = temp_obj_500m['pred_y'], temp_obj_500m['real_y']





#%%




crr1 = (all_real_y_fixed_250m == 0).sum(axis=1) / all_real_y_fixed_250m.shape[1]
crr2 = (all_real_y_fixed_500m == 0).sum(axis=1) / all_real_y_fixed_500m.shape[1]
crr3 = (all_real_y_fixed_750m == 0).sum(axis=1) / all_real_y_fixed_750m.shape[1]


drr1 = (all_pred_y_fixed_250m == 0).sum(axis=1) / all_pred_y_fixed_250m.shape[1]
drr2 = (all_pred_y_fixed_500m == 0).sum(axis=1) / all_pred_y_fixed_500m.shape[1]
drr3 = (all_pred_y_fixed_750m == 0).sum(axis=1) / all_pred_y_fixed_750m.shape[1]


#%%



# 박스플롯 생성
bp = plt.boxplot([crr1, drr1, crr2, drr2, crr3, drr3], positions=[1, 2, 4, 5, 7, 8], widths=0.6, showfliers=False)

# 색상 지정
colors = ['blue', 'orange']


for i, color in enumerate(colors * 3):
    plt.setp(bp['boxes'][i], color=color)
    plt.setp(bp['whiskers'][i*2], color=color)
    plt.setp(bp['whiskers'][i*2 + 1], color=color)
    plt.setp(bp['caps'][i*2], color=color)
    plt.setp(bp['caps'][i*2 + 1], color=color)
    plt.setp(bp['medians'][i], color=color)

legend_elements = [Patch(facecolor='blue', edgecolor='blue', label='Real'),
                   Patch(facecolor='orange', edgecolor='orange', label='Predicted')]
plt.legend(handles=legend_elements)

# x 축 레이블 설정
plt.xticks([1.5, 4.5, 7.5], ['250m', '500m', '750m'])
plt.ylabel('Proportion')
plt.xlabel('Width')

plt.show()

#%%

















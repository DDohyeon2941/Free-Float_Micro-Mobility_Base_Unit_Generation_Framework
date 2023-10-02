# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import user_utils as uu


#%%




temp_obj_250m = uu.load_gpickle(r'../test/minneapolis_real_pred_y_fixed_250m_0820.pickle')
temp_obj_750m = uu.load_gpickle(r'../test/minneapolis_real_pred_y_fixed_750m_0820.pickle')
temp_obj_500m = uu.load_gpickle(r'../test/minneapolis_real_pred_y_fixed_0820.pickle')


all_pred_y_fixed_250m, all_real_y_fixed_250m = temp_obj_250m['pred_y'], temp_obj_250m['real_y']
all_pred_y_fixed_750m, all_real_y_fixed_750m = temp_obj_750m['pred_y'], temp_obj_750m['real_y']
all_pred_y_fixed_500m, all_real_y_fixed_500m = temp_obj_500m['pred_y'], temp_obj_500m['real_y']







all_pred_y_fixed_250m.shape


(((all_real_y_fixed_250m == 0).sum(axis=1) / all_real_y_fixed_250m.shape[1])==1).sum()

(((all_real_y_fixed_500m == 0).sum(axis=1) / all_real_y_fixed_500m.shape[1])==1).sum()

(((all_real_y_fixed_750m == 0).sum(axis=1) / all_real_y_fixed_750m.shape[1])==1).sum()




crr1 = (all_real_y_fixed_250m == 0).sum(axis=1) / all_real_y_fixed_250m.shape[1]
crr2 = (all_real_y_fixed_500m == 0).sum(axis=1) / all_real_y_fixed_500m.shape[1]
crr3 = (all_real_y_fixed_750m == 0).sum(axis=1) / all_real_y_fixed_750m.shape[1]


drr1 = (all_pred_y_fixed_250m == 0).sum(axis=1) / all_pred_y_fixed_250m.shape[1]
drr2 = (all_pred_y_fixed_500m == 0).sum(axis=1) / all_pred_y_fixed_500m.shape[1]
drr3 = (all_pred_y_fixed_750m == 0).sum(axis=1) / all_pred_y_fixed_750m.shape[1]




plt.boxplot({1:crr1, 2:crr2, 3:crr3}.values(), showfliers=True)


plt.boxplot({1:drr1, 2:drr2, 3:drr3}.values(), showfliers=True)

plt.boxplot({1:crr1, 2:drr1, 3:crr2, 4:drr2, 5:crr3, 6:drr3}.values(), showfliers=True)

#%%
from matplotlib.patches import Patch
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
#이 코드는 색상에 따른 범례를 추가하고 있으며, Patch 객체를 사용하여 범례의 색상과 레이블을 지정합니다. 이후에 plt.legend() 함수를 사용하여 범례를 그래프에 추가합니다.




#%%


real1 = all_real_y_fixed_250m[0,:]
pred1 = all_pred_y_fixed_250m[0,:]

real2 = all_real_y_fixed_750m[50,:]
pred2 = all_pred_y_fixed_750m[50,:]


np.intersect1d(np.where(real1==0)[0],np.where(pred1==0)[0]).shape
np.intersect1d(np.where(pred1==0)[0],np.where(real1==0)[0]).shape


def count_real_0_as_0(real_1d, pred_1d):
    return np.intersect1d(np.where(real_1d==0)[0],np.where(pred_1d==0)[0]).shape[0]


frr1 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed_750m, all_pred_y_fixed_750m)])
frr2 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed_500m, all_pred_y_fixed_500m)])
frr3 = np.array([count_real_0_as_0(xx,yy) / np.where(xx==0)[0].shape[0] for xx,yy in zip(all_real_y_fixed_250m, all_pred_y_fixed_250m)])

plt.boxplot({1:frr3, 2:frr2, 3:frr1}.values(), showfliers=True)


#%%


drr1 = np.array([count_real_0_as_0(xx,yy) for xx,yy in zip(all_real_y_fixed_750m, all_pred_y_fixed_750m)]) / all_real_y_fixed_750m.shape[1]

drr2 = np.array([count_real_0_as_0(xx,yy) for xx,yy in zip(all_real_y_fixed_500m, all_pred_y_fixed_500m)]) / all_real_y_fixed_500m.shape[1]

drr3 = np.array([count_real_0_as_0(xx,yy) for xx,yy in zip(all_real_y_fixed_250m, all_pred_y_fixed_250m)]) / all_real_y_fixed_250m.shape[1]


plt.boxplot({1:drr3, 2:drr2, 3:drr1}.values(), showfliers=True)


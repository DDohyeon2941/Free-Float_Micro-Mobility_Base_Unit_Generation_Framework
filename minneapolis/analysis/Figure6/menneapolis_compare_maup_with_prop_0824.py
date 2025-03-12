# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:27:22 2023

@author: dohyeon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import user_utils as uu

save_date = '0615'
save_date1 = '0530'
save_date2 = '0809'


temp_pkl1 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_750m_%s.pickle'%(save_date))['train_dataset']
temp_pkl2 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_%s.pickle'%(save_date1))['train_dataset']
temp_pkl3 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_250m_%s.pickle'%(save_date))['train_dataset']
temp_pkl4 = uu.load_gpickle(r'minneapolis_prop_train_test_scaler_dataset_%s.pickle'%(save_date2))['train_dataset']




arr1 = np.exp(temp_pkl1.demand)-1
arr2 = np.exp(temp_pkl2.demand)-1
arr3 = np.exp(temp_pkl3.demand)-1
arr4 = np.exp(temp_pkl4.demand)-1

#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse

bw_val =  2

sns.kdeplot(arr3[arr3 >= 1], label='250m',bw_adjust=bw_val)
sns.kdeplot(arr2[arr2 >= 1], label='500m',bw_adjust=bw_val)
sns.kdeplot(arr1[arr1 >= 1], label='750m',bw_adjust=bw_val)
sns.kdeplot(arr4[arr4 >= 1], label='Prop',bw_adjust=bw_val)
plt.xlabel('Demand')

plt.xlim(0,50)
plt.legend()

# 서브그래프 생성
ax = plt.gca()  # 현재의 Axes 객체를 가져옴
axins = inset_axes(ax, width="30%", height="30%", 
                   bbox_to_anchor=(0.6, 0.2, 1, 1),
                   bbox_transform=ax.transAxes, loc=3)  # 서브그래프의 크기와 위치를 설정

# 서브그래프에 동일한 데이터로 그래프 그리기
sns.kdeplot(arr3[arr3 >= 1], label='250m', ax=axins,bw_adjust=bw_val)
sns.kdeplot(arr2[arr2 >= 1], label='500m', ax=axins,bw_adjust=bw_val)
sns.kdeplot(arr1[arr1 >= 1], label='750m', ax=axins,bw_adjust=bw_val)
sns.kdeplot(arr4[arr4 >= 1], label='Prop', ax=axins,bw_adjust=bw_val)

# 서브그래프에 대한 축 범위 설정
axins.set_xlim(5, 30)
axins.set_ylim(0, 0.01)
#plt.legend()

ellipse = Ellipse(xy=(17, 0.020), width=30, height=0.04, edgecolor='r', fc='None', linestyle='--')
ax.add_patch(ellipse)

##
ax.annotate("", xy=(0.6, 0.2), xycoords='axes fraction',
            xytext=(17, 0.020), textcoords='data',
            arrowprops=dict(arrowstyle="->", lw=1.5))

plt.xlabel('Demand')
plt.ylabel('Density')
plt.show()

#%%

b1 = (arr3==0).sum() / arr3.shape[0]
b2 = (arr2==0).sum() / arr2.shape[0]
b3 = (arr1==0).sum() / arr1.shape[0]
b4 = (arr4==0).sum() / arr4.shape[0]


(b1+b2+b3+b4) / 4











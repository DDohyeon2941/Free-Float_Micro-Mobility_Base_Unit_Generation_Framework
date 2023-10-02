# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:34:23 2023

@author: dohyeon
"""

"""

prop 방식의 base grid와 fixed 방식의 base grid 인덱스를 맞춰봄


"""


import user_utils as uu
import numpy as np
obj1 = uu.load_gpickle(r'kansas_fixed_500m_262g_0507.pickle')
obj2 = uu.load_gpickle(r'kansas_prop_45c_237f_0809.pickle')


#%% prop base unit을 대상으로, 같은 base grid를 포함하는 fixed grid 인덱스 정보 산출


obj2_mask_dic = {}

for uni_key, uni_vals in obj2.items() :
    obj2_mask_dic[uni_key] = []
    for uni_val in uni_vals:
        for uni_key1, uni_vals1 in obj1.items():
            if uni_val in uni_vals1:
                obj2_mask_dic[uni_key].append(uni_key1)
                break


obj2_mask_dic
#%% fixed grid로 산출한 base unit을 대상으로, 같은 base grid를 포함하는 prop방식의 base unit인덱스 정보 산출

obj1_mask_dic = {}

for uni_key, uni_vals in obj1.items() :
    obj1_mask_dic[uni_key] = []
    for uni_val in uni_vals:
        for uni_key1, uni_vals1 in obj2.items():
            if uni_val in uni_vals1:
                obj1_mask_dic[uni_key].append(uni_key1)
                break

obj1_mask_dic
#%%

"""overlapped한 영역을 서칭함"""

"""prop 방식의 base unit을 대상으로 단일한 fixed grid를 구성하는 base grid들과 pair인 경우를 산출"""

tt_li = []
for aa,bb in obj2_mask_dic.items():
    if (len(bb) == 4) & (len(set(bb)) == 1):
        tt_li.append(aa)
"""fixed 방식의 base unit을 대상으로 단일한 prop grid를 구성하는 base grid들과 pair인 경우를 산출"""
tt_li1 = []
for aa,bb in obj1_mask_dic.items():
    if (len(bb) == 4) & (len(set(bb)) == 1):
        tt_li1.append(np.unique(bb))
"""위 두 정보중 서로 겹치는 인덱스 정보만을 추출
    이렇게 하는 이유는, base unit 중 4개 이상의 base grid를 포함하는 경우도 있기 때문에 prop와 fixed 관점 모두 고려"""

prop_arr  = np.intersect1d(tt_li, tt_li1)

#%%
"""prop 방식의 base unit을 대상으로 단일한 fixed grid를 구성하는 base grid들과 pair인 경우를 산출"""
tt_li2 = []
for aa,bb in obj2_mask_dic.items():
    if (len(bb) == 4) & (len(set(bb)) == 1):
        tt_li2.append(np.unique(bb))

"""fixed 방식의 base unit을 대상으로 단일한 prop grid를 구성하는 base grid들과 pair인 경우를 산출"""
tt_li3 = []
for aa,bb in obj1_mask_dic.items():
    if (len(bb) == 4) & (len(set(bb)) == 1):
        tt_li3.append(aa)

fixed_arr = np.intersect1d(tt_li3, np.array(tt_li2).squeeze())

#%%

"""prop base unit의 20번과, fixed grid 84번은 서로 matching 됨"""


prop_arr  = np.intersect1d(tt_li, tt_li1)
prop_arr = prop_arr[np.where(prop_arr>=45)[0]]
fixed_arr = fixed_arr[(fixed_arr!=84)&(fixed_arr!=131)]


uu.save_gpickle(r'kansas_overlapped_index_fixed_prop_0809.pickle',{'fixed_grid_index' : fixed_arr, 'prop_grid_index':prop_arr})














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
obj1 = uu.load_gpickle(r'minneapolis_500m_366g_0809.pickle')
#obj2 = uu.load_gpickle(r'minneapolis_prop_55c_351f_0809.pickle')
obj2 = uu.load_gpickle(r'minneapolis_prop_45c_357f_0809.pickle')


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
"""overlap 되는 경우를 서칭해야함"""

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

whole_fixed_idx = np.unique(sum(obj2_mask_dic.values(),[]))
whole_prop_idx = np.unique(sum(obj1_mask_dic.values(),[]))



uu.save_gpickle(r'minneapolis_fixed_prop_index_0809.pickle',
{'bigger_whole_idx':whole_fixed_idx,
 'bigger_sep_idx':np.setdiff1d(whole_fixed_idx, fixed_arr),
 'bigger_ovl_idx':fixed_arr, 'prop_whole_idx':whole_prop_idx,
 'prop_sep_idx':np.setdiff1d(whole_prop_idx, prop_arr),
 'prop_ovl_idx':prop_arr})














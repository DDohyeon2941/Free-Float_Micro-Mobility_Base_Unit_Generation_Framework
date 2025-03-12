# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:31:32 2023

@author: dohyeon
"""

from  kansas_new_main_indi_grids_fixed_train_vali_123_500m_0812 import LSTM1

import pandas as pd
import numpy as np

import user_utils as uu


def merge_loss_valis(loss_dic_info:list, val_idx=0, val_num=5):
    new_dic_list1 = []
    for xx in range(val_num):
        new_dict = {}
        for ukey, uval in loss_dic_info[xx].items():
            if val_idx ==-1:
                new_dict[(ukey[0], ukey[1], ukey[2], xx)] = uval[val_idx]
            else:
                new_dict[(ukey[0], ukey[1], ukey[2], xx)] = uval[val_idx][-1]
        new_dic_list1.append(new_dict)
    
    new_new_dict= {}
    for uni_dict in new_dic_list1:
        for uni_key, uni_val in uni_dict.items():
            if not uni_key in new_new_dict.keys():
                new_new_dict[uni_key] = [uni_val, ]
            else:
                new_new_dict[uni_key].append(uni_val)

    score_df = pd.DataFrame(data=new_new_dict)

    return score_df


def print_best_params(hyper_params, hyper_epoch):

    best_params = {'lr':hyper_params[0], 'hidden': hyper_params[1], 'batch': hyper_params[2], 'epoch':hyper_epoch}
    print(best_params)

#%%
if __name__ == '__main__':
    save_date = '0812'
    try_num = 5
    val_num = 4
    merged_loss_li = []
    merged_epoch_li = []
    for subset_name in range(try_num):

        loss_dic_list = [
            uu.load_gpickle(r'kansas_fixed_loss_history_vali_500m_%s_%s_%s.pickle'%(
                subset_name, xx, save_date)) for xx in range(1, val_num+1)
            ]
        merged_df = merge_loss_valis(loss_dic_list, 1, val_num)
        merged_epoch_df = merge_loss_valis(loss_dic_list, -1, val_num)

        merged_loss_li.append(merged_df)
        merged_epoch_li.append(merged_epoch_df)

    temp_hyper_tuple = pd.concat(merged_loss_li).reset_index(drop=True).T.mean(axis=1).unstack().mean(axis=1).idxmin()
    temp_epoch = round(pd.concat(merged_epoch_li).reset_index(drop=True).T.mean(axis=1).unstack().mean(axis=1)[temp_hyper_tuple])
    print_best_params(temp_hyper_tuple, temp_epoch)



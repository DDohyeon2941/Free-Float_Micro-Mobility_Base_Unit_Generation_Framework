# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 15:31:32 2023

@author: dohyeon
"""

from  kansas_main_indi_grids_fixed_500m_train_vali_12345_750m_0824 import LSTM1

import pandas as pd
import numpy as np

import user_utils as uu


def merge_loss_valis(loss_dic_info:list, val_idx=0, val_num=5):
    new_dic_list1 = []
    for xx in range(val_num):
        new_dict = {}
        for ukey, uval in loss_dic_info[xx].items():
            if val_idx ==-1:
                new_dict[(ukey[0], ukey[1], ukey[2])] = uval[val_idx]
            else:
                new_dict[(ukey[0], ukey[1], ukey[2])] = uval[val_idx][-1]
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


def merge_loss_valis1(loss_dic_info:list, val_idx=0, val_num=5):
    new_dic_list1 = []
    for xx in range(val_num):
        new_dict = {}
        for ukey, uval in loss_dic_info[xx].items():
            if val_idx == -1:
                new_dict[(ukey[0], ukey[1], ukey[2])] = uval[val_idx]
            else:
                new_dict[(ukey[0], ukey[1], ukey[2])] = uval[val_idx][-1]
        new_dic_list1.append(new_dict)
    
    new_new_dict= {}
    for uni_dict in new_dic_list1:
        for uni_key, uni_val in uni_dict.items():
            if not uni_key in new_new_dict.keys():
                new_new_dict[uni_key] = [uni_val, ]
            else:
                new_new_dict[uni_key].append(uni_val)


    return new_new_dict

def find_hyperparams(score_info):
    hyper1 = score_info.min(axis=0).idxmin()
    hyper2 = score_info.idxmin(axis=0).loc[hyper1]
    return hyper2, hyper1

def get_best_params(score_info, epoch_info):
    hyper2, hyper1 = find_hyperparams(score_info)

    hyper3 = int(epoch_info.loc[hyper2, hyper1])


    best_params = {'lr':hyper2[0], 'hidden': hyper2[1], 'epoch':hyper3}
    return best_params

#best_params = get_best_params(new_score_df, new_epoch_df)


def best_params_main(loss_dic_info, val_idx, val_num):

    score_df = merge_loss_valis(loss_dic_info, val_idx=val_idx, val_num=val_num)
    epoch_df = merge_loss_valis(loss_dic_info, val_idx=-1, val_num=val_num)
    new_score_df = (score_df.T).mean(axis=1).unstack()
    new_epoch_df = (epoch_df.T).mean(axis=1).unstack()
    best_params = get_best_params(new_score_df, new_epoch_df)
    return best_params
#%%
if __name__ == '__main__':
    save_date = '0824'

    loss_dic_list = [uu.load_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(xx, save_date)) for xx in [1,2,3,4,5]]

    fixed_best_params = best_params_main(loss_dic_list,1,5)
    print(fixed_best_params)
    """ 
    trn: {'lr': 0.0009, 'hidden': 24, 'epoch': 21}
    vali: {'lr': 0.0001, 'hidden': 24, 'epoch': 41}"""

    #%%
    gg1= merge_loss_valis(loss_dic_list, val_num=5)
    gg2= merge_loss_valis1(loss_dic_list, val_num=5)


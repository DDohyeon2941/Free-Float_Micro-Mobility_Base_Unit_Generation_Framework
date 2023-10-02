# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:09:29 2022

@author: dohyeon
"""

import torch
import torch.nn as nn
import itertools
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
import  multiprocessing
from torch.utils.data import Dataset, DataLoader
import gc
#import ipdb

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

import user_utils as uu

from datetime import datetime

def combine_date_time(date_obj, hour):
    date_str = date_obj.strftime('%Y-%m-%d')
    return datetime.strptime(date_str + f' {hour}:00:00', '%Y-%m-%d %H:%M:%S')

argparser = argparse.ArgumentParser()
argparser.add_argument('--batch_size', type=int, default=6)
argparser.add_argument('--seq_length', type=int, default=6)
argparser.add_argument('--hidden_size', type=int, default=6)
argparser.add_argument('--step_num', type=int, default=1)
argparser.add_argument('--epoch_num', type=int, default=50)
argparser.add_argument('--n_layers', type=int, default=1)
argparser.add_argument('--lr', type=float, default=0.0001)
argparser.add_argument('--dropout', type=float, default=0)
argparser.add_argument('--weight_decay', type=float, default=0)
args = argparser.parse_args()
class SharingBikes_test(Dataset):


    gr_var = 'level_2'
    subset_var = ['mask_wq','scaled_y']
    target_var = 'scaled_y'

    def __init__(self, temp_2d,
                 seq_len=21,
                 step_size=10,
                 batch_size=128,
                 feature_len=28):
        
        self.temp_2d = temp_2d
        self.seq_len=seq_len
        self.step_size=step_size
        self.batch_size = batch_size
        self.feature_len = feature_len

        bin_list = []
        bin_num_list = []
        bin_index_list = []
        bin_ii = 0
        for _, uni_df in self.temp_2d.groupby(self.gr_var):
            uni_df = uni_df.reset_index(drop=True).set_index(['date','hour'])
            uni_df = uni_df[self.subset_var]
            #ipdb.set_trace()
            uni_df1 = uni_df
            iter_len = ((uni_df1.shape[0]-self.seq_len)//self.step_size)
            ttt_list = []
            tttt_list = []
            ttttt_list = []
            for uni_idx in range(iter_len):
                uni_idx *= self.step_size
                inputs=uni_df1.iloc[uni_idx: uni_idx+self.seq_len,:]
                target=uni_df1.iloc[uni_idx + self.seq_len,-1]
                #print((uni_df1.index[uni_idx + self.seq_len]))

                target_date =  combine_date_time(*(uni_df1.index[uni_idx + self.seq_len]))

                ttt_list.append([inputs, target])
                tttt_list.append(bin_ii)
                ttttt_list.append(target_date)

            bin_list+=ttt_list[:((iter_len//self.batch_size)*self.batch_size)]
            bin_num_list+=tttt_list[:((iter_len//self.batch_size)*self.batch_size)]
            bin_index_list+=ttttt_list[:((iter_len//self.batch_size)*self.batch_size)]

            bin_ii += 1
        self.bin_list = np.array(bin_list, dtype=object)
        self.bin_num_list = np.array(bin_num_list)
        self.bin_index_list = bin_index_list

    def __len__(self):

        return (len(self.bin_list)//self.batch_size)

    def __getitem__(self, idx):

        idx *= self.batch_size
        #idx *= self.step_size

        #print(idx)
        inputs=torch.FloatTensor(np.array([xx[0] for xx in self.bin_list[idx: idx+ self.batch_size]]))
        target=torch.FloatTensor(np.array([xx[1] for xx in self.bin_list[idx: idx+ self.batch_size]]))
        grid_index = torch.FloatTensor(np.array([xx for xx in self.bin_num_list[idx: idx+ self.batch_size]]))
        date_index = [str(xx) for xx in self.bin_index_list[idx: idx+ self.batch_size]]
        #date_index = torch.FloatTensor(np.array([xx for xx in self.bin_index_list[idx: idx+ self.batch_size]]))
        #date_index = [xx for xx in self.bin_index_list[idx: idx+ self.batch_size]]
        # write your codes here

        return grid_index, inputs, target, date_index


#%%


save_date = '0530'



temp_pkl1 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_%s.pickle'%(save_date))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']

#%%
save_date = '0615'



temp_pkl1 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_250m_%s.pickle'%(save_date))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']

#%%


save_date = '0615'



temp_pkl1 = uu.load_gpickle(r'minneapolis_fixed_train_test_scaler_dataset_750m_%s.pickle'%(save_date))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']
#%%




save_date = '0530'


temp_pkl1 = uu.load_gpickle(r'minneapolis_prop_train_test_scaler_dataset_%s.pickle'%(save_date))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']



#%%

BIKE1 = SharingBikes_test(test_df, seq_len=   8, step_size=1, batch_size=8)
Validation_loader=DataLoader(BIKE1, batch_size=1,shuffle=False,drop_last=True)
#%%
ii=0

date_list1 = []
for _,_,_,data in Validation_loader:
    date_list1.append(data)
    if ii==91: break
    ii+=1
date_list1
#%%

date_list2 = sum(date_list1,[])

pd.DataFrame(date_list2).to_csv(r'minneapolis_test_target_date_0820.csv',index=False)

#%%









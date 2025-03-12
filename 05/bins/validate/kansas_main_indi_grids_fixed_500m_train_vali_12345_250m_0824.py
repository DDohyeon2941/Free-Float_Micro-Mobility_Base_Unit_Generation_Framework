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

from kansas_main_indi_grids_fixed_500m_train_vali_12345_750m_0824 import SharingBikes, LSTM1, train, validate, main, get_smape

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

#%%
if __name__ == '__main__':

    save_date = '0615'
    save_date1 = '0824'

    lr_range = np.round(np.arange(0.0001, 0.001, 0.0002),4)
    hidden_range = [8, 16, 24]
    seq_length = 8
    batch_size = 8

    #%%

    vali_num = 1
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_%s.pickle'%(save_date))[vali_num]
    train_df, vali_df = temp_pkl['train_dataset'], temp_pkl['valid_dataset']

    loss_dic = {}

    exp_round = 0
    start_time = time.time()
    for uni_lr, uni_hidden in itertools.product(lr_range, hidden_range):

        aa1=main(train_df, vali_df, uni_lr, uni_hidden, seq_length, batch_size)
        loss_dic[(uni_lr, uni_hidden, batch_size)] = aa1
        print('%s / %s' % (exp_round, len(lr_range)*len(hidden_range)))
        exp_round+=1
        aa1=None

    print(uu.get_datetime())
    print(time.time() - start_time)

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_250m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)




    #%%
    vali_num = 2
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_%s.pickle'%(save_date))[vali_num]
    train_df, vali_df = temp_pkl['train_dataset'], temp_pkl['valid_dataset']

    loss_dic = {}

    exp_round = 0
    start_time = time.time()
    for uni_lr, uni_hidden in itertools.product(lr_range, hidden_range):

        aa1=main(train_df, vali_df, uni_lr, uni_hidden, seq_length, batch_size)
        loss_dic[(uni_lr, uni_hidden, batch_size)] = aa1
        print('%s / %s' % (exp_round, len(lr_range)*len(hidden_range)))
        exp_round+=1
        aa1=None

    print(uu.get_datetime())
    print(time.time() - start_time)

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_250m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)



    #%%
    vali_num = 3
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_%s.pickle'%(save_date))[vali_num]
    train_df, vali_df = temp_pkl['train_dataset'], temp_pkl['valid_dataset']

    loss_dic = {}

    exp_round = 0
    start_time = time.time()
    for uni_lr, uni_hidden in itertools.product(lr_range, hidden_range):

        aa1=main(train_df, vali_df, uni_lr, uni_hidden, seq_length, batch_size)
        loss_dic[(uni_lr, uni_hidden, batch_size)] = aa1
        print('%s / %s' % (exp_round, len(lr_range)*len(hidden_range)))
        exp_round+=1
        aa1=None

    print(uu.get_datetime())
    print(time.time() - start_time)

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_250m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)



    #%%
    vali_num = 4
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_%s.pickle'%(save_date))[vali_num]
    train_df, vali_df = temp_pkl['train_dataset'], temp_pkl['valid_dataset']

    loss_dic = {}

    exp_round = 0
    start_time = time.time()
    for uni_lr, uni_hidden in itertools.product(lr_range, hidden_range):

        aa1=main(train_df, vali_df, uni_lr, uni_hidden, seq_length, batch_size)
        loss_dic[(uni_lr, uni_hidden, batch_size)] = aa1
        print('%s / %s' % (exp_round, len(lr_range)*len(hidden_range)))
        exp_round+=1
        aa1=None

    print(uu.get_datetime())
    print(time.time() - start_time)

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_250m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)




#%%
    vali_num = 5
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_250m_%s.pickle'%(save_date))[vali_num]
    train_df, vali_df = temp_pkl['train_dataset'], temp_pkl['valid_dataset']

    loss_dic = {}

    exp_round = 0
    start_time = time.time()
    for uni_lr, uni_hidden in itertools.product(lr_range, hidden_range):

        aa1=main(train_df, vali_df, uni_lr, uni_hidden, seq_length, batch_size)
        loss_dic[(uni_lr, uni_hidden, batch_size)] = aa1
        print('%s / %s' % (exp_round, len(lr_range)*len(hidden_range)))
        exp_round+=1
        aa1=None

    print(uu.get_datetime())
    print(time.time() - start_time)

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_250m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)

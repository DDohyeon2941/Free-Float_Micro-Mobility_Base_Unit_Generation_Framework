# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:09:29 2022

@author: dohyeon
"""

import torch
import torch.nn as nn

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
#from main_indi_grids_fixed_500m_train_vali_12345_0419 import SharingBikes, LSTM1, train, validate, get_smape

from kansas_main_indi_grids_fixed_500m_train_vali_12345_0824 import SharingBikes, get_smape
from kansas_main_indi_grids_fixed_500m_test_0820 import main1

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


save_date = '0809'



temp_pkl1 = uu.load_gpickle(r'kansas_prop_train_test_scaler_dataset_%s.pickle'%(save_date))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']

#%%
if __name__ == '__main__':

    save_date1 = '0824'
    best_params = {'lr':0.0003, 'hidden': 16, 'seq_length':8, 'batch':8, 'epoch':37}


    aa2 = main1(lr=best_params['lr'],
                hidden_size=best_params['hidden'],
                seq_length=best_params['seq_length'],
                batch_size=best_params['batch'],
                epoch_num=best_params['epoch'])

    uu.save_gpickle(r'kansas_prop_model_loss_test_%s.pickle'%(save_date1), aa2)

    #%%

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mmodel = aa2[-1].to(device)
    hidden = mmodel.init_hidden(mmodel.batch_size)
    BIKE1 = SharingBikes(test_df, seq_len=    mmodel.input_size, step_size=args.step_num, batch_size=mmodel.batch_size)
    Validation_loader=DataLoader(BIKE1, batch_size=1,shuffle=False,drop_last=True)
    
    with torch.no_grad():
        pred = []
        realy = []
        grid_num_li = []
        grid_y_li = []
        grid_pred_li = []
        mmodel.eval()
        for data in Validation_loader:
            #seq, target= data
            yy = data[2]
            realy+=yy.squeeze().detach().numpy().tolist()
            #grid_num_li += data[0].squeeze().detach().numpy().tolist()
            grid_num_li.append(data[0][0].detach().numpy().tolist())
            grid_y_li.append(yy.squeeze().detach().numpy().tolist())
            data= data[1][0].to(torch.float32)
        
            data = data.to(device)
            out,hidden = mmodel(data,hidden)
            pred+=torch.flatten(out).cpu().detach().numpy().tolist()
            grid_pred_li.append(torch.flatten(out).cpu().detach().numpy().tolist())
    #%%
    y1 = np.array(realy)
    pred = np.array(pred)
    pred[np.where(np.array(pred)<0)[0]]= float(0)
    
    new_y1 = scaler.inverse_transform(y1.reshape(-1,1)).squeeze()
    new_pred = scaler.inverse_transform(pred.reshape(-1,1)).squeeze()
    
    final_y1 = np.exp(new_y1)-1
    final_pred = np.exp(new_pred)-1
    
    grid_final_y1 = final_y1.reshape(-1, mmodel.batch_size)
    grid_final_pred = final_pred.reshape(-1, mmodel.batch_size)

    #%%
    s_mse = mse(final_y1, final_pred)**0.5
    s_mae = mae(final_y1, final_pred)
    s_smape = get_smape(final_y1, final_pred)
    s_mape = mape(final_y1[np.where(final_y1>0)[0]],  final_pred[np.where(final_y1>0)[0]])
    

    #%%
    sstep_num = int((np.array(grid_num_li).shape[0]/test_df['level_2'].nunique()))
    yy_li = []
    pp_li = []
    sstep_li = []
    for uni_g_idx in range(len(grid_final_y1)//sstep_num):
        uni_g_idx *= sstep_num
        aa11=np.concatenate(np.array(grid_final_y1[uni_g_idx:uni_g_idx+sstep_num]))
        aa21=np.concatenate(np.array(grid_final_pred[uni_g_idx:uni_g_idx+sstep_num]))
        aa21[aa21<0] = float(0)
        yy_li.append(aa11)
        pp_li.append(aa21)
        if np.sum(aa11>0) > 0:
            sstep_li.append([mse(aa11,aa21)**0.5, mae(aa11,aa21), mape(aa11[np.where(aa11>0)[0]],  aa21[np.where(aa11>0)[0]]), get_smape(aa11,aa21)])
        else:
            sstep_li.append([mse(aa11,aa21)**0.5, mae(aa11,aa21), np.nan, get_smape(aa11,aa21)])

    uu.save_gpickle(r'kansas_real_pred_y_prop_%s.pickle'%(save_date1),{'pred_y':np.array(pp_li),'real_y':np.array(yy_li)})

    #%%

    
    score_df = pd.DataFrame(data=[s_mse, s_mae, s_mape, s_smape]).T
    score_df.columns = ['RMSE','MAE','MAPE','SMAPE']
    score_df.loc[1,:] = np.nanmean(np.array(sstep_li),axis=0)
    score_df.index = ['all','indi']
    #score_df.to_csv(r'score_prop_0215.csv',index=False)

    o_dict = dict(zip(np.argsort(np.mean(np.array(yy_li),axis=1)),np.arange(np.array(yy_li).shape[0])))

    o_score_df = pd.DataFrame(data=    np.array(sstep_li))
    o_score_df.columns = ['RMSE','MAE','MAPE','SMAPE']
    o_score_df.loc[:,'rank'] =     np.array([o_dict[xx] for xx in np.arange(np.array(yy_li).shape[0])])
    o_score_df.loc[:, 'avg_real'] = np.mean(np.array(yy_li), axis=1)
    o_score_df.loc[:, 'avg_pred'] = np.mean(np.array(pp_li), axis=1)


    o_score_df.to_pickle(r'kansas_o_score_df_prop_%s.pkl'%(save_date1))

    #%%

    fig1, axes1 = plt.subplots(2,1, sharex=True)
    #ax2 = axes1[0].twinx()
    #ax2.plot(o_score_df.sort_values(by='rank')['avg_real'].values, label='real', c='r')
    #ax2.plot(o_score_df.sort_values(by='rank')['avg_pred'].values, label='pred', c='g')

    axes1[0].plot(o_score_df.sort_values(by='rank')['RMSE'].values, label='RMSE')
    axes1[0].plot(o_score_df.sort_values(by='rank')['MAE'].values, label='MAE')
    axes1[1].plot(o_score_df.sort_values(by='rank')['SMAPE'].values, label='SMAPE')
    axes1[1].plot(o_score_df.sort_values(by='rank')['MAPE'].fillna(0).values, label='MAPE')

    #ax3 = axes1[1].twinx()
    #ax3.plot(o_score_df.sort_values(by='rank')['avg_real'].values, label='real', c='r')
    #ax3.plot(o_score_df.sort_values(by='rank')['avg_pred'].values, label='pred', c='g')

    axes1[0].legend()
    axes1[1].legend()
    #ax2.legend(loc='lower left')
    #ax3.legend(loc='lower left')
    axes1[1].set_xlabel('Sorted Grid Order')
    axes1[0].set_title("Prop Base Unit")

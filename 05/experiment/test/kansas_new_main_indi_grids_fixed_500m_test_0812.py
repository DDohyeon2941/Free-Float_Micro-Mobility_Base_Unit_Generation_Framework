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
from torch.utils.data import DataLoader
import gc
#import ipdb

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae

import user_utils as uu
from kansas_new_main_indi_grids_fixed_train_vali_123_500m_0812 import SharingBikes, LSTM1, train, validate


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


save_date = '0812'
save_date1 = '0810'



temp_pkl1 = uu.load_gpickle(r'kansas_fixed_train_test_scaler_dataset_%s.pickle'%(save_date1))
scaler, new_train_df, test_df = temp_pkl1['scaler'],  temp_pkl1['train_dataset'],  temp_pkl1['test_dataset']

#hyper_params_dic = uu.load_gpickle(r'kansas_sub12345_prop_vali_hyper_params_%s.pickle'%(save_date))[subset_name]

def main1(lr, hidden_size, seq_length, batch_size, epoch_num):

    #set arguments
    ##
    
    #n_cpu= multiprocessing.cpu_count()
    #device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    n_cpu= multiprocessing.cpu_count()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    BIKE = SharingBikes(new_train_df,
                        seq_len=seq_length, step_size=args.step_num, batch_size=batch_size)
    Train_loader=DataLoader(BIKE, batch_size=1,shuffle=True,drop_last=True)

    BIKE1 = SharingBikes(test_df, seq_len=seq_length, step_size=args.step_num, batch_size=batch_size)
    Validation_loader=DataLoader(BIKE1, batch_size=1,shuffle=True,drop_last=True)

    cost_function = nn.MSELoss()
    #cost_function = nn.L1Loss()
    #cost_function = nn.SmoothL1Loss()

    min_loss=float('inf')

    #lstm = nn.LSTM(23, 23, num_layers=args.n_layer)
    lstm = LSTM1(input_size=seq_length,  hidden_size=hidden_size, batch_size=batch_size,
                 num_layer=args.n_layers).to(device)
    optimizer = Adam(lstm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    train_loss = []
    vali_loss = []

    patience = 2
    trigger_times = 0
    last_loss = 100

    
    c_time=0

    for epoch in range(epoch_num):
        #print(epoch)
        start = time.time()  # 시작 시간 저장
        trn_loss=train(lstm, Train_loader, device, cost_function, optimizer)
        #print('here')
        test_loss=validate(lstm, Validation_loader, device, cost_function)
        
        c_time += time.time()-start
        print('model : %s ,  %s epoch : , trn_loss : %.5f,  tst_loss : %.5f, time: %.3f, c_time: %.3f'%('Charlstm',epoch+1, trn_loss, test_loss, (time.time() - start), c_time))
        
        if (epoch >=10) & ((trn_loss > last_loss) | ((last_loss - trn_loss)<0.00001 )):
            trigger_times +=1
            print('Trigger Times:', trigger_times)

            if trigger_times >= patience:
                print('Ealy stopping')
                gc.collect()
                torch.cuda.empty_cache()
                return train_loss, vali_loss, lstm
        else:

            trigger_times = 0

        last_loss = trn_loss

        train_loss.append(trn_loss)

        vali_loss.append(test_loss)
        scheduler.step()
    gc.collect()
    torch.cuda.empty_cache()
    return train_loss, vali_loss, lstm

def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum((np.abs(pred_y-real_y)) / ((real_y+pred_y)/2))/real_y.shape[0]

def find_both_non_zero_idx(real_y, pred_y):
    return np.setdiff1d(np.arange(real_y.shape[0]), np.intersect1d(np.where(real_y==0)[0], np.where(pred_y==0)[0]))

#%%
if __name__ == '__main__':




    best_params = {'lr':0.0009,
                   'hidden': 32,
                   'seq_length':16, 'batch':32,
                   'epoch':14}

    try_num = 1
    for uni_try in range(try_num, 5):
        print("%s try start"%(uni_try))
        aa2 = main1(lr=best_params['lr'],
                    hidden_size=best_params['hidden'],
                    seq_length=best_params['seq_length'],
                    batch_size=best_params['batch'],
                    epoch_num=best_params['epoch'])
    
        uu.save_gpickle(r'kansas_fixed_model_loss_test_%s_%s.pickle'%(uni_try, save_date), aa2)
    
        #%%
        #uni_try = 4
        #aa2= uu.load_gpickle(r'kansas_fixed_model_loss_test_%s_%s.pickle'%(uni_try, save_date))
    
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        mmodel = aa2[-1].to(device)
        #hidden = mmodel.init_hidden(mmodel.batch_size)
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
                hidden = mmodel.init_hidden(mmodel.batch_size)
                hidden = (hidden[0].to(device), hidden[1].to(device))

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
    
        both_non_zero_idx = find_both_non_zero_idx(final_y1, final_pred)
    
        s_smape = get_smape(final_y1[both_non_zero_idx], final_pred[both_non_zero_idx])
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
                sstep_li.append([mse(aa11,aa21)**0.5, mae(aa11,aa21), mape(aa11[np.where(aa11>0)[0]],  aa21[np.where(aa11>0)[0]]), get_smape(aa11[find_both_non_zero_idx(aa11, aa21)],aa21[find_both_non_zero_idx(aa11, aa21)])])
    
            else:
                sstep_li.append([mse(aa11,aa21)**0.5, mae(aa11,aa21), np.nan, get_smape(aa11[find_both_non_zero_idx(aa11, aa21)],aa21[find_both_non_zero_idx(aa11, aa21)])])
    
        uu.save_gpickle(r'kansas_real_pred_y_fixed_%s_%s.pickle'%(uni_try, save_date),
                        {'pred_y':np.array(pp_li),'real_y':np.array(yy_li)})
    
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
    
    
        o_score_df.to_csv(r'kansas_o_score_df_fixed_%s_%s.csv'%(uni_try, save_date), index=False)
    
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
        axes1[0].set_title("Fixed Unit")

    #%%
        print("%s try end"%(uni_try))
        aa2 = None

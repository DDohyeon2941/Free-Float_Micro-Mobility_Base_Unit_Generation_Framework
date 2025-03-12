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
class SharingBikes(Dataset):


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
        bin_ii = 0
        for _, uni_df in self.temp_2d.groupby(self.gr_var):

            uni_df = uni_df[self.subset_var].reset_index(drop=True)
            #ipdb.set_trace()
            uni_df1 = uni_df
            iter_len = ((uni_df.shape[0]-self.seq_len)//self.step_size)
            ttt_list = []
            tttt_list = []
            for uni_idx in range(iter_len):
                uni_idx*= self.step_size
                inputs=uni_df1.values[uni_idx: uni_idx+self.seq_len,:]
                target=uni_df1.values[uni_idx + self.seq_len,-1]
                ttt_list.append([inputs, target])
                tttt_list.append(bin_ii)

            bin_list+=ttt_list[:((iter_len//self.batch_size)*self.batch_size)]
            bin_num_list+=tttt_list[:((iter_len//self.batch_size)*self.batch_size)]

            bin_ii += 1
        self.bin_list = np.array(bin_list, dtype=object)
        self.bin_num_list = np.array(bin_num_list)
    def __len__(self):

        return (len(self.bin_list)//self.batch_size)

    def __getitem__(self, idx):

        idx *= self.batch_size
        #idx *= self.step_size

        #print(idx)
        inputs=torch.FloatTensor(np.array([xx[0] for xx in self.bin_list[idx: idx+ self.batch_size]]))
        target=torch.FloatTensor(np.array([xx[1] for xx in self.bin_list[idx: idx+ self.batch_size]]))
        grid_index = torch.FloatTensor(np.array([xx for xx in self.bin_num_list[idx: idx+ self.batch_size]]))
        # write your codes here

        return grid_index, inputs, target

class LSTM1(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, num_embedding=28, embedding_size=5, output_size=1, num_layer=1):
        super(LSTM1, self).__init__()
        self.batch_size = batch_size
        self.n_layer = num_layer
        self.hidden_size=hidden_size
        self.input_size = input_size
        self.embbeding = nn.Embedding(num_embedding, embedding_size)
        self.net = nn.LSTM(embedding_size+1, self.hidden_size, batch_first=True, dropout=args.dropout, num_layers=self.n_layer)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x, hidden):
        #ipdb.set_trace()

        tt1=x[:,:,1].view((self.batch_size,self.input_size,1))
        tt2=self.embbeding(x[:,:,0].type(torch.long))
        input = torch.cat([tt2,tt1], dim=2)
        output, hidden=self.net(input, hidden)
        #ipdb.set_trace()
        output = self.fc(output[:,-1,:])
        #print(output)
        #output[torch.where(output<0)[0]] = float(0)
        #print(output)

        return output, hidden

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        initial_hidden=(weight.new(self.n_layer,
                                   batch_size, self.hidden_size).zero_(),
                        weight.new(self.n_layer, batch_size, self.hidden_size).zero_())

        return initial_hidden


def train(model, trn_loader, device, criterion, optimizer):

    model.train()
    running_loss = 0.0

    #gc.collect()
    #torch.cuda.empty_cache()

    h0= model.init_hidden(model.batch_size)
    h0 = (h0[0].to(device), h0[1].to(device))

    #max_grad_norm=0.5

    for train_i,(_, inputs, targets) in enumerate(trn_loader):

        #ipdb.set_trace()
        inputs = inputs[0]
        #ipdb.set_trace()
        inputs, targets = inputs.to(device), targets.to(device)

        # 디바이스 확인
        assert h0[0].device == device and h0[1].device == device
        assert inputs.device == device
        assert targets.device == device

        # get the inputs; data is a list of [inputs, labels]
        #inputs, targets = inputs.to(device), targets.to(device, dtype=torch.int64)

        #model.zero_grad()
        model.zero_grad()
        #ipdb.set_trace()
        outputs,hidden = model(inputs,h0)
        h0 = (hidden[0].detach(), hidden[1].detach())

        #compare models outputs from inputs with targets
        loss = criterion(torch.flatten(outputs), torch.flatten(targets))
        loss.backward()
        #nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
        optimizer.step()
        #print(i)
        running_loss += loss.item()

    #i starts from 0, so +1
    trn_loss=running_loss/(train_i+1)
    return trn_loss




def validate(model, val_loader, device, criterion):


    # write your codes here
    model.eval()
    h0= model.init_hidden(model.batch_size)
    h0 = (h0[0].to(device), h0[1].to(device))


    # write your codes here
    running_loss = 0.0
    for val_i, (_, inputs, targets) in enumerate(val_loader):
        inputs = inputs[0]
        inputs, targets = inputs.to(device), targets.to(device)

        # 디바이스 확인
        assert h0[0].device == device and h0[1].device == device
        assert inputs.device == device
        assert targets.device == device

        outputs,hidden = model(inputs,h0)
        h0 = (hidden[0].detach(), hidden[1].detach())

        # compare the outputs of models from inputs with targets
        loss = criterion(torch.flatten(outputs), torch.flatten(targets))
        running_loss += loss.item()

    #i starts from 0, so +1
    val_loss=running_loss/(val_i+1)
    del val_i


    return val_loss


def main(train_df, vali_df, lr, hidden_size, seq_length, batch_size):

    #set arguments
    ##
    
    #n_cpu= multiprocessing.cpu_count()
    #device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_cpu= multiprocessing.cpu_count()
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    BIKE = SharingBikes(train_df, seq_len=seq_length, step_size=args.step_num, batch_size=batch_size)
    Train_loader=DataLoader(BIKE, batch_size=1,shuffle=False,drop_last=True)

    BIKE1 = SharingBikes(vali_df, seq_len=seq_length, step_size=args.step_num, batch_size=batch_size)
    Validation_loader=DataLoader(BIKE1, batch_size=1,shuffle=False,drop_last=True)

    cost_function = nn.MSELoss()
    #cost_function = nn.L1Loss()
    #cost_function = nn.SmoothL1Loss()


    min_loss=float('inf')

    #lstm = nn.LSTM(23, 23, num_layers=args.n_layer)
    lstm = LSTM1(input_size=seq_length,  hidden_size=hidden_size, batch_size=batch_size,
                 num_layer=args.n_layers).to(device)
    optimizer = Adam(lstm.parameters(), lr,  weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train_loss = []
    vali_loss = []

    
    patience = 2
    trigger_times = 0
    last_loss = 100

    c_time=0
    
    #ipdb.set_trace()
    for epoch in range(args.epoch_num):
        #print(epoch)
        start = time.time()  # 시작 시간 저장
        trn_loss=train(lstm, Train_loader, device, cost_function, optimizer)
        #print('here')
        current_loss=validate(lstm, Validation_loader, device, cost_function)
        
        c_time += time.time()-start
        print('model : %s ,  %s epoch : , trn_loss : %.5f,  var_loss : %.5f, time: %.3f, c_time: %.3f'%('Charlstm',epoch+1, trn_loss, current_loss, (time.time() - start), c_time))

        if (epoch >=10) & ((current_loss > last_loss) | ((last_loss - current_loss) < 0.000005)):
            trigger_times +=1
            print('Trigger Times:', trigger_times)

            gc.collect()
            torch.cuda.empty_cache()


            if trigger_times >= patience:
                print('Ealy stopping')
                gc.collect()
                torch.cuda.empty_cache()

                return train_loss, vali_loss, lstm, epoch
        else:

            trigger_times = 0

        last_loss = current_loss

        train_loss.append(trn_loss)

        vali_loss.append(current_loss)
        scheduler.step()
    gc.collect()
    torch.cuda.empty_cache()

    return train_loss, vali_loss, lstm, epoch



def get_smape(real_y, pred_y):
    """
    real_y: real values
    pred_y: predicted values

    Notice
    ----------
    Small values are added to the each elements of the fraction to avoid zero-division error

    """
    return np.sum((np.abs(pred_y-real_y)) / (((real_y+pred_y)/2)+1e-14))/real_y.shape[0]

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
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_750m_%s.pickle'%(save_date))[vali_num]
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

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)




    #%%
    vali_num = 2
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_750m_%s.pickle'%(save_date))[vali_num]
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

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)



    #%%
    vali_num = 3
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_750m_%s.pickle'%(save_date))[vali_num]
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

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)



    #%%
    vali_num = 4
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_750m_%s.pickle'%(save_date))[vali_num]
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

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)




#%%
    vali_num = 5
    temp_pkl = uu.load_gpickle(r'kansas_fixed_train_valid_12345_scaler_dataset_750m_%s.pickle'%(save_date))[vali_num]
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

    uu.save_gpickle(r'kansas_fixed_loss_history_vali_750m_%s_%s.pickle'%(vali_num, save_date1), loss_dic)



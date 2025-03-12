# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 11:38:22 2021

@author: dohyeon
"""

import os
import sys
import json
import gzip
import pickle
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    )
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.utils.validation import _num_samples
from sklearn.utils.extmath import stable_cumsum


def get_scores1(y_real, y_pred):
    rmse=mean_squared_error(y_real, y_pred)**0.5
    mae=mean_absolute_error(y_real, y_pred)
    mape=mean_absolute_percentage_error(y_real, y_pred)
    return np.around([rmse, mae, mape],4)


def get_scores2(y_real, y_pred):
    mape=mean_absolute_percentage_error(y_real, y_pred)
    rmspe=np.sqrt(np.sum(np.power((y_pred-y_real)/y_real,2))/y_pred.shape[0])
    return np.around([rmspe,mape],4)


def get_scores(y_real, y_pred):
    rmse=mean_squared_error(y_real, y_pred)**0.5
    mae=mean_absolute_error(y_real, y_pred)
    return np.around([rmse, mae],4)

def predict_node_LR(i_est, LR_dict, X, ishow=False):
    aa,bb = X.shape
    i_result = np.zeros(aa)
    uni_labels_1d = i_est.apply(X)
    uni_node_labels = np.unique(uni_labels_1d)
    assert np.all(np.isin(uni_node_labels, np.array(list(LR_dict.keys()))))
    pii = 0
    for uni_node_label in uni_node_labels:
        un_idx = np.where(uni_labels_1d == uni_node_label)[0]
        uni_lr, uni_col_idx = LR_dict[uni_node_label]
        un_result = uni_lr.predict(X[un_idx][:,np.arange(bb)[:-9][uni_col_idx]])
        i_result[un_idx] = un_result
        #print(pii)
        pii+=1

    return i_result


def get_mean_predict(X, f_reg, limit, get_all=False):
    # Evaluate predictions of all estimators
    predictions = np.array([
        predict_node_LR(i_est=est, LR_dict=lrd, X=X) for est, lrd in
                            zip(f_reg.estimators_[:limit], f_reg.lrs_[:limit])]).T
    if not get_all:
        output = np.mean(predictions, axis=1)
    else:
        output = predictions
    return output

def get_median_predict(X, f_reg, limit, get_all=False):
    # Evaluate predictions of all estimators
    predictions = np.array([predict_node_LR(i_est=est, LR_dict=lrd, X=X) for
                            est, lrd in zip(f_reg.estimators_[:limit], f_reg.lrs_[:limit])]).T

    if not get_all:
        # Sort the predictions
        sorted_idx = np.argsort(predictions, axis=1)
    
        # Find index of median prediction for each sample
        weight_cdf = stable_cumsum(f_reg.estimator_weights_[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
    
        median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]
        # Return median predictions
        return predictions[np.arange(_num_samples(X)), median_estimators]
    else:
        return predictions


def section1():
    print("data_handling")
    return None


def split_x_y(data):
    """Split the dataset into independent variables and dependent variable."""
    if isinstance(data, pd.core.frame.DataFrame):
        x_arr = data[data.columns[:-1]].values
        y_arr = data[data.columns[-1]].values
    elif isinstance(data, np.ndarray):
        x_arr = data[:, :-1]
        y_arr = data[:, -1]
    return x_arr, y_arr


def read_data(data_path):
    """Load csv dataset, then split into x_arr, y_arr."""
    temp_data = pd.read_csv(data_path)
    temp_xarr, temp_yarr = split_x_y(temp_data)
    return temp_xarr, temp_yarr

def get_cv_obj(is_stratified=True, n_splits=5, random_state=0, shuffle=True):
    kwargs = dict(
        zip(
            ['n_splits', 'random_state', 'shuffle'],
            [n_splits, random_state, shuffle]
            )
        )
    return StratifiedKFold(**kwargs) if is_stratified else KFold(**kwargs)


def load_gpickle(gpickle_path):
    with gzip.open(gpickle_path, 'rb') as fff:
        gp_obj=pickle.load(fff)
    return gp_obj


def save_gpickle(gpickle_path, gpickle_obj):
    with gzip.open(gpickle_path, 'wb') as fff:
        pickle.dump(gpickle_obj, fff)

def section2():
    print("evaluate")
    return None


def get_metric(y_pred, y_proba, y_real):
    """Return performance scores for binary classification.

    Parameters
    ----------
    y_pred : 1d array
        The predicted class of test dataset
    y_proba : 1d array
        The probability of being minority class of test dataset
    y_real : 1d array
        The real class of test dataset

    Notes
    -----
    The type of performance scores
        Accuracy, F1, Precision, Recall, Specificity,
        Roc_Auc_score(Area Under the Receiver Operating Characteristic Curve),
        G_Mean(Geometric Mean)

    Return
    ------
    scores : list
        It contains performance scores
    """
    acc = np.sum(y_real == y_pred)/len(y_pred)
    f1 = f1_score(y_real, y_pred, average='binary')
    pre = precision_score(y_real, y_pred, zero_division=0)
    recall = recall_score(y_real, y_pred, zero_division=0)
    spec = sensitivity_specificity_support(y_real, y_pred, average='binary')[1]
    auc_score = roc_auc_score(y_real, y_proba[:, 1])
    g_mean = geometric_mean_score(y_real, y_pred, average='binary')
    scores = [acc, f1, pre, recall, spec, auc_score, g_mean]

    return scores


def section3():
    print("monitoring")
    return None


def get_datetime():
    now = time.localtime()
    output = "%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year,
                                                now.tm_mon, now.tm_mday,
                                                now.tm_hour, now.tm_min,
                                                now.tm_sec)

    return output


def get_from():
    return time.time()

def working_time(from_time):
    return time.time() - from_time

def section4():
    print("save")
    return None

def save_process(df, save_dir, f_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f_name), index=False)

def opj(*args):
    return os.path.join(*args)

def get_leaf_dir(*conditions):
    """"""
    if len(list(conditions)) > 1:
        sep = '\\'
        return sep.join(conditions)
    else:
        return ''.join(conditions)

def get_base_dir(root_dir, *leaf_dirs):
    return os.path.join(root_dir, get_leaf_dir(*leaf_dirs))

def get_full_path(base_dir, file_name):
    full_path = os.path.join(base_dir, file_name)
    return full_path

def open_dir(*dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)


def save_csv(df,base_dir,full_name, index=False):
    open_dir(base_dir)
    df.to_csv(os.path.join(base_dir, full_name), index=index)

def save_pickle(df,base_dir,full_name):
    open_dir(base_dir)
    df.to_pickle(os.path.join(base_dir, full_name))

def get_name(leaf_path):
    return os.path.splitext(leaf_path)[0]

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def get_data_name_list(data_dir):
    data_name_list = [csv.split('.')[0] for csv in os.listdir(data_dir)]
    return data_name_list


def get_file_names(dir_path, isint=True):
    if isint:
        file_names = [int(csv.split('.')[0]) for csv in os.listdir(dir_path)]
    else:
        file_names = [csv.split('.')[0] for csv in os.listdir(dir_path)]
    return file_names

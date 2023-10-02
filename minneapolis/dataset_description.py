# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 16:05:23 2023

@author: dohyeon
"""
import pandas as pd
temp_df = pd.read_csv(r'dataset/Motorized_Foot_Scooter_Trips_2022.csv')
temp_df = temp_df.loc[temp_df.StartCenterlineType != 'trail'].reset_index(drop=True)
temp_df = temp_df.loc[temp_df.EndCenterlineType != 'trail'].reset_index(drop=True)
temp_df.loc[:, 'start_month'] = pd.to_datetime(temp_df.StartTime).dt.month
temp_df.loc[:, 'end_month'] = pd.to_datetime(temp_df.EndTime).dt.month
temp_df.loc[:, 'start_hour'] = pd.to_datetime(temp_df.StartTime).dt.hour
temp_df.loc[:, 'end_hour'] = pd.to_datetime(temp_df.EndTime).dt.hour
temp_df.loc[:, 'start_day'] = pd.to_datetime(temp_df.StartTime).dt.day
temp_df.loc[:, 'end_day'] = pd.to_datetime(temp_df.EndTime).dt.day
temp_df.loc[:, 'year'] = pd.to_datetime(temp_df.EndTime).dt.year

###
new_temp_df = temp_df.loc[(((temp_df.start_month==7)&(temp_df.year==2022)) & ((temp_df.end_month==7)&(temp_df.year==2022)))|((temp_df.start_month==8)&(temp_df.year==2022) & ((temp_df.end_month==8)&(temp_df.year==2022)))].dropna().reset_index(drop=True)




new_temp_df.columns

new_temp_df.TripID.nunique()


temp_obj = pd.read_pickle(r'minneapolis_roughly_filtered_dataset_0530.pkl')

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 12:57:22 2021

@author: dohyeon
"""

import numpy as np
import pandas as pd
import time
from math import  pi
from shapely.geometry import Point, Polygon
import fiona
from haversine import haversine, inverse_haversine
import user_utils as uu
import matplotlib.pyplot as plt

#%%
if __name__ == '__main__':
    # Read history dataset (csv)
    temp_df = pd.read_csv(r'Microtransit__Scooter_and_Ebike__Trips.csv')
    temp_df.loc[:, 'start_date'] = pd.to_datetime(temp_df['Start Date'])
    temp_df.loc[:, 'end_date'] = pd.to_datetime(temp_df['End Date'])
    temp_df.loc[:, 'start_month'] = temp_df.start_date.dt.month
    temp_df.loc[:, 'end_month'] = temp_df.end_date.dt.month

    temp_df.loc[:, 'year'] = temp_df.start_date.dt.year
    temp_df = pd.DataFrame.sort_values(temp_df, by='start_date').reset_index(drop=True)
    #%%
    new_temp_df = temp_df.loc[(((temp_df.start_month==7)&(temp_df.year==2019)) & ((temp_df.end_month==7)&(temp_df.year==2019)))|((temp_df.start_month==8)&(temp_df.year==2019) & ((temp_df.end_month==8)&(temp_df.year==2019)))].dropna().reset_index(drop=True)
    new_temp_df.loc[:, 'start_day'] = [xx.day for xx in new_temp_df.start_date]
    new_temp_df.loc[:, 'end_day'] = [xx.day for xx in new_temp_df.end_date]


    new_temp_df.columns


    temp_obj = pd.read_pickle(r'roughly_filtered_dataset_0414.pkl')


    (temp_obj.groupby(['start_month','Day of Week', 'hour']).count().loc[7]['trip_id']/31).plot()

    (temp_obj.groupby(['start_month','Day of Week', 'hour']).count().loc[7]['trip_id']/31).unstack().T.plot()

    temp_obj.columns

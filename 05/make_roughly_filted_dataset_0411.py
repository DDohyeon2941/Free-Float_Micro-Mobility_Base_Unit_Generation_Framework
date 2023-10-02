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


def read_shp(shp_path):
    """single f"""
    f_list = []
    with fiona.open(shp_path) as src:
        for f in src:
            f_list.append(f)
    return f_list

def get_grid_coef(shp_list_):
    bin_li1 = []
    for b1 in shp_list_:
        c1 = b1['geometry']['coordinates'][0]
        d1 = [xx[0] for xx in c1]
        bin_li1 += d1
    bin_li2 = []
    for b1 in shp_list_:
        c1 = b1['geometry']['coordinates'][0]
        d2 = [xx[1] for xx in c1]
        bin_li2 += d2
    lon_min, lon_max = np.min(bin_li1), np.max(bin_li1)
    lat_min, lat_max = np.min(bin_li2), np.max(bin_li2)
    return lon_min, lon_max, lat_min, lat_max

def get_coef_one_grid(lon1, lat1, x_tick, y_tick):
    lon2 = lon1+x_tick
    lat2 = lat1+y_tick
    return Polygon([(lon1, lat1), (lon2,lat1), (lon2,lat2),(lon1, lat2) ])

def get_xy_tick_size(boundary_coef, size=0.5):
    tt1=(boundary_coef[2],boundary_coef[0])
    tt2=inverse_haversine(tt1, distance=size, direction=0)
    tt3=inverse_haversine(tt1, distance=size, direction=pi*0.5)

    #lon
    x_size= tt3[1] - tt1[1]
    #lat
    y_size = tt2[0]- tt1[0]

    return x_size, y_size


def generate_grid_in_boundary(boundary_coef, x_size, y_size):

    xrngs = np.arange(boundary_coef[0], boundary_coef[1], x_size)
    yrngs = np.arange(boundary_coef[2], boundary_coef[3], y_size)

    grid_list = []
    for uni_x in xrngs:
        for uni_y in yrngs:
            grid_list.append(get_coef_one_grid(uni_x, uni_y, x_size, y_size))
    return grid_list

def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


def extract_target_grids(usage_info, grid_info, isstart=True):

    if isstart:
        lat_col = 'Start Latitude'
        lon_col = 'Start Longitude'
    else:
        lat_col = 'End Latitude'
        lon_col = 'End Longitude'

    max1=usage_info[lat_col].max()
    min1=usage_info[lat_col].min()

    max2=usage_info[lon_col].max()
    min2=usage_info[lon_col].min()

    #범위내 grid rough하게 추출
    target_grids = [xx for x_idx, xx in enumerate(grid_info) if ((xx.bounds[1]<=max1)&(xx.bounds[1]>=min1)&(xx.bounds[0]>=min2)&(xx.bounds[0]<=max2))]
    target_idxs = [x_idx for x_idx, xx in enumerate(grid_info) if ((xx.bounds[1]<=max1)&(xx.bounds[1]>=min1)&(xx.bounds[0]>=min2)&(xx.bounds[0]<=max2))]

    return target_grids, target_idxs

def extract_target_grids_main(usage_info, grid_info):
    _, start_idx = extract_target_grids(usage_info, grid_info, True)
    _, end_idx = extract_target_grids(usage_info, grid_info, False)
    start_end_idx = np.intersect1d(start_idx, end_idx)
    start_end_grid = [grid_info[xx] for xx in start_end_idx]
    return start_end_grid, start_end_idx


@logging_time
def mask_located_grid(usage_info, target_cols, target_grid_info):
    grid_idx_list = []
    point_gen = (Point(uni_row) for _, uni_row in usage_info[target_cols].iterrows())

    for uni_pnt in point_gen:
        temp_len = len(grid_idx_list)
        for grid_idx, uni_grid in enumerate(target_grid_info):
            if uni_pnt.intersects(uni_grid):
                grid_idx_list.append(grid_idx)
                break
        if len(grid_idx_list) <= temp_len:
            grid_idx_list.append(np.nan)
    return grid_idx_list

def find_intersect_grid(usage_info):

    start_target_idx = usage_info.groupby(['start_grid','start_month']).count()['year'].unstack().dropna().index.tolist()
    end_target_idx = usage_info.groupby(['end_grid','end_month']).count()['year'].unstack().dropna().index.tolist()

    target_start_end_idx = np.intersect1d(start_target_idx, end_target_idx)
    return target_start_end_idx

def index_usage_df(usage_info, grid_idx_info):
    return usage_info.loc[((usage_info.start_grid.isin(grid_idx_info))&(usage_info.end_grid.isin(grid_idx_info)))].reset_index(drop=True)
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

    # service area를 Grid 단위로 나눔
    shape_list = read_shp(r"kansas_shp/geo_export_f3314f99-1250-4c35-bfa1-1e654fbe2772.shp")
    bdry_coef = get_grid_coef(shape_list)
    
    alpha, beta = get_xy_tick_size(bdry_coef, 0.5)
    
    grid_list2 = generate_grid_in_boundary(bdry_coef, alpha, beta)
    
    # 대상기간내, 대여기록이 존재한 그리드만 추출
    #masked_lat_lon1, _= extract_target_grids(new_temp_df, grid_list2)
    masked_lat_lon1, _= extract_target_grids_main(new_temp_df, grid_list2)


    #%% 대여와 반납이 발생한 대상 그리드 인덱스를 찾음

    print(uu.get_datetime())
    grid_idx_list = mask_located_grid(new_temp_df, target_cols=['Start Longitude','Start Latitude'],target_grid_info=masked_lat_lon1)
    
    grid_idx_list_end = mask_located_grid(new_temp_df, target_cols=['End Longitude','End Latitude'],target_grid_info=masked_lat_lon1)
    print(uu.get_datetime())


    new_temp_df.loc[:,'start_grid'] = grid_idx_list
    new_temp_df.loc[:,'end_grid'] = grid_idx_list_end

    # 대여 또는 반납이 대상 그리드에서 발생하지 않은 경우 nan값 산출 >> 실험대상에서 제외

    new_temp_df = new_temp_df.dropna().reset_index(drop=True).astype({'start_grid':int, 'end_grid':int})

    # 특정 그리드가 대상 그리드에서 제외되는 경우, 실험에서 제외할 대여기록이 추가로 발생하게 됨
    # 실험에서 제외할 대여기록이 더이상 발생하지 않을때까지 이를 반복함
    #%% 실험에서 제외할 대여기록이 더이상 발생하지 않을때까지 이를 반복함
    keep_indexing = True
    
    while keep_indexing:
        start_end_idx= find_intersect_grid(new_temp_df)
        num_grid1 = start_end_idx.shape[0]
        new_temp_df = index_usage_df(new_temp_df, start_end_idx)
    
        if num_grid1 == find_intersect_grid(new_temp_df).shape[0]:
            keep_indexing=False
    #%% 저장

    uu.save_gpickle('grid_elements_0414.pickle', {'grid_idx_list':start_end_idx, 'grid_list':masked_lat_lon1,  'alpha':alpha, 'beta':beta})


    new_temp_df.to_pickle(r'roughly_filtered_dataset_0414.pkl')
    
    pd.DataFrame(data=[xx.wkt for xx in grid_list2]).to_csv(r'All_500m.csv')
    pd.DataFrame(data=[masked_lat_lon1[xx].wkt for xx in start_end_idx]).to_csv('Base_500m.csv')

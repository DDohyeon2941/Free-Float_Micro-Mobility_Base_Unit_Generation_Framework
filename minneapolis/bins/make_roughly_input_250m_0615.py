# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:14:40 2023

@author: dohyeon
"""
import numpy as np
import pandas as pd
from math import  pi
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
import shapely.wkt
from haversine import haversine, inverse_haversine
import fiona
import user_utils as uu
import time


def read_shp(shp_path):
    """single f"""
    f_list = []
    with fiona.open(shp_path) as src:
        for f in src:
            try:
                f_list.append(f)
            except: f_list.append(0)
    return f_list

def get_grid_coef(shp_list_):
    bin_li1 = []
    for b1 in shp_list_:
        c1 = b1['geometry']['coordinates']
        d1 = [xx[0] for xx in c1]
        bin_li1 += d1
    bin_li2 = []
    for b1 in shp_list_:
        c1 = b1['geometry']['coordinates']
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




#%%
if __name__ == '__main__':
    
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

    ###
    temp_obj = read_shp(r'dataset\MPLS_Centerline\MPLS_Centerline.shp')
    
    bdry_coef = get_grid_coef(temp_obj)
    
    alpha, beta = get_xy_tick_size(bdry_coef, 0.25)

    grid_list2 = generate_grid_in_boundary(bdry_coef, alpha, beta)

    #pd.DataFrame(data=[xx.wkt for xx in grid_list2]).to_csv(r'minneapolis_all_grid.csv')
    ###
    a1 = [xx['properties']['GBSID'] for xx in temp_obj]
    a2 = [LineString(xx['geometry']['coordinates']).centroid for xx in temp_obj]

    GBSID_dict = dict(zip(a1,a2))

    ###
    new_temp_df = new_temp_df.loc[new_temp_df.StartCenterlineID.astype(np.int64).isin(GBSID_dict.keys())]
    new_temp_df = new_temp_df.loc[new_temp_df.EndCenterlineID.astype(np.int64).isin(GBSID_dict.keys())]

    #%%
    keep_indexing = True
    while keep_indexing:
        train_df = new_temp_df.loc[(new_temp_df.start_month ==7)&(new_temp_df.end_month ==7)]
        test_df = new_temp_df.loc[(new_temp_df.start_month ==8)&(new_temp_df.end_month ==8)]

        intersect_idx_trn = np.intersect1d(train_df.EndCenterlineID.unique(),train_df.StartCenterlineID.unique())
        intersect_idx_tst = np.intersect1d(test_df.EndCenterlineID.unique(),test_df.StartCenterlineID.unique())
        intersect_idx = np.intersect1d(intersect_idx_trn, intersect_idx_tst)
        intersect_idx1 = np.intersect1d(intersect_idx_tst,intersect_idx_trn)
        intersect_idx2 = np.intersect1d(intersect_idx1, intersect_idx)


        new_temp_df = (new_temp_df.loc[(new_temp_df.StartCenterlineID.isin(intersect_idx2))&(new_temp_df.EndCenterlineID.isin(intersect_idx2))]).reset_index(drop=True)

        if len(intersect_idx_trn) != len(intersect_idx_tst):
            keep_indexing=True
        else:
            if len(intersect_idx_trn) != len(intersect_idx2):
                keep_indexing = True
            else:
                keep_indexing=False


    #%%
    new_temp_df.loc[:, 'start_point'] = [GBSID_dict[xx] for xx in new_temp_df.StartCenterlineID.astype(np.int64)]
    new_temp_df.loc[:, 'end_point'] = [GBSID_dict[xx] for xx in new_temp_df.EndCenterlineID.astype(np.int64)]


    #%%

    grid_idx_list = []
    grid_poly_list = []
    for grid_idx, grid_poly in enumerate(grid_list2):
        for uni_start_pnt in new_temp_df['start_point'].unique():
            if grid_poly.intersects(uni_start_pnt):
                grid_idx_list.append(grid_idx)
                grid_poly_list.append(grid_poly)
                break

#%%
    pd.DataFrame(data=[xx.wkt for xx in grid_poly_list]).to_csv(r'minneapolis_filtered_grid_250m.csv')
    """

    pd.DataFrame(data=[xx.wkt for xx in new_temp_df.start_point.unique()]).to_csv(r'minneapolis_start_unique_points.csv')
    pd.DataFrame(data=[xx.wkt for xx in new_temp_df.end_point.unique()]).to_csv(r'minneapolis_end_unique_points.csv')
    """
    #%%

    start_list = []
    for uni_pnt in new_temp_df.start_point:
        for uni_idx, uni_poly in enumerate(grid_poly_list):
            if uni_pnt.intersects(uni_poly):
                start_list.append(uni_idx)
                break

    #%%

    end_list = []
    for uni_pnt in new_temp_df.end_point:
        for uni_idx, uni_poly in enumerate(grid_poly_list):
            if uni_pnt.intersects(uni_poly):
                end_list.append(uni_idx)
                break
    #%%
    new_temp_df.loc[:, 'start_grid'] = start_list
    new_temp_df.loc[:, 'end_grid'] = end_list



    uu.save_gpickle('minneapolis_grid_elements_250m_0615.pickle', {'grid_list':grid_poly_list,  'alpha':alpha, 'beta':beta})


    new_temp_df.to_pickle(r'minneapolis_roughly_filtered_dataset_250m_0615.pkl')

    #%%
    import matplotlib.pyplot as plt

    plt.plot(new_temp_df.groupby('start_grid').count()['ObjectId'].sort_values().values)



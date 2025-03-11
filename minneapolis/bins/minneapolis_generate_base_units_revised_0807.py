# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 12:35:42 2023

@author: dohyeon
"""


import user_utils as uu
import pandas as pd
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon

temp_df = pd.read_pickle(r'minneapolis_roughly_filtered_dataset_0530.pkl')
temp_obj = uu.load_gpickle(r'minneapolis_grid_elements_0530.pickle')
base_sub_grid_dic = {0:[0,1,4,5], 1:[2,3,6,7],2:[8,9,12,13],3:[10,11,14,15]}


#%%
shp_dict = {}

thr_val = 5
thr_val2 = 8
thr_val3 = 4
num_clusters = 45

#각 샘플별 속하는 Sub-grid의 인덱스를 구함 >> base-grid 인덱스를 구할 예정
###### 1)
def get_coef_one_grid(lon1, lat1, x_tick, y_tick):
    """하나의 point를 기준으로 하나의 grid polygon 생성"""
    lon2 = lon1+x_tick
    lat2 = lat1+y_tick
    return Polygon([(lon1, lat1), (lon2,lat1), (lon2,lat2),(lon1, lat2) ])

def split_grid_uni1(uni_grid, x_size, y_size):
    """하나의 fixed grid를 여러 subgrid로 분할함"""
    xrngs1 = np.arange(uni_grid.bounds[0], uni_grid.bounds[2], x_size)
    yrngs1 = np.arange(uni_grid.bounds[1], uni_grid.bounds[3], y_size)

    base_tups1 = [get_coef_one_grid(uni_x, uni_y, x_size, y_size) for uni_x in xrngs1 for uni_y in yrngs1]

    return base_tups1

def get_subgrid_dic1(grid_list, nsplit, alpha, beta):
    """fixed grid polygon들을 대상으로, subgrid 단위로 쪼갬"""
    subgrid_dic = {}
    for uni_temp_idx, uni_temp_poly in enumerate(grid_list):
        subgrid_dic[uni_temp_idx] = split_grid_uni1(uni_grid=uni_temp_poly,
                                                    x_size = alpha/nsplit, y_size=beta/nsplit)
    return subgrid_dic

###### 2)

def get_subgrid_idx(subgrid_polygons, usage_info,   target_cols=['start_grid','start_point']):
    """대여기록별 대여가 발생한 subgrid의 인덱스를 산출함"""
    subgrid_idx_list = []
    point_start1 = ((uni_row[0], uni_row[1]) for _, uni_row in usage_info[target_cols].iterrows())

    for uni_pnt in point_start1:
        for sub_idx, sub_poly in enumerate(subgrid_polygons[uni_pnt[0]]):
            if sub_poly.intersects(uni_pnt[1]):
                subgrid_idx_list.append(sub_idx)
                break
    return subgrid_idx_list

###### 3)


def get_base_idx(subgrid_idx_uni):
    """subgrid의 인덱스를 기반해 base grid의 인덱스를 산출함"""
    if subgrid_idx_uni in base_sub_grid_dic[0]: return 0
    elif subgrid_idx_uni in base_sub_grid_dic[1]: return 1
    elif subgrid_idx_uni in base_sub_grid_dic[2]: return 2
    elif subgrid_idx_uni in base_sub_grid_dic[3]: return 3


def get_base_grid_idx_4_split(subgrid_idx_list):
    """각 subgrid별 속해있는 base grid의 인덱스를 산출"""
    return [get_base_idx(xx) for xx in subgrid_idx_list]

###### 4)


def merge_subgrid_uni_base(subgrid_info, fixed_idx, sub_idxs):
    """subgrid를 병합해 basegrid로 만듬"""
    return MultiPolygon([subgrid_info[fixed_idx][xx] for xx in sub_idxs])

def merge_subgrid_uni(subgrid_info, fixed_idx):
    """fixed grid 관점에서 merge_subgrid_uni_base 진행 (4등분하므로, 총 4군데)"""
    p1=merge_subgrid_uni_base(subgrid_info, fixed_idx, base_sub_grid_dic[0])
    p2=merge_subgrid_uni_base(subgrid_info, fixed_idx, base_sub_grid_dic[1])
    p3=merge_subgrid_uni_base(subgrid_info, fixed_idx, base_sub_grid_dic[2])
    p4=merge_subgrid_uni_base(subgrid_info, fixed_idx, base_sub_grid_dic[3])
    return [(fixed_idx, 0), (fixed_idx, 1), (fixed_idx, 2), (fixed_idx, 3)], [p1, p2, p3, p4]

def merge_subgrid(subgrid_info):
    """merge_subgrid_uni를 여러 fixed grid 단위로 진행함"""
    key_li = []
    val_li = []
    for uni_fixed_idx in subgrid_info.keys():
        uni_key, uni_val = merge_subgrid_uni(subgrid_info, uni_fixed_idx)
        key_li += uni_key
        val_li += uni_val
    return dict(zip(key_li, val_li))

###### 5)


def mask_skewed_central(sub_usage_info, threshold1=5):
    test_row_list = []
    for _, test_row1 in sub_usage_info.iterrows():
        if test_row1.sum() < threshold1: test_row_list.append(-1)
        else:
            if np.max(test_row1 / test_row1.sum()) > 0.5:
                test_row_list.append(np.argmax(test_row1 / test_row1.sum()))
            else: test_row_list.append(-1)
    return test_row_list

## fixed grid, base grid, sub grid 각각에 대한 central point coef 산출

###### 6)

def get_skewed_central_coef1(base_central_mask_info, basegrid_info, subgrid_info):
    """특정 base grid의 central point를 해당 base grid를 구성하는 
    4개의 subgrid 중 한 subgrid의 central point로 대체하는지 여부를 마스킹"""
    central_coef_df = pd.DataFrame(index=base_central_mask_info.index,
                                   columns=base_central_mask_info['cp_mask'].columns)
    for g_idx in base_central_mask_info.index:
        for bg_idx in [0,1,2,3]:
            max_subgrid = base_central_mask_info.loc[g_idx][bg_idx]
            if max_subgrid == -1:
                central_coef_df.loc[g_idx][bg_idx] = basegrid_info[(g_idx, bg_idx)].centroid
            else:
                central_coef_df.loc[g_idx][bg_idx] = subgrid_info[g_idx][max_subgrid].centroid
    return central_coef_df

#%% Base Grid의 Polygon과 central coef를 구하기

#Base Grid 정보를 기반해, Subgrid dict 만들기

element1 = {}


#### Base Grid에 대한 polygon 생성
element1['subgrid_dic'] = get_subgrid_dic1(temp_obj['grid_list'], float(4), temp_obj['alpha'], temp_obj['beta'])
element1['basegrid_dic'] = merge_subgrid(element1['subgrid_dic'])
####

#### 각 샘플별 속하는 Grid의 인덱스를 구하기
# 각 샘플별 속하는 Sub-grid 인덱스를 구함
# sub-grid 인덱스를 기반해 Base-grid의 인덱스를 구함
#%%
element1['start_subgrid_idxs'] = get_subgrid_idx(element1['subgrid_dic'], temp_df)
element1['end_subgrid_idxs'] = get_subgrid_idx(element1['subgrid_dic'], temp_df, ['end_grid','end_point'])


temp_df.loc[:, 'start_subgrid_idx'] = element1['start_subgrid_idxs']
temp_df.loc[:, 'end_subgrid_idx'] = element1['end_subgrid_idxs']


temp_df.loc[:, 'start_basegrid_idx'] =get_base_grid_idx_4_split(element1['start_subgrid_idxs'])

temp_df.loc[:, 'end_basegrid_idx'] =get_base_grid_idx_4_split(element1['end_subgrid_idxs'])


#%%
#### Skewness 정의하기
# Fixed Grid의 base grid-sub-grid별 일 평균 대여량
element1['subgrid_avg_demand'] = ((((temp_df.groupby(['start_month','start_grid','start_basegrid_idx','start_subgrid_idx']).count()).loc[7])['ObjectId'])/31).unstack([-2,-1],fill_value=0).stack(-2).fillna(0)


#### Base Grid별 Central Point 산출하기
# Base Grid별 central point의 위치를 마스킹 (skewness 반영하지 않으면 -1, 그외는 subgrid의 인덱스를 표시)
element1['base_skewed_mask'] = pd.DataFrame(index=element1['subgrid_avg_demand'].index,
                                            data=mask_skewed_central(element1['subgrid_avg_demand'], threshold1=thr_val),
                                            columns=['cp_mask']).unstack()

# Skewness를 반영해 Base Grid별 central coef를 구함
element1['basegrid_central_coef']= get_skewed_central_coef1(element1['base_skewed_mask'],
                                                            element1['basegrid_dic'],
                                                            element1['subgrid_dic'])

#%% 2) Main Grid, Support Grid 선정하기

from haversine import haversine
from shapely.ops import cascaded_union, unary_union


def convert_str_float(temp_str):
    """str형태의 좌표정보를 float 형태로 변환, Point 한정"""
    return (float(temp_str.split()[1][1:]), float(temp_str.split()[2][:-1]))


def split_main_support_grid(usage_info, central_coef_info, threshold2=5):
    """base grid들을 main과 support grid로 구분
    그 기준은, 일평균 대여량이 threshold2 이상여부에 따름"""

    is_maingrid_mask=(usage_info.sum(axis=1).unstack()>=threshold2)*1
    maingrid_mask = np.where(is_maingrid_mask == 1)
    supgrid_mask = np.where(is_maingrid_mask == 0)

    main_row_col_pair = list(zip(is_maingrid_mask.index.values[maingrid_mask[0]],maingrid_mask[1]))
    support_row_col_pair = list(zip(is_maingrid_mask.index.values[supgrid_mask[0]],supgrid_mask[1]))

    main_central_coef_dic = dict(zip(main_row_col_pair,[central_coef_info.loc[xx[0]][xx[1]] for xx in main_row_col_pair]))
    support_central_coef_dic = dict(zip(support_row_col_pair,[central_coef_info.loc[xx[0]][xx[1]] for xx in support_row_col_pair]))
    return main_central_coef_dic, support_central_coef_dic

# Support Grid별 가장 가까운 Main Grid의 인덱스를 구하기

def get_nn_info_from_support_raw(main_coef_dic, support_coef_dic):
    """support grid별로 가장 가까운 Main grid Index(nn_list)와 거리 구하기(nn_list1)"""
    nn_index_list = []
    nn_dist_list = []
    for uni_key in support_coef_dic.keys():
        dist_li = [haversine(
            convert_str_float(support_coef_dic[(uni_key[0], uni_key[1])].wkt)[::-1],
            convert_str_float(main_coef_dic[(xx,yy)].wkt)[::-1]) for xx, yy in main_coef_dic.keys()]
        nn_index_list.append(list(main_coef_dic.keys())[np.argmin(dist_li)])
        nn_dist_list.append(np.min(dist_li))
    return nn_index_list, nn_dist_list

def get_nn_info_from_support(main_dic, support_coef_dic):
    """main grid별 support grid 기준, 가장 가깝다고 표시된 경우를 모두 산출"""
    nn_index_list, nn_dist_list = get_nn_info_from_support_raw(main_dic, support_coef_dic)

    nn_index_dict  = {}
    nn_dist_dict  = {}

    for nn_idx, nn_key in enumerate(nn_index_list):
        if nn_key not in nn_index_dict.keys():
            nn_index_dict[nn_key]=[list(support_coef_dic.keys())[nn_idx], ]
            nn_dist_dict[nn_key]=[nn_dist_list[nn_idx], ]
        else:
            nn_index_dict[nn_key].append(list(support_coef_dic.keys())[nn_idx])
            nn_dist_dict[nn_key].append(nn_dist_list[nn_idx], )
    return nn_index_dict, nn_dist_dict

def divide_support_grid1(nn_idx_info, nn_dist_info, basegrid_info, threshold3=3):
    """main grid별 최종 support grid pair 선정 [merge를 위해서]"""
    solo_support_list = []
    #solo_fraction_list = []
    nn_dict1 = {}
    for xx,yy in nn_dist_info.items():
        if xx not in nn_dict1.keys():
            nn_dict1[xx] = []
        knn_list = []
        knn_list1 = []
        for zz in np.argsort(yy)[:threshold3]:
            poly1 = basegrid_info[xx]
            poly2 = basegrid_info[nn_idx_info[xx][zz]]
            tup11 = convert_str_float(poly1.centroid.wkt)
            tup22 = convert_str_float(poly2.centroid.wkt)

            if poly1.intersects(poly2):
                if not (tup11[0] != tup22[0]) & (tup11[1] != tup22[1]):
                    knn_list.append(nn_idx_info[xx][zz])
                else:
                    knn_list1.append(nn_idx_info[xx][zz])
                    #solo_fraction_list.append(cascaded_union(poly2))
            else:
                knn_list1.append(nn_idx_info[xx][zz])
        knn_list1 += [nn_idx_info[xx][zz] for zz in np.argsort(yy)[threshold3:]]

        nn_dict1[xx]+=knn_list
        solo_support_list+=knn_list1

    return nn_dict1, solo_support_list

# Main Grid와 결합되지 못한 Support Grid 시각화

def conv_support_to_fixed(support_info):
    """main grid와 병합되지 못한 support grid를 fixed grid단위로 병합함"""
    ii=0
    idict={}
    for fixed_idx, fixed_base_idx_df in pd.DataFrame(support_info).groupby(0):
        idict[ii]=[(fixed_idx,bbb) for bbb in fixed_base_idx_df[1].values]
        ii+=1
    return idict


##Main Grid (Support grid와 결합), 결합안된 Support Grid에 대한 인덱스 추출

def get_main_dic(main_support_info, solo_main_info):
    """main grid별 포함하는 base grid 인덱스 정보 산출"""
    main_masked_dic = {}

    for kvidx, (main_idx, support_idx) in enumerate(main_support_info.items()):
        main_masked_dic[kvidx] = [main_idx,]+support_idx

    for main_idx in solo_main_info:
        kvidx +=1
        main_masked_dic[kvidx] = main_idx

    return main_masked_dic


element2 = {}

element2['main_central_point'], element2['support_central_point'] = split_main_support_grid(element1['subgrid_avg_demand'], element1['basegrid_central_coef'], threshold2=thr_val2)

element2['main_support_idx'], element2['main_support_dist'] = get_nn_info_from_support(element2['main_central_point'], element2['support_central_point'])

#element2['merged_main_idx'], element2['solo_support_idx'], element2['solo_support_fraction'] = divide_support_grid1(element2['main_support_idx'],element2['main_support_dist'], element1['basegrid_dic'],threshold3=thr_val3)

element2['merged_main_idx'], element2['solo_support_idx']= divide_support_grid1(element2['main_support_idx'],element2['main_support_dist'], element1['basegrid_dic'],threshold3=thr_val3)


element2['solo_main_idx'] = [xx for xx in list(element2['main_central_point'].keys()) if xx not in list(element2['main_support_idx'].keys())]
element2['fixed_support_idx'] = conv_support_to_fixed(element2['solo_support_idx'])

main_masked_dic = get_main_dic(element2['merged_main_idx'], element2['solo_main_idx'])

#%%

def get_main_grid_polygon(main_masked_info, basegrid_info):
    """main grid별 polygon 산출"""
    polygon_list = []
    for base_idxs in main_masked_info.values():
        if isinstance(base_idxs, list): polygon_list.append(unary_union([basegrid_info[xx] for xx in base_idxs]))
        else: polygon_list.append(unary_union(basegrid_info[base_idxs]))
    return polygon_list



shp_dict['main_all_poly'] = get_main_grid_polygon(main_masked_dic, element1['basegrid_dic'])

pd.DataFrame(data=[xx.wkt for xx in shp_dict['main_all_poly']]).to_csv(r'main_all_poly_0809.csv')
#%% Spatial representation

def get_pairwise_dist(main_mask_info, central_coef_info):
    """서로 다른 main grid들의 central point간 거리를 구함"""
    main_idx_list = [xx[0] if isinstance(xx[0], tuple) else xx for xx in main_mask_info.values() ]
    main_central_list = [convert_str_float(central_coef_info.loc[xx].wkt)[::-1] for xx in main_idx_list]
    dist_df = pd.DataFrame([[haversine(xx,yy) for yy in main_central_list] for xx in main_central_list])
    return dist_df

def intersect_types(main_poly_info):
    """서로 다른 main grid들의 인접(intersection)여부를 확인"""
    intersect_count=0
    intersect_type_list = []
    for uni_main_poly1 in main_poly_info:
        #uni_main_poly1 = uni_main_poly1.buffer(10**-15)
        intersect_mask_list = []
        for uni_main_poly2 in main_poly_info:
            if not uni_main_poly1.intersects(uni_main_poly2):
                intersect_mask_list.append(1)
            else:
                intersect_type = uni_main_poly1.intersection(uni_main_poly2).geom_type
                print(intersect_type)
                intersect_count+=1
                #print(MultiPolygon([uni_main_poly1, uni_main_poly2]))
                if intersect_type =='Point': intersect_mask_list.append(1)
                else: intersect_mask_list.append(0)
        intersect_type_list.append(intersect_mask_list)
    print(intersect_count)
    return intersect_type_list


def find_diag_grid_pair(shp_info, pairwise_dist_info):
    """main grid들 중 서로 최근접 이웃이면서, 대각으로 연결된 경우를 산출
    대각으로 연결된 경우는 intersection 영역의 type이 point인 경우임"""
    diag_intersect_mask_df = pd.DataFrame(data=intersect_types(shp_info['main_all_poly']))
    diag_tup_list  = list(zip(*np.where(diag_intersect_mask_df==1)))
    nn_tup_list = list(zip(np.arange(pairwise_dist_info.shape[0]),
                   pairwise_dist_info.replace(0,100).idxmin(axis=1).values))

    nn_dict0 = dict(nn_tup_list)

    
    nn_dict = {}
    for aa,bb in nn_dict0.items():
        if (nn_dict0[aa] == bb) & (nn_dict0[bb] == aa):
            if (aa not in nn_dict.keys()) & (aa not in nn_dict.values()):
                nn_dict[aa]=bb
    
    return dict([nn_pair for nn_pair in nn_dict.items() if nn_pair in diag_tup_list])


def extract_spatio_repr(dist_info, shp_info):

    """Main Grid별 spatial representation 정보 산출"""

    CONSTANT = 5
    MULTIPLYER = 2

    rel_dist_arr_2d = np.vstack([np.where(np.argsort(dist_info.values,axis=1)==xx)[1] for xx in range(dist_info.shape[0])])
    rel_weight_2d= (1/(1+(MULTIPLYER*rel_dist_arr_2d)))
    rel_weight_2d1 = rel_weight_2d.copy()

    nn_diag_dic = find_diag_grid_pair(shp_info, dist_info)
    print(nn_diag_dic)
    for main_idx1,main_idx2 in nn_diag_dic.items():
        rel_weight_2d1[main_idx1,main_idx2] = rel_weight_2d[main_idx1,main_idx2]/CONSTANT

    return rel_weight_2d1

def extract_spatio_repr1(dist_info, shp_info):

    """Main Grid별 spatial representation 정보 산출"""

    CONSTANT = 5
    MULTIPLYER = 2

    rel_weight_2d= (1/(1+(MULTIPLYER*dist_info)))
    rel_weight_2d1 = rel_weight_2d.copy()

    nn_diag_dic = find_diag_grid_pair(shp_info, dist_info)
    print(nn_diag_dic)
    for main_idx1,main_idx2 in nn_diag_dic.items():
        rel_weight_2d1.loc[main_idx1,main_idx2] = 0
        rel_weight_2d1.loc[main_idx2,main_idx1] = 0

    return rel_weight_2d1



def find_diag_grid_pair1(shp_info, pairwise_dist_info):
    """main grid들 중 서로 최근접 이웃이면서, 대각으로 연결된 경우를 산출
    대각으로 연결된 경우는 intersection 영역의 type이 point인 경우임"""
    diag_intersect_mask_df = pd.DataFrame(data=intersect_types(shp_info['main_all_poly']))
    diag_tup_list  = list(zip(*np.where(diag_intersect_mask_df==1)))
    nn_tup_list = list(zip(np.arange(pairwise_dist_info.shape[0]),
                   pairwise_dist_info.replace(0,100).idxmin(axis=1).values))
    return diag_tup_list, nn_tup_list



#%%
element3 = {}
element3['pairwise_dist'] = get_pairwise_dist(main_masked_dic, element1['basegrid_central_coef'])
element3['spatio_repr'] = extract_spatio_repr1(element3['pairwise_dist'], shp_dict)



#%% temporal representation
"""클러스터링"""
#클러스터링을 위한 데이터 전처리

def get_main_masking(usage_info, main_mask_info, target_cols=['start_grid','start_basegrid_idx', 'start_subgrid_idx']):
    """대여기록별 발생한 main grid 마스킹"""
    main_masked_li = []
    for _, (fixed_idx, base_idx, sub_idx) in usage_info[target_cols].iterrows():
        temp_len = len(main_masked_li)
        for main_idx, base_tup_list in main_mask_info.items():
            if isinstance(base_tup_list, list):
                if (fixed_idx,base_idx) in base_tup_list:
                    main_masked_li.append(main_idx)
                    break
            elif isinstance(base_tup_list, tuple):
                if (fixed_idx,base_idx) == base_tup_list:
                    main_masked_li.append(main_idx)
                    break
        if temp_len == len(main_masked_li): main_masked_li.append(np.nan)
    return main_masked_li

def cos_sim(a,b):
    return np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))

def extract_temporal_repr(usage_info, main_masked_info):
    """Main Grid별 temporal representation 정보 산출"""
    target_cols1 = ['start_month','start_new_mask','start_hour']
    target_cols2 = ['end_month','end_new_mask','end_hour']


    start_mask_h1 =usage_info.dropna().groupby(target_cols1).count().loc[7]['ObjectId'].unstack().fillna(0)
    end_mask_h1 =usage_info.dropna().groupby(target_cols2).count().loc[7]['ObjectId'].unstack().fillna(0)

    base_h1 = pd.DataFrame(index=np.arange(len(main_masked_info)), columns=np.arange(24)).fillna(0)
    base_h1.loc[start_mask_h1.index, start_mask_h1.columns] = start_mask_h1.values

    base_h2 = pd.DataFrame(index=np.arange(len(main_masked_info)), columns=np.arange(24)).fillna(0)
    base_h2.loc[end_mask_h1.index, end_mask_h1.columns] = end_mask_h1.values

    base_h3 = (base_h1/(base_h1 + base_h2)).fillna(0)

    temporal_repr = np.vstack([np.apply_along_axis(cos_sim, 1, base_h3, base_h3.loc[xx]) for xx in range(len(base_h3))])

    return temporal_repr
#%%
temp_df.loc[:, 'start_new_mask'] = get_main_masking(temp_df, main_masked_dic)
temp_df.loc[:, 'end_new_mask'] = get_main_masking(temp_df, main_masked_dic, ['end_grid','end_basegrid_idx', 'end_subgrid_idx'])


#temp_df.loc[:, 'end_hour'] = pd.to_datetime(temp_df['end_hour']).dt.hour

element3['temporal_repr'] = extract_temporal_repr(temp_df, main_masked_dic)


#plt.plot(element3['temporal_repr'][[2,21]].T)
#plt.plot(element3['spatio_repr'][[2,21]].T)

#%% Clustering
from sklearn.cluster import AgglomerativeClustering
agglomerative_model = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete')
clustering_input = 1-(element3['temporal_repr'] * element3['spatio_repr'])
for i in clustering_input.columns:
    clustering_input.at[i, i] = 0

agglomerative_result = agglomerative_model.fit_predict(clustering_input)
# 클러스터링 결과를 기반해서 마스킹 변수 생성하기

def get_base_unit_main(clustering_result, main_masked_info):
    """base unit별 포함하는 base grid 인덱스 정보 산출"""
    new_dict = {}
    for rel_idx, cluster_idx in enumerate(clustering_result):
        if cluster_idx not in new_dict.keys():
            new_dict[cluster_idx] = []
        base_idx_list = main_masked_info[rel_idx]
        if isinstance(base_idx_list, tuple): new_dict[cluster_idx].append(base_idx_list)
        elif isinstance(base_idx_list, list): new_dict[cluster_idx]+=base_idx_list
    base_unit_dict = dict(zip(np.arange(len(new_dict)),[new_dict[xx] for xx in np.arange(len(new_dict))]))
    return base_unit_dict


def get_base_unit(clustering_result, main_masked_info, support_masked_info):
    """base unit별 포함하는 base grid 인덱스 정보 산출 [support grid까지 포함]"""
    base_unit_dict = get_base_unit_main(clustering_result, main_masked_info)
    ii=len(base_unit_dict)
    for _,base_grid_idx in support_masked_info.items():
        base_unit_dict[ii]=base_grid_idx
        ii+=1
    return base_unit_dict

def mask_base_unit(usage_info, baseunit_info):
    """대여기록별 발생한 base unit 마스킹"""
    target_cols = ['start_grid','start_basegrid_idx']
    main_masked_li1 = []
    for _, (fixed_idx, base_idx) in usage_info[target_cols].iterrows():
        for base_unit_idx, base_grid_idxs in baseunit_info.items():
            if isinstance(base_grid_idxs, list):
                if (fixed_idx,base_idx) in base_grid_idxs:
                    main_masked_li1.append(base_unit_idx)
                    break
            elif isinstance(base_grid_idxs, tuple):
                if (fixed_idx,base_idx) == base_grid_idxs:
                    main_masked_li1.append(base_unit_idx)
                    break
    return main_masked_li1

element3['base_unit_idx_only_main'] = get_base_unit_main(agglomerative_result, main_masked_dic)
element3['base_unit_idx'] = get_base_unit(agglomerative_result, main_masked_dic, element2['fixed_support_idx'])


temp_df.loc[:, 'final_new_mask'] = mask_base_unit(temp_df, element3['base_unit_idx'])


#%%
shp_dict['raw_main_poly'] = [unary_union(MultiPolygon(element1['basegrid_dic'][xx])) for xx in list(element2['main_central_point'].keys())]

shp_dict['raw_support_poly'] = [unary_union(MultiPolygon(element1['basegrid_dic'][xx])) for xx in list(element2['support_central_point'].keys())]
shp_dict['solo_main_poly'] = [unary_union(MultiPolygon(element1['basegrid_dic'][xx])) for xx in element2['solo_main_idx']]

##

#shp_dict['main_all_central_point'] = get_main_central_coef(main_masked_dic, element1['basegrid_central_coef'])
shp_dict['main_all_poly'] = get_main_grid_polygon(main_masked_dic, element1['basegrid_dic'])


###
shp_dict['base_unit_poly_only_main'] = [unary_union([element1['basegrid_dic'][nn_tup] for nn_tup in nn_tups]) if len(nn_tups)>1 else unary_union(element1['basegrid_dic'][nn_tups[0]]) for nn_idx, nn_tups in element3['base_unit_idx_only_main'].items()]

shp_dict['base_unit_poly_all'] = [unary_union([element1['basegrid_dic'][nn_tup] for nn_tup in nn_tups]) if len(nn_tups)>1 else unary_union(element1['basegrid_dic'][nn_tups[0]]) for nn_idx, nn_tups in element3['base_unit_idx'].items()]

shp_dict['support_fixed_poly'] = [unary_union([element1['basegrid_dic'][nn_tup] for nn_tup in nn_tups]) if len(nn_tups)>1 else unary_union(element1['basegrid_dic'][nn_tups[0]]) for nn_idx, nn_tups in element2['fixed_support_idx'].items()]

shp_dict['main_support_poly'] = [unary_union([element1['basegrid_dic'][xx] for xx in element2['merged_main_idx'][(nn_start,nn_base)]+[(nn_start,nn_base),]]) for nn_start, nn_base in element2['merged_main_idx'].keys()]

#%%

len(shp_dict['base_unit_poly_all'] )

uu.save_gpickle(r'minneapolis_generated_elements_0809.pickle',{'ele1':element1, 'ele2':element2, 'ele3':element3, 'shp_dict':shp_dict})


len(shp_dict['main_all_poly'])

#%%

pd.DataFrame(data=[xx.wkt for xx in shp_dict['main_all_poly']]).to_csv(r'minneapolis_main_all_poly_0809.csv')


#pd.DataFrame(data=[xx.wkt for xx in shp_dict['main_support_poly']]).to_csv(r'kansas_main_support_poly.csv')

pd.DataFrame(data=[unary_union(xx).buffer(10**-14).wkt for xx in shp_dict['base_unit_poly_only_main']]).to_csv(r'minneapolis_base_unit_main_poly_0809.csv')

pd.DataFrame(data=[xx.buffer(10**-14).wkt for xx in shp_dict['support_fixed_poly']]).to_csv(r'minneapolis_support_fixed_poly_0809.csv')


#len(shp_dict['support_fixed_poly'])
#%%

uu.save_gpickle(r'minneapolis_prop_45c_357f_0809.pickle', element3['base_unit_idx'])
temp_df.to_pickle(r'minneapolis_500m_4b_45c_dataset_0809.pkl')

#%%
fixed_grid_dic = {}
for uni_idx, _ in enumerate(temp_obj['grid_list']):
    fixed_grid_dic[uni_idx] = [(uni_idx,0), (uni_idx,1), (uni_idx,2), (uni_idx,3)]

uu.save_gpickle(r'minneapolis_500m_366g_0809.pickle',fixed_grid_dic)

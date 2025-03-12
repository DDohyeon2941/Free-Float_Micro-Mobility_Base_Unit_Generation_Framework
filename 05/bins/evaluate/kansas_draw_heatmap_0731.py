# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:33:37 2023

@author: dohyeon
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler

from shapely.geometry import Polygon



temp_df = pd.read_csv(r'central_point_target_0607.csv')
temp_df['geometry'] = temp_df['0'].apply(wkt.loads)

#gdf_boundary = gpd.read_file('dataset/WARDS_2002/WARDS_2002.shp')



gdf = gpd.GeoDataFrame(temp_df, geometry='geometry')


# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# '1' 칼럼 데이터를 정규화하고 업데이트
gdf['1'] = scaler.fit_transform(gdf[['1']])

min_val = gdf['1'].min()
max_val = gdf['1'].max()

norm_min_val = 0.1  # 0으로 설정하지 않고 조금 상승시킵니다.
norm_max_val = 1.0
#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (norm_min_val, 'yellow'), (0.5, 'orange'), (norm_max_val, 'blue')], N=256)
#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.2, 'yellow'), (0.4, 'red'), (0.6, 'red'), (0.8, 'blue'), (1, 'blue')], N=512)

cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.03, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'),(0.4, 'blue'), (0.6, 'blue'), (0.8, 'blue'), (1, 'blue')], N=512)


fig, ax = plt.subplots(1, 1)


fig, ax = plt.subplots(1, 1)
gdf.plot(column='1',cmap=cmap, legend=False, ax=ax, markersize=100)


#gdf_boundary.boundary.plot(ax=ax, color='black', linewidth=0.5, ls='--')



ax.set_xticks([])  # x축 눈금을 없앱니다.
ax.set_yticks([])  # y축 눈금을 없앱니다.

# 테두리 제거
for spine in ax.spines.values():
    spine.set_visible(False)



df_polygon = pd.read_csv(r'kansas_base_unit_main_poly.csv', index_col=0)

# WKT를 geometry로 변환
df_polygon['geometry'] = df_polygon['0'].apply(wkt.loads)
gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry')

# 경계선을 그립니다


# 같은 축에 그리기
gdf_polygon.plot(ax=ax, color='none', edgecolor='black', label='gdf_polygon')

df_polygon1 = pd.read_csv(r'kansas_support_fixed_poly.csv', index_col=0)
df_polygon1['geometry'] = df_polygon1['0'].apply(wkt.loads)
gdf_polygon1 = gpd.GeoDataFrame(df_polygon1, geometry='geometry')

# 같은 축에 그리기
#gdf_polygon1.plot(ax=ax, color='gray', edgecolor='black',alpha=0.3)
#gdf.plot(ax=ax, facecolor='none', hatch='//', edgecolor='black')

gdf_polygon1.plot(ax=ax, color='gray', edgecolor='black', alpha=0.3, label='gdf_polygon1')

#plt.colorbar(ticks=[0.0, 0.2, 0.4, 1.0])
#ax.legend(loc='upper right')  # 범례를 상단 우측에 배치

#pc = gdf.plot(column='1', cmap=cmap, ax=ax, markersize=100)

# ScalarMappable 객체를 생성하고 colorbar에 전달
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(mappable=sm, ax=ax, shrink=0.5)


#%%

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

temp_df = pd.read_csv(r'central_point_target_0607.csv')
temp_df['geometry'] = temp_df['0'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(temp_df, geometry='geometry')

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# '1' 칼럼 데이터를 정규화하고 업데이트
gdf['1'] = scaler.fit_transform(gdf[['1']])
gdf['1'] = gdf['1'].clip(lower=0, upper=0.4)
#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'), (0.4, 'blue'), (0.6, 'blue'), (0.8, 'blue'), (1, 'blue')], N=512)
#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'), (0.4, 'lightblue'), (0.6, 'blue'), (0.8, 'blue'), (1, 'darkblue')], N=512)

#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'), (0.4, 'lightblue'), (0.6, 'blue'), (0.8, 'blue'), (1, 'darkblue')], N=512)

cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'), (0.4, '#add8e6'), (1, '#0000ff')], N=512)
fig, ax = plt.subplots(1, 1)
gdf.plot(column='1',cmap=cmap, legend=False, ax=ax, markersize=100)

ax.set_xticks([])  # x축 눈금을 없앱니다.
ax.set_yticks([])  # y축 눈금을 없앱니다.

# 테두리 제거
for spine in ax.spines.values():
    spine.set_visible(False)

df_polygon = pd.read_csv(r'kansas_base_unit_main_poly.csv', index_col=0)
df_polygon['geometry'] = df_polygon['0'].apply(wkt.loads)
gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry')

# 같은 축에 그리기
gdf_polygon.plot(ax=ax, color='none', edgecolor='black')

df_polygon1 = pd.read_csv(r'kansas_support_fixed_poly.csv', index_col=0)
df_polygon1['geometry'] = df_polygon1['0'].apply(wkt.loads)
gdf_polygon1 = gpd.GeoDataFrame(df_polygon1, geometry='geometry')

# 같은 축에 그리기
gdf_polygon1.plot(ax=ax, color='gray', edgecolor='black', alpha=0.3)

# ScalarMappable 객체를 생성하고 colorbar에 전달
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

# 우리가 원하는 범위 내에서 레이블을 생성
boundaries = [0, 0.1, 0.2, 0.3, 0.4]
labels = ['0', '0.1', '0.2', '0.3', '>=0.4']

# 새로운 colorbar를 만들고 레이블을 적용
cax = plt.gcf().add_axes([0.27, 0.125, 0.5, 0.025]) # 이 값들은 colorbar의 위치를 결정합니다. 적절히 조정해 주세요.
cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 0.4), orientation='horizontal')
cb.set_ticks(boundaries)
cb.set_ticklabels(labels)

plt.show()


#%% 최종


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
from shapely.geometry import Polygon
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

temp_df = pd.read_csv(r'central_point_target_0607.csv')
temp_df['geometry'] = temp_df['0'].apply(wkt.loads)

gdf = gpd.GeoDataFrame(temp_df, geometry='geometry')

# MinMaxScaler 객체 생성
scaler = MinMaxScaler()

# '1' 칼럼 데이터를 정규화하고 업데이트
gdf['1'] = scaler.fit_transform(gdf[['1']])
gdf['1'] = gdf['1'].clip(lower=0, upper=0.4)

#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, 'orange'), (0.2, 'red'), (0.3, 'purple'), (0.4, 'blue'), (0.6, 'blue'), (0.8, 'blue'), (1, 'blue')], N=512)


cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.2, 'orange'), (0.4, 'red'), (0.6, 'purple'), (0.8, 'blue'), (1, 'blue')], N=512)


#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.1, (1, 0.5, 0)), (0.2, 'red'), (0.3, (0.5, 0, 0.5)), (0.4, (0.5, 0.5, 1)), (0.6, 'blue'), (0.8, 'blue'), (1, 'blue')], N=512)


fig, ax = plt.subplots(1, 1)
gdf.plot(column='1',cmap=cmap, legend=False, ax=ax, markersize=20)

ax.set_xticks([])  # x축 눈금을 없앱니다.
ax.set_yticks([])  # y축 눈금을 없앱니다.

# 테두리 제거
for spine in ax.spines.values():
    spine.set_visible(False)

df_polygon = pd.read_csv(r'kansas_base_unit_main_poly_0809.csv', index_col=0)
df_polygon['geometry'] = df_polygon['0'].apply(wkt.loads)
gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry')

# 같은 축에 그리기
gdf_polygon.plot(ax=ax, color='none', edgecolor='black')

df_polygon1 = pd.read_csv(r'kansas_support_fixed_poly_0809.csv', index_col=0)
df_polygon1['geometry'] = df_polygon1['0'].apply(wkt.loads)
gdf_polygon1 = gpd.GeoDataFrame(df_polygon1, geometry='geometry')

# 같은 축에 그리기
gdf_polygon1.plot(ax=ax, color='gray', edgecolor='black', alpha=0.3)

# ScalarMappable 객체를 생성하고 colorbar에 전달
#sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))

# 우리가 원하는 범위 내에서 레이블을 생성
boundaries = [0, 0.1, 0.2, 0.3, 0.4]
labels = ['0', '0.1', '0.2','0.3','≥0.4']

# 새로운 colorbar를 만들고 레이블을 적용
cax = plt.gcf().add_axes([0.27, 0.125, 0.5, 0.025]) # 이 값들은 colorbar의 위치를 결정합니다. 적절히 조정해 주세요.
#cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation='horizontal', boundaries=boundaries)
cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 0.4), orientation='horizontal',)
#cb.set_label('>0.4 in blue')
cb.set_ticks(boundaries)
cb.set_ticklabels(labels)

plt.show()

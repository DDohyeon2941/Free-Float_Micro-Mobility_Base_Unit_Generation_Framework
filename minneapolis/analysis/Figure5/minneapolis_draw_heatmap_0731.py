# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:33:37 2023

@author: dohyeon
"""



# 최종

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import wkt
import matplotlib.colors as colors
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colorbar import ColorbarBase
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
temp_df = pd.read_csv(r'central_point_target_0701.csv')
temp_df['geometry'] = temp_df['0'].apply(wkt.loads)

gdf_boundary = gpd.read_file('dataset/WARDS_2002/WARDS_2002.shp')

gdf = gpd.GeoDataFrame(temp_df, geometry='geometry')

scaler = MinMaxScaler()
gdf['1'] = scaler.fit_transform(gdf[['1']])

#cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.1, 'yellow'), (0.2, 'orange'), (0.4, 'red'), (0.6, 'purple'), (0.8, 'blue'), (1, 'blue')], N=512)


cmap = colors.LinearSegmentedColormap.from_list('custom', [(0, 'white'), (0.01, 'yellow'), (0.2, 'orange'), (0.4, 'red'), (0.6, 'purple'), (0.8, 'blue'), (1, 'blue')], N=512)

fig, ax = plt.subplots(1, 1)
gdf.plot(column='1',cmap=cmap, legend=False, ax=ax, markersize=20)

ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

df_polygon = pd.read_csv(r'minneapolis_base_unit_main_poly_0809.csv', index_col=0)
df_polygon['geometry'] = df_polygon['0'].apply(wkt.loads)
gdf_polygon = gpd.GeoDataFrame(df_polygon, geometry='geometry')

gdf_polygon.plot(ax=ax, color='none', edgecolor='black')

df_polygon1 = pd.read_csv(r'minneapolis_support_fixed_poly_0809.csv', index_col=0)
df_polygon1['geometry'] = df_polygon1['0'].apply(wkt.loads)
gdf_polygon1 = gpd.GeoDataFrame(df_polygon1, geometry='geometry')

gdf_polygon1.plot(ax=ax, color='gray', edgecolor='black',alpha=0.3)

# 새로운 coloarbar를 만들고 레이블을 적용하는 부분을 추가
# 우리가 원하는 범위 내에서 레이블을 생성
boundaries = [0, 0.2, 0.4, 0.6, 0.8, 1]
labels = ['0', '0.2', '0.4', '0.6', '0.8', '1']

# 새로운 colorbar를 만들고 레이블을 적용
cax = plt.gcf().add_axes([0.27, 0.125, 0.5, 0.025]) # 이 값들은 colorbar의 위치를 결정합니다. 적절히 조정해 주세요.
#cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation='horizontal', boundaries=boundaries)
cb = ColorbarBase(cax, cmap=cmap, norm=Normalize(0, 1), orientation='horizontal')

#cb.set_label('>0.4 in blue')
cb.set_ticks(boundaries)
cb.set_ticklabels(labels)

plt.show()
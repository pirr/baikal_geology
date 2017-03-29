# -*- coding: utf-8 -*-

# Обеспечим совместимость с Python 2 и 3
# pip install future
from __future__ import (absolute_import, division, print_function, unicode_literals)

# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

# импортируем Pandas и Numpy
import pandas as pd
import numpy as np

import urllib2
from smb.SMBHandler import SMBHandler
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import DistanceMetric

director = urllib2.build_opener(SMBHandler)

fh = director.open(u'smb://192.168.3.57/4 квартал/приложения на МНЗ/39_Каталог проявлений УВ/Приложение_39.xls')
self_registry = pd.read_excel(fh, skiprows=17)
fh.close()
self_registry_xy = self_registry[['Unnamed: 6', 'Unnamed: 5']]
self_registry_xy_rad = self_registry_xy / 57.29578

fh = director.open(u'smb://192.168.3.78/Каталог УВ/окончательный_в_письмо/ТН_Iэтап_март.xlsx')
vsegei_registry = pd.read_excel(fh, skiprows=2)

fh.close()
vsegei_registry_xy = vsegei_registry[['y', 'x']]
vsegei_registry_xy = vsegei_registry_xy[vsegei_registry_xy.applymap(np.isreal).all(1)].astype(float)
vsegei_registry_id = vsegei_registry.loc[vsegei_registry_xy.index, u'Unnamed: 0'].as_matrix()
vsegei_registry_xy_rad = vsegei_registry_xy / 57.29578

# dist = pairwise_distances(self_registry_xy, vsegei_registry_xy, metric='euclidean')
# r = dist * 6372795

haversine = DistanceMetric.get_metric('haversine')
D = haversine.pairwise(self_registry_xy_rad, vsegei_registry_xy_rad)
r = D * 6372795
rx = np.where(r <= 50, r, np.NaN)

indxs = pd.DataFrame(np.argwhere(~np.isnan(rx)), columns=('self', 'vsegei'))
indxs = indxs.groupby('self')['vsegei'].apply(list).reset_index()
indxs['vsegei_join'] = indxs['vsegei'].apply(lambda gr: ','.join(str(vsegei_registry_id[x]) for x in gr))

for indx in indxs.as_matrix():
    self_registry.loc[indx[0], 'vsegei'] = indx[2]
#
# self_registry['vsegei_join'] = self_registry['vsegei'].applymap(','.join)
self_registry['vsegei'].to_csv('similarly.csv')

#
#
# from math import radians, cos, sin, asin, sqrt
#
# def _haversine(lon1, lat1, lon2, lat2):
#     """
#     Calculate the great circle distance between two points
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians
#     lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
#
#     # haversine formula
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a))
#     r = 6372795 # Radius of earth in kilometers. Use 3956 for miles
#     return c * r
#
# xy_1 = vsegei_registry_xy.as_matrix()
# xy_2 = self_registry_xy.as_matrix()
# test = _haversine(xy_1[1][1], xy_1[1][0], xy_2[0][1], xy_2[0][0])
#
#
# def _haversine_2(llong1, llat1, llong2, llat2):
#     import math
#
#     # pi - число pi, rad - радиус сферы (Земли)
#     rad = 6372795
#
#     # координаты двух точек
#        # llat1 = 77.1539
#     # llong1 = -120.398
#
#     # llat2 = 77.1804
#     # llong2 = 129.55
#
#     # в радианах
#     lat1 = llat1 * math.pi / 180.
#     lat2 = llat2 * math.pi / 180.
#     long1 = llong1 * math.pi / 180.
#     long2 = llong2 * math.pi / 180.
#
#     # косинусы и синусы широт и разницы долгот
#     cl1 = math.cos(lat1)
#     cl2 = math.cos(lat2)
#     sl1 = math.sin(lat1)
#     sl2 = math.sin(lat2)
#     delta = long2 - long1
#     cdelta = math.cos(delta)
#     sdelta = math.sin(delta)
#
#     # вычисления длины большого круга
#     y = math.sqrt(math.pow(cl2 * sdelta, 2) + math.pow(cl1 * sl2 - sl1 * cl2 * cdelta, 2))
#     x = sl1 * sl2 + cl1 * cl2 * cdelta
#     ad = math.atan2(y, x)
#     return ad * rad
#
# test_2 = _haversine_2(xy_1[0][0], xy_1[0][1], xy_2[0][0], xy_2[0][1])
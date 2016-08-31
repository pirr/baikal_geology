# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from anomaly_searcher import get_near

ice_data = pd.read_csv('big_data_290816.csv', sep=';')
ice_protocol = pd.read_csv('protocol_290816.csv', sep=';')
reestr_data = pd.read_csv('3108clusteriz_300_500_1000.csv', sep=';', encoding='cp1251')
gps_log_data = pd.read_excel('logger.xlsx')

yx_reestr = reestr_data[['y', 'x']].as_matrix() / 57.29578
yx_gps_log = gps_log_data[['Y_DD', 'X_DD']].as_matrix() / 57.29578

groups = ice_data.groupby('segment_num')

ice_nearanomaly_dist_dict = dict()
ice_near_gps_log_dict = dict()

for name, items in groups:
    yx_ice_gr = items[['y', 'x']].as_matrix() / 57.29578
    args = get_near(yx_ice_gr, yx_reestr)
    if args.size:
        ice_nearanomaly_dist_dict[name] = args
    middle_coord = yx_ice_gr[int(len(yx_ice_gr)/2)].reshape(1,-1)
    args_midd = get_near(middle_coord, yx_gps_log, 110)
    if args_midd.size:
        ice_near_gps_log_dict[name] = args_midd

ice_protocol['anomaly_num'] = np.nan


for ice_segm_num, anomalies_num in ice_nearanomaly_dist_dict.items():
    anomaly_gr = reestr_data.loc[anomalies_num, '№ проявления УВ (после кластеризации)  ']
    ice_protocol.loc[ice_protocol['segment_num']==ice_segm_num, 'anomaly_num'] = ','.join([str(x) for x in anomaly_gr])

ice_protocol['time'] = np.nan
temp = pd.DatetimeIndex(gps_log_data['date'])
gps_log_data['Date'] = temp.date
gps_log_data['Time'] = temp.time
del gps_log_data['DateTime']
for ice_segm_num, gps_log_row_num in ice_near_gps_log_dict.items():
    time = gps_log_data.loc[gps_log_row_num, 'Time'].values
    ice_protocol.loc[ice_protocol['segment_num']==ice_segm_num, 'time'] = time

ice_protocol.to_csv('ice_for_import.csv', sep=';')
    

    
    
    
    

    
   
    
    




# -*- coding: utf-8 -*-
'''
Input: grouping_anomalies clusteriz file, 
       ice_anomalies files (ice_for_import, big_data) after main.py,
       logger.xlsx - file with coordinates and time
Output: ice_protocol, reestr_with_ice
'''

import numpy as np
import pandas as pd
from anomaly_searcher import get_near

protocol_num = '2016-0-L'

ice_data = pd.read_csv('big_data_050916.csv', sep=';')
ice_for_import = pd.read_csv('ice_for_import.csv', sep=';')
reestr_data = pd.read_csv('0209clusteriz_100.csv', sep=';', encoding='cp1251')
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
        ice_near_gps_log_dict[name] = (args_midd, middle_coord[0][1]*57.29578, middle_coord[0][0]*57.29578)

ice_for_import['anomaly_num'] = np.nan

for ice_segm_num, anomalies_num in ice_nearanomaly_dist_dict.items():
    anomaly_gr = reestr_data.loc[anomalies_num, '№ проявления УВ (после кластеризации)  ']
    ice_for_import.loc[ice_for_import['segment_num']==ice_segm_num, 'anomaly_num'] = ','.join([str(x) for x in anomaly_gr])

ice_for_import['time'] = np.nan
ice_for_import['middle_x'] = np.nan
ice_for_import['middle_y'] = np.nan
temp = pd.DatetimeIndex(gps_log_data['date'])
gps_log_data['Date'] = temp.date
gps_log_data['Time'] = temp.time
for ice_segm_num, gps_log_row_num in ice_near_gps_log_dict.items():
    time = gps_log_data.loc[gps_log_row_num[0], 'Time'].values
    time_middle_coord = (time, gps_log_row_num[1], gps_log_row_num[2])
    ice_for_import.loc[ice_for_import['segment_num']==ice_segm_num, 'time':'middle_y'] = time_middle_coord

ice_for_import.sort_values(by=['date', 'time'], inplace=True)
ice_for_import.reset_index(inplace=True)
ice_for_import['n'] = np.nan
ice_for_import['num'] = np.nan
for index, row in ice_for_import.iterrows():
    n = str(index+1)
    num = '-'.join([protocol_num, n])
    ice_for_import.loc[index, 'n':'num'] = n, num

anom_prot_num_df = ice_for_import[['anomaly_num', 'n']].dropna()
anom_prot_num_df = anom_prot_num_df.apply(pd.to_numeric)
group_anom_prot_num = anom_prot_num_df.groupby('anomaly_num')

for anom_num, row in group_anom_prot_num:
    n = ','.join(str(x) for x in list(row['n']))
    ledomer_protocol = '-'.join([protocol_num, n])
    ledomer_reestr = reestr_data.loc[reestr_data['№ проявления УВ (после кластеризации)  ']==anom_num, 'Unnamed: 42'].iloc[0]
    if pd.isnull(ledomer_reestr):
        new_ledomer = ledomer_protocol
    else:
        new_ledomer = ', '.join([ledomer_reestr, ledomer_protocol])
    reestr_data.loc[reestr_data['№ проявления УВ (после кластеризации)  ']==anom_num, 'Unnamed: 42'] = new_ledomer

ice_for_import.to_csv(protocol_num+'_ice_protocol_test.csv', sep=';')
reestr_data.to_csv(protocol_num+'_reestr_with_ice_test.csv', sep=';')

   

    
    
    
    

    
   
    
    




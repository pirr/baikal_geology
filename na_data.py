import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
# from scipy import stats
import math
import re

uwb_logs_folder = '!UWB_logs'
data = pd.read_csv('resultgpsdel.csv', sep=';')

uwb_logs_files = (f for f in os.listdir(uwb_logs_folder) if f[-3:] == 'del')

# data = data[data['gpsfilename'] == '_17_ 09.04.2016 9_40_42.gps']
for f in uwb_logs_files:
    uwb_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
    uwb_log_df = uwb_log_df.ix[:, 0:3]
    uwb_log_df.columns = ['frame', 'min', 'max']
    gps_f = ''.join([f[:-3], 'gps'])
    uwb_log_df['gpsfilename'] = gps_f
    print(gps_f, len(data[data['gpsfilename'] == gps_f]), len(uwb_log_df))
    # data = pd.merge(data, uwb_log_df, how='outer', on=['gpsfilename', 'frame'])
    # print(gps_f,
    #       len(data[data['gpsfilename'] == gps_f]),
    #       len(data[(~pd.isnull(data['lon'])) & (data['gpsfilename'] == gps_f)])
    #       )


import os
import sys
import pandas as pd
from functools import partial
from multiprocessing import Pool

from anomaly_searcher import get_segments, merge_segments, join_coords


def multy_get_data(uwb_logs_folder, f):
    del_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
    del_log_df = del_log_df.ix[:, 0:3]
    del_log_df.columns = ['frame', 'min', 'max']
    f = f[:-4]
    del_log_df['filename'] = f
    gps_f = ''.join([f, '.gps'])
    del_log_df['min'] = del_log_df['min'].astype(
        str).str.replace(',', '.').astype(float)
    del_log_df['max'] = del_log_df['max'].astype(
        str).str.replace(',', '.').astype(float)
    del_log_df['thickness'] = del_log_df.apply(
        lambda row: round((row['max'] - row['min']) * 8.93561103810775, 0), axis=1)
    gps_log_df = pd.read_csv(os.path.join(uwb_logs_folder, gps_f), sep=' ')
    gps_log_df = gps_log_df.ix[:, 0:3]
    gps_log_df.columns = ['frame', 'x', 'y']
    gps_log_df['x'] = gps_log_df['x'].astype(
        str).str.replace(',', '.').astype(float)
    gps_log_df['y'] = gps_log_df['y'].astype(
        str).str.replace(',', '.').astype(float)
    del_log_df = join_coords(del_log_df, gps_log_df)
    del_log_df = del_log_df.dropna()

    return del_log_df[['frame', 'filename', 'thickness', 'x', 'y']]


def multy_get_anomaly(filegroup, limit=300, amplitude=20):
    name = filegroup[0]
    group = filegroup[1]
    anomaly_segments = get_segments(limit, amplitude, group)
    if not anomaly_segments:
        return None
    segments = merge_segments(anomaly_segments)
    return {'name': name, 'anomaly_segments': segments}


if __name__ == '__main__':

    uwb_logs_folder = 'testdata'
    del_logs_files = (f for f in os.listdir(
        uwb_logs_folder) if f[-4:] == '.del')
    with Pool(5) as pool:
        data = pool.map(
            partial(multy_get_data, uwb_logs_folder), del_logs_files)
        data = pd.concat(data)
        filegroups = data.groupby('filename')
        sys.stdout.write('\ndata prepared')
        sys.stdout.write('\nsearch segments')
        anomaly_segments = pool.map(multy_get_anomaly, filegroups)
        anomaly_segments = [s for s in anomaly_segments if s is not None]
    sys.stdout.write('\nanomalys prepared')

    jumps_dict = dict()
    if anomaly_segments:
        anomalys_df = pd.concat(anomaly_segments[0][1][0], ignore_index=True)
        jumps_dict[anomaly_segments[0][0]] = anomaly_segments[0][1][1]
    else:
        print('None anomalys')
        sys.exit()

    for name, anomalys in anomaly_segments[1:]:
        jumps_dict[name] = anomalys[1]
        anomalys = pd.concat(anomalys[0], ignore_index=True)
        anomalys_df = pd.concat([anomalys_df, anomalys], ignore_index=True)

    anomalys_df.to_csv('ice_anomalies_120816_1335.csv', sep=';')
    sys.stdout.write('\nDONE')

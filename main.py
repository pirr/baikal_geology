
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
    f = f[:-3]
    del_log_df['filename'] = f
    gps_f = ''.join([f, 'gps'])
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
    return name, merge_segments(anomaly_segments)


if __name__ == '__main__':

    uwb_logs_folder = 'testdata'
    del_logs_files = (f for f in os.listdir(
        uwb_logs_folder) if f[-3:] == 'del')
    with Pool(5) as pool:
        data = pool.map(
            partial(multy_get_data, uwb_logs_folder), del_logs_files)
        data = pd.concat(data)
        filegroups = data.groupby('filename')
        sys.stdout.write('\ndata prepared')
        sys.stdout.write('\nsearch segments')
    sys.stdout.write('\nanomalys prepared')


import os
import sys
import pandas as pd
from multiprocessing import Pool

from anomaly_searcher import get_segments, merge_segments, join_coords


def multy_get_data(f, uwb_logs_folder='testdata'):
    del_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
    del_log_df = del_log_df.ix[:, 0:3]
    del_log_df.columns = ['frame', 'min', 'max']
    f = f[:-3]
    del_log_df['filename'] = f
    gps_f = ''.join([f, 'gps'])
    # del_log_df['gpsfilename'] = gps_f
    del_log_df['min'] = del_log_df['min'].str.replace(',', '.').astype(float)
    del_log_df['max'] = del_log_df['max'].str.replace(',', '.').astype(float)
    del_log_df['thickness'] = del_log_df.apply(
        lambda row: (row['max'] - row['min']) * 8.93561103810775, axis=1)
    gps_log_df = pd.read_csv(os.path.join(uwb_logs_folder, gps_f), sep=' ')
    gps_log_df = gps_log_df.ix[:, 0:3]
    gps_log_df.columns = ['frame', 'x', 'y']
    gps_log_df['x'] = gps_log_df['x'].str.replace(',', '.').astype(float)
    gps_log_df['y'] = gps_log_df['y'].str.replace(',', '.').astype(float)
    del_log_df = join_coords(del_log_df, gps_log_df)
    del_log_df = del_log_df.dropna()

    return del_log_df


def multy_segment_looker(filegroup, limit=300, amplitude=20, segments_dict=dict()):
    name = filegroup[0]
    group = filegroup[1]
    segments_dict[name] = get_segments(limit, amplitude, group[:3000])
    segments_dict[name], segments_dict[
        'jumps'] = merge_segments(segments_dict[name])
    return segments_dict


if __name__ == '__main__':

    pool = Pool()
    uwb_logs_folder = 'testdata'
    del_logs_files = (f for f in os.listdir(
        uwb_logs_folder) if f[-3:] == 'del')
    data = pool.map(multy_get_data, del_logs_files)
    data = pd.concat(data)
    filegroups = data.groupby('filename')
    sd = pool.map(multy_segment_looker, filegroups)
        sys.stdout.write('\ndata prepared')
        sys.stdout.write('\nsearch segments')
    sys.stdout.write('\nanomalys prepared')

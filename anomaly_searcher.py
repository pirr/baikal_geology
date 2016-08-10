# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
import sys
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
import matplotlib.pyplot as plt
# from scipy import stats
import math
import re


def get_dist(llat1, llong1, llat2, llong2):
    rad = 6372795

    lat1 = llat1 * math.pi / 180.
    lat2 = llat2 * math.pi / 180.
    long1 = llong1 * math.pi / 180.
    long2 = llong2 * math.pi / 180.

    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    y = math.sqrt(math.pow(cl2 * sdelta, 2) +
                  math.pow(cl1 * sl2 - sl1 * cl2 * cdelta, 2))
    x = sl1 * sl2 + cl1 * cl2 * cdelta
    ad = math.atan2(y, x)
    dist = ad * rad

    return dist


def get_deduct(group, si, count):
    deduct = 0
    for k in range(si, count):
        deduct += (group['thickness'].iloc[k] -
                   group['thickness'].iloc[k - 1])
    return deduct


def max_amplitude(group, si, count):
    min_val = group['thickness'].iloc[si:count].min()
    max_val = group['thickness'].iloc[si:count].max()
    return max_val - min_val


def get_segments(limit, group, amplitude):
    end = len(group)
    fin = end - 1
    segments = []
    si = 0
    start_anomaly = None
    deduct = None
    frames = list(range(1, end))
    while frames:
        count = frames[0]

        deduct = get_deduct(group, si, count)
        sys.stdout.write(
            'processing search segments... {3}:{0}/{1} found:{2} || {4}, {5}\r'
            .format(count, fin, len(segments), si, start_anomaly, deduct))

        if np.fabs(deduct) >= amplitude:
            if start_anomaly is not None:
                if start_anomaly[0] / deduct < 0:
                    segments.append(group.iloc[start_anomaly[1]:count])
                    start_anomaly = None
                    si = count
            else:
                start_anomaly = [deduct, si]

        elif count is fin:
            if max_amplitude(group, si, count) >= amplitude:
                segments.append(group.iloc[si:count])

        if (count - si) > limit:
            si = count

        frames.pop(0)
    return segments


def merge_segments(segments):
    r = list(range(1, len(segments)))
    jumps = [1] * len(segments)
    while r:
        sys.stdout.write('processing merging segments... {}\r'.format(len(r)))
        i = r[0]
        if segments[i - 1].iloc[-1]['frame'] + 1 == segments[i].iloc[0]['frame']:
            segments[i - 1] = pd.concat([segments[i - 1], segments.pop(i)])
            jumps[i - 1] += 1
            jumps.pop(i)
            r.pop()
        else:
            r.pop(0)

    return segments, jumps

# data = pd.read_csv('testdata/resultgpsdel.csv', sep=';')
# filegroups = data.groupby('gpsfilename')
filegroups = dict()
uwb_logs_folder = '!UWB_logs'
uwb_logs_files = (f for f in os.listdir(uwb_logs_folder) if f[-3:] == 'del')

for f in uwb_logs_files:
    uwb_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
    uwb_log_df = uwb_log_df.ix[:, 0:3]
    uwb_log_df.columns = ['frame', 'min', 'max']
    gps_f = ''.join([f[:-3], 'gps'])
    uwb_log_df['gpsfilename'] = gps_f
    uwb_log_df['min'] = uwb_log_df['min'].str.replace(',', '.').astype(float)
    uwb_log_df['max'] = uwb_log_df['max'].str.replace(',', '.').astype(float)
    uwb_log_df['thickness'] = uwb_log_df.apply(
        lambda row: (row['max'] - row['min']) * 8.93561103810775, axis=1)
    filegroups[gps_f] = uwb_log_df


#     print(gps_f, len(data[data['gpsfilename'] == gps_f]), len(uwb_log_df))
# data = pd.merge(data, uwb_log_df, how='outer', on=['gpsfilename', 'frame'])
# print(gps_f,
#       len(data[data['gpsfilename'] == gps_f]),
#       len(data[(~pd.isnull(data['lon'])) & (data['gpsfilename'] == gps_f)])
#       )


segments_dict = dict()
jumps_dict = dict()
limit = 300
amplitude = 20
for name, group in filegroups.items():
    sys.stdout.write(name + ':\n')
    segments_dict[name] = get_segments(limit, group, amplitude)
    segments_dict[name], jumps_dict[name] = merge_segments(segments_dict[name])
    sys.stdout.write("\033[K")
    sys.stdout.write(' ' * 10 + 'found {} segments in {} frames\n'.format(
        len(segments_dict[name]), len(group)))
    # sys.stdout.write("\033[F")
    # sys.stdout.write("\033[K")

    # sys.stdout.flush()
    # print('\n')

# segments_dict = {k: v for k, v in segments_dict.items() if len(v) > 4}
# jumps_dict = dict()
# merging_segments_dict = dict()
# for name, segments in segments_dict.items():
#     merging_segments_dict[name] = merge_segments(segments)
#     plt.plot(merging_segments_dict[name][0]['thickness'])
#     plt.axis([merging_segments_dict[name][0]['frame'].min(),
#               merging_segments_dict[name][0]['frame'].max(),
#               merging_segments_dict[name][0]['thickness'].min(),
#               merging_segments_dict[name][0]['thickness'].max()])

draft_reestr = []
for name, segments in segments_dict.items():
    date = re.search(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}', name).group(0)
    for k, segment in enumerate(segments):
        # seg_dist = sum(get_dist(
        #     segment['lat'].iloc[i - 1],
        #     segment['lon'].iloc[i - 1],
        #     segment['lat'].iloc[i],
        #     segment['lon'].iloc[i])
        #     for i in range(1, len(segment)))
        min_val = np.fabs(segment['thickness'].min())
        max_val = np.fabs(segment['thickness'].max())
        draft_reestr.append([
            segment['frame'].iloc[0],
            segment['frame'].iloc[-1],
            date, np.nan, 'прф',
            np.nan, min_val, max_val,
            max_val - min_val, np.median(segment['thickness']),
            jumps_dict[name][k], np.nan,
            name, 'gps, dat'
        ])


df_protocol = pd.DataFrame(draft_reestr,
                           columns=[
                               'frame_st', 'frame_end',
                               'date', 'time', 'type',
                               'Z', 'tk_min', 'tk_max',
                               'amplitude', 'middle',
                               'jumps', 'other',
                               'file_name', 'file_source'
                           ])

plt.show()
# df_station = pd.read_csv('Station_summer_2016.csv', sep=';', encoding='utf8')
# df_anomaly = pd.read_csv('увеличенная_таблица.csv', sep=';', encoding='utf8')

# df_reestr = pd.read_excel(
#     'Реестр проявлений углеводородов_2807_автогруппировка.xlsx')
# cols = [c for c in df_reestr.columns[:5]]
# cols.extend(df_reestr.columns[13:])
# reestr = df_reestr[7:][cols]
# coords = reestr[['y', 'x']].as_matrix()


# df_anomaly['x,N,16,6'] = df_anomaly[
#     'x,N,16,6'].str.replace(',', '.').astype(float)
# df_anomaly['y,N,16,6'] = df_anomaly[
#     'y,N,16,6'].str.replace(',', '.').astype(float)
# df_station['LAT,N,16,12'] = df_station[
#     'LAT,N,16,12'].str.replace(',', '.').astype(float)
# df_station['LONG,N,16,13'] = df_station[
#     'LONG,N,16,13'].str.replace(',', '.').astype(float)


# station_YX = df_station[['LAT,N,16,12', 'LONG,N,16,13']].as_matrix()
# anomaly_YX = df_anomaly[['y,N,16,6', 'x,N,16,6']].as_matrix()
# protocol_XY = df_protocol['coords']

# total = len(protocol_XY)
# for ir, crs in enumerate(protocol_XY):
#     min_dist_stat = None
#     min_dist_anom = None

#     print('\rProcessing search nearest anomalys, station... {}/{}'.format(ir, total))

#     for cr in crs:

#         for ist, cst in enumerate(station_YX):
#             dist = get_dist(cr[1], cr[0], cst[0], cst[1])
#             if min_dist_stat is None or dist < min_dist_stat[0]:
#                 min_dist_stat = [dist, ist]

#         if min_dist_stat[0] < 2000:
#             df_protocol['station'].iloc[ir] = df_station[
#                 'STATION,C,5'].iloc[min_dist_stat[1]]

#         for ian, can in enumerate(anomaly_YX):
#             dist = get_dist(cr[1], cr[0], can[0], can[1])
#             if min_dist_anom is None or dist < min_dist_anom[0]:
#                 min_dist_anom = [dist, ian]

#         if min_dist_anom[0] < 2000:
#             df_protocol['other'].iloc[ir] = df_anomaly[
#                 'имя,C,254'].iloc[min_dist_anom[1]]

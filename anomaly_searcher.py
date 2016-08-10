import sys
import os
import numpy as np
import pandas as pd
import math
import re


def get_deduct(group, startframe, endframe):
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
    for k in range(startframe, endframe):
        deduct += (group['thickness'].iloc[k] -
                   group['thickness'].iloc[k - 1])
    return deduct


def max_amplitude(group, startframe, endframe):
    min_val = group['thickness'].iloc[startframe:endframe].min()
    max_val = group['thickness'].iloc[startframe:endframe].max()
    return max_val - min_val


def get_segments(limit, group, amplitude):
    end = len(group)
    fin = end - 1
    segments = []
    startframe = 0
    start_anomaly = None
    deduct = None
    frames = list(range(1, end))
    while frames:
        endframe = frames[0]

        deduct = get_deduct(group, startframe, endframe)
        sys.stdout.write(
            'processtartframeng search segments... {3}:{0}/{1} found:{2} || {4}, {5}\r'
            .format(endframe, fin, len(segments), startframe, start_anomaly, deduct))

        if np.fabs(deduct) >= amplitude:
            if start_anomaly is not None:
                if start_anomaly[0] / deduct < 0:
                    segments.append(group.iloc[start_anomaly[1]:endframe])
                    start_anomaly = None
                    startframe = endframe
            else:
                start_anomaly = [deduct, startframe]

        elif endframe is fin:
            if max_amplitude(group, startframe, endframe) >= amplitude:
                segments.append(group.iloc[startframe:endframe])

        if (endframe - startframe) > limit:
            startframe = endframe

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

draft_reestr = []
for name, segments in segments_dict.items():
    date = re.search(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}', name).group(0)
    for k, segment in enumerate(segments):
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


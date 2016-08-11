import sys
import os
import numpy as np
import pandas as pd
import re


def get_deduct(group, startframe, endframe):
    deduct = 0
    for k in range(startframe, endframe):
        deduct += (group['thickness'].iloc[k] -
                   group['thickness'].iloc[k - 1])
    return deduct


def max_amplitude(group, startframe, endframe):
    min_val = group['thickness'].iloc[startframe:endframe].min()
    max_val = group['thickness'].iloc[startframe:endframe].max()
    return np.fabs(max_val - min_val)


def get_segments(limit, amplitude, group):
    end = len(group)
    fin = end - 1
    segments = []
    startframe = 0
    start_anomaly = None
    deduct = None
    for endframe in list(range(1, end)):
        deduct = get_deduct(group, startframe, endframe)
        sys.stdout.write("\033[K")
        sys.stdout.write(
            'process search segments... {3:6d}:{0:6d}/{1:6d} found:{2:3d} || {4}, {5:3d}\r'
            .format(endframe, fin, len(segments),
                    startframe, start_anomaly, int(deduct)))

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


def join_coords(del_df, gps_df):
    del_df = pd.merge(del_df, gps_df, how='outer', on=['frame'])
    del_df = del_df.interpolate(method='linear', limit_direction='both')
    return del_df


if name == '__main__':
    filegroups = dict()
    uwb_logs_folder = 'testdata'
    del_logs_files = (f for f in os.listdir(uwb_logs_folder) if f[-3:] == 'del')
    gps_logs_files = (f for f in os.listdir(uwb_logs_folder) if f[-3:] == 'gps')

    for f in del_logs_files:
        del_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
        del_log_df = del_log_df.ix[:, 0:3]
        del_log_df.columns = ['frame', 'min', 'max']
        f = f[:-3]
        del_log_df['filename'] = f
        gps_f = ''.join([f, 'gps'])
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

        filegroups[gps_f] = del_log_df


    segments_dict = dict()
    jumps_dict = dict()
    limit = 300
    amplitude = 20
    for name, group in filegroups.items():
        sys.stdout.write(name + ':\n')
        segments_dict[name] = get_segments(limit, amplitude, group[:3000])
        segments_dict[name], jumps_dict[name] = merge_segments(segments_dict[name])
        sys.stdout.write("\033[K")
        sys.stdout.write(' ' * 10 + 'found {} segments in {} frames\n'.format(
            len(segments_dict[name]), len(group)))

    concat_segments = pd.concat([pd.concat(s) for s in segments_dict.values() if s],
                                ignore_index=True)

    concat_segments.to_csv('coords.csv', sep=';')

    # draft_reestr = []
    # for name, segments in segments_dict.items():
    #     date = re.search(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}', name).group(0)
    #     for k, segment in enumerate(segments):
    #         min_val = np.fabs(segment['thickness'].min())
    #         max_val = np.fabs(segment['thickness'].max())
    #         draft_reestr.append([
    #             segment['frame'].iloc[0],
    #             segment['frame'].iloc[-1],
    #             date, np.nan, 'прф',
    #             np.nan, min_val, max_val,
    #             max_val - min_val, np.median(segment['thickness']),
    #             jumps_dict[name][k], np.nan,
    #             name, 'gps, dat'
    #         ])

    # df_protocol = pd.DataFrame(draft_reestr,
    #                            columns=[
    #                                'frame_st', 'frame_end',
    #                                'date', 'time', 'type',
    #                                'Z', 'tk_min', 'tk_max',
    #                                'amplitude', 'middle',
    #                                'jumps', 'other',
    #                                'file_name', 'file_source'
    #                            ])

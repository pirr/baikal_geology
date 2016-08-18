import sys
import numpy as np
import pandas as pd
from random import randint


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
            '\r{}'.format('.' * randint(0, 9)))

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
        i = r[0]
        if segments[i - 1].iloc[-1]['frame'] + 1 == segments[i].iloc[0]['frame']:
            segments[i - 1] = pd.concat([segments[i - 1], segments.pop(i)])
            jumps[i - 1] += 1
            jumps.pop(i)
            r.pop()
        else:
            r.pop(0)

    return zip(segments, jumps)


def join_coords(del_df, gps_df):
    del_df = pd.merge(del_df, gps_df, how='outer', on=['frame'])
    del_df = del_df.interpolate(method='linear', limit_direction='both')
    return del_df





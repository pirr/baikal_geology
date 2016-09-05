
import os
import sys
import pandas as pd
from functools import partial
from multiprocessing import Pool
import numpy as np
import re
from osgeo import ogr
from datetime import datetime
import matplotlib.pyplot as plt


from anomaly_searcher import (get_segments, merge_segments,
                              join_coords, yx_from_geom, get_near_stations,
                              get_segment_len, get_chunk_segment)


start = datetime.now()

def multy_get_data(uwb_logs_folder, f):
    filename = f[:-4]
    gps_name = ''.join([filename, '.gps'])
    gps_f = os.path.join(uwb_logs_folder, gps_name)

    if os.path.exists(gps_f):
        del_log_df = pd.read_csv(os.path.join(uwb_logs_folder, f), sep=' ')
        del_log_df = del_log_df.ix[:, 0:3]
        del_log_df.columns = ['frame', 'min', 'max']
        del_log_df['filename'] = filename
        del_log_df['min'] = del_log_df['min'].astype(
            str).str.replace(',', '.').astype(float)
        del_log_df['max'] = del_log_df['max'].astype(
            str).str.replace(',', '.').astype(float)
        del_log_df['thickness'] = del_log_df.apply(
            lambda row: round((row['max'] - row['min']) * 8.93561103810775, 0), axis=1)
        gps_log_df = pd.read_csv(gps_f, sep=' ')
        gps_log_df = gps_log_df.ix[:, 0:3]
        gps_log_df.columns = ['frame', 'x', 'y']
        gps_log_df['x'] = gps_log_df['x'].astype(
            str).str.replace(',', '.').astype(float)
        gps_log_df['y'] = gps_log_df['y'].astype(
            str).str.replace(',', '.').astype(float)
        del_log_df = join_coords(del_log_df, gps_log_df)
        del_log_df = del_log_df.dropna()

    else:
        return None

    return del_log_df[['frame', 'filename', 'thickness', 'x', 'y']]


def multy_get_anomaly(filegroup, limit=500, amplitude=20):
    name = filegroup[0]
    group = filegroup[1]
    sys.stdout.write('{} search anomalies\n'.format(name))
    anomaly_segments = get_segments(limit, amplitude, group)
    if not anomaly_segments:
        sys.stdout.write('{} no anomalies\n'.format(name))
        return None
    merge_segms = []
    for segment in merge_segments(anomaly_segments):
        segment[0][['Y', 'X']] = segment[0][['y', 'x']].apply(
            lambda row: row / 57.29578).apply(pd.Series)
        merge_segms.append(segment)
    return {name: merge_segms}

if __name__ == '__main__':
    uwb_logs_folder = 'testdata'

    del_logs_files = (f for f in os.listdir(
        uwb_logs_folder) if f[-4:] == '.del')

    with Pool(5) as pool:
        data = pool.map(
            partial(multy_get_data, uwb_logs_folder), del_logs_files)
        data = [d for d in data if d is not None]
        data = pd.concat(data)
        filegroups = data.groupby('filename')
        sys.stdout.write('data prepared/n')
        anomaly_segments = pool.map(multy_get_anomaly, filegroups)
        anomaly_segments = [s for s in anomaly_segments if s is not None]
    sys.stdout.write('\nanomalys prepared')

    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_file_stations = 'gis/Station_summer_2016.shp'
    shp_file_detected_anomalies = 'gis/увеличенная_таблица.shp'
    stations = driver.Open(shp_file_stations, 0)
    daLayer_stats = stations.GetLayer(0)

    YX_stations = []
    station_dict = dict()
    for i, feature in enumerate(daLayer_stats):
        YX_stations.append(yx_from_geom(feature))
        station_dict[i] = feature.GetField('STATION')
    YX_stations = np.array(YX_stations) / 57.29578

    big_data_segments = pd.DataFrame()
    anomalys_list = []
    sgm_num = 1
    for named_segments in anomaly_segments:
        for name, segments in named_segments.items():
            date = re.search(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}',
                             name).group(0)

            for segment in segments:
                anomaly_dict = dict()
                min_val = np.fabs(segment[0]['thickness'].min())
                max_val = np.fabs(segment[0]['thickness'].max())

                anomaly_dict['segment_num'] = sgm_num
                anomaly_dict['file'] = name
                anomaly_dict['x_start'] = segment[0]['x'].iloc[0]
                anomaly_dict['y_start'] = segment[0]['y'].iloc[0]
                anomaly_dict['x_end'] = segment[0]['x'].iloc[-1]
                anomaly_dict['y_end'] = segment[0]['y'].iloc[-1]
                anomaly_dict['frame_start'] = segment[0]['frame'].iloc[0]
                anomaly_dict['frame_end'] = segment[0]['frame'].iloc[-1]
                anomaly_dict['jumps'] = segment[1]

                anomaly_dict['near_stations'] = []
                anomaly_dict['len'] = 0
                for chunk_XY_segment in get_chunk_segment(segment[0]):
                    anomaly_dict['near_stations'].append(get_near_stations(chunk_XY_segment[['Y', 'X']],
                                                                           YX_stations, station_dict))
                    anomaly_dict[
                        'len'] += get_segment_len(chunk_XY_segment[['Y', 'X']])

                anomaly_dict['near_stations'] = [
                    ns for ns in anomaly_dict['near_stations'] if ns is not None]
                if anomaly_dict['near_stations']:
                    anomaly_dict['near_stations'] = ', '.join(
                        str(s) for s in set.union(*anomaly_dict['near_stations']))
                else:
                    anomaly_dict['near_stations'] = np.nan
                anomaly_dict['min'] = min_val
                anomaly_dict['max'] = max_val
                anomaly_dict['date'] = date
                anomaly_dict['amplitude'] = max_val - min_val
                anomaly_dict['median'] = np.median(segment[0]['thickness'])
                segment[0]['segment_num'] = sgm_num
                big_data_segments = pd.concat([big_data_segments, segment[0]])

                sgm_num += 1

                anomalys_list.append(anomaly_dict)
    
    for name, group in big_data_segments.groupby('segment_num'):
        group['thickness'] = group['thickness'] * -1
        ax = group.plot(x=['frame'], y=['thickness'], title=str(name) + ' ' + group.iloc[0]['filename'])
        fig = ax.get_figure()
        fig.savefig('figs/' + str(name) + '.png')
    
    anomalys_df = pd.DataFrame(anomalys_list)
    anomalys_df.to_csv('ice_for_import.csv', sep=';')
    big_data_segments.to_csv('big_data_050916.csv', sep=';')
    sys.stdout.write('\nDONE in {}'.format(datetime.now() - start))

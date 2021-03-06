import ogr
import numpy as np
import networkx as nx
import pandas as pd
import json
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import DistanceMetric
from scipy.ndimage.interpolation import shift
from itertools import combinations

def get_deduct(group, startframe, endframe):
    M1 = group['thickness'].iloc[startframe: endframe]
    M2 = shift(M1, -1, cval=np.nan)
    deduct = (M2[:-1] - M1[:-1]).sum()
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
    previous_deduct = 0
    start_anomaly = None
    deduct = None
    for endframe in list(range(1, end)):
        deduct = get_deduct(group, startframe, endframe)
        
        if start_anomaly is None:
            if deduct <= amplitude * -1:
                start_anomaly = [deduct, startframe]
                startframe = endframe
#            elif previous_deduct < deduct:
#                startframe = endframe
            elif (endframe - startframe) > limit:
                startframe = endframe
        elif endframe is fin:
            if max_amplitude(group, startframe, endframe) >= amplitude:
                segments.append(group.iloc[startframe:endframe])
        
        else:
            if deduct >= amplitude:
                segments.append(group.iloc[start_anomaly[1]:endframe])
                start_anomaly = None
                startframe = endframe
            elif start_anomaly[0] >= deduct: 
                start_anomaly = [deduct, startframe]
            elif (endframe - startframe) > limit:
                startframe = endframe

        previous_deduct = deduct                        
        
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


def get_point(yx):
    point = ogr.Geometry(ogr.wkbPoint)
    return point.AddPoint(yx[0], yx[1])


def near_point(geom_p1, geom_p2, common_name, max_near_dist=100, metric=6372795):
    dist = geom_p1.Distance(geom_p2) / 57.29578 * metric
    return common_name if dist <= max_near_dist else False


def yx_from_geom(feature):
    return json.loads(feature.geometry().ExportToJson())['coordinates'][::-1]


def get_near(YX1, YX2, max_dist=100):
    D = pairwise_distances(YX1, YX2) * 6372795
    args = np.argwhere((D == D.min()) & (D < max_dist))[:, 1]
    return args

def get_near_stations(YX_anomalies, YX_stations, station_dict, max_dist_to_station=300):
    args = get_near(YX_anomalies, YX_stations, max_dist_to_station)    
    near_stations = None
    if args.size:
        near_stations = {station_dict[i] for i in args}
    return near_stations


def get_segment_len(YX_segment):
    dist = DistanceMetric.get_metric(metric='haversine')
    D = dist.pairwise(YX_segment)
    segment_len = np.sum(D.diagonal(1) * 6372795)
    return segment_len


def get_chunk_segment(YX_segment, size=5000):
    for i in range(0, len(YX_segment), size):
        yield YX_segment.iloc[i:i + size]

def get_min_weight_max_cliques(G_with_weight):
    max_weight_cliques = set()
    min_weight_cliques = set()    
    weight_dict = {}
    cliques = sorted(list(nx.find_cliques(G_with_weight)), key=len)[::-1]
    clique_combos = list(combinations(cliques, 2))
    
    
    for i, clique in enumerate(cliques):
        combos = combinations(sorted(clique), 2)
        weight_dict[i] = sum(G_with_weight[c[0]][c[1]]['weight'] for c in combos)    
    
    for clique_comb in clique_combos:
        i_1 = cliques.index(clique_comb[0])
        i_2 = cliques.index(clique_comb[1])
        
        if not {i_1, i_2} & max_weight_cliques:
            u = set(clique_comb[0]) & set(clique_comb[1])
            if u:
                if len(clique_comb[0]) == len(clique_comb[1]):
                    p1 = weight_dict[i_1]
                    p2 = weight_dict[i_2]
                    if p1 <= p2:
                        max_weight_cliques.add(i_2)
                        min_weight_cliques.add(i_1)
                    else:
                        max_weight_cliques.add(i_1)
                        min_weight_cliques.add(i_2)
                
                elif len(clique_comb[0]) > len(clique_comb[1]):
                    max_weight_cliques.add(i_2)
                    min_weight_cliques.add(i_1)
                
                else:
                    max_weight_cliques.add(i_1)
                    min_weight_cliques.add(i_2)
            
            else:
                min_weight_cliques.add(i_1)
                min_weight_cliques.add(i_2)
        
        elif i_1 in max_weight_cliques and i_2 not in max_weight_cliques:
            min_weight_cliques.add(i_2)
        
        elif i_2 in max_weight_cliques and i_1 not in max_weight_cliques:
            min_weight_cliques.add(i_1)
    
    min_weight_max_cliques_index = min_weight_cliques - max_weight_cliques
    min_weight_max_cliques = [x for i, x in enumerate(cliques) if i in min_weight_max_cliques_index]
            
    return min_weight_max_cliques
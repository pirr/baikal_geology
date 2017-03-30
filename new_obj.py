import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric
from anomaly_searcher import get_min_weight_max_cliques
import networkx as nx


max_dis = 50
OBJS_COLUMNS = ['x', 'y']


class NewObjException(Exception):
    pass


def get_objs_from_csv(csv_file, sep=';'):
    objs = pd.read_csv(csv_file, sep=sep)
    miss_cols = [col for col in OBJS_COLUMNS if col not in objs.columns]
    if miss_cols:
        raise NewObjException(u'Not enough columns:{}'.format(','.join(miss_cols)))
    return objs


def get_dist_between_objs(*args, **kwargs):
    to_rad = kwargs.get('to_rad', True)
    rad = 57.29578
    dm = DistanceMetric.get_metric(metric='haversine')
    objs_yx_list = []
    for objs in args:
        objs_yx = objs[['y', 'x']].as_matrix()
        if to_rad:
            objs_yx /= rad
        objs_yx_list.append(objs_yx)
    return dm.pairwise(*objs_yx_list)

def get_groups(dist_matrix, add_to_index_1=0, add_to_index_2=0):

    for i in range(len(dist_matrix)):
        dist_matrix[i, :i + 1] = np.nan

    R = dist_matrix * 6372795

    RX = np.where(R <= max_dis, R, np.NAN)
    true_indexes = np.argwhere(~np.isnan(RX))

    # _true_indexes = np.copy(true_indexes)
    # _true_indexes[:, 0] = _true_indexes[:, 0] + add_to_index_1
    # _true_indexes[:, 1] = _true_indexes[:, 1] + add_to_index_2

    nodes = np.union1d(true_indexes[:, 0] + add_to_index_1, true_indexes[:, 1] + add_to_index_2)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for index in true_indexes:
        G.add_edge(index[0] + add_to_index_1, index[1] + add_to_index_2, weight=RX[index[0], index[1]])

    min_weight_max_cliques = [cliq for cliq in get_min_weight_max_cliques(G) if len(cliq)>1]

    return min_weight_max_cliques, list(nx.connected_components(G))


if __name__ == '__main__':

    objs_1 = get_objs_from_csv(u'testdata//objs_1.csv')
    objs_2 = get_objs_from_csv(u'testdata//objs_2.csv')

    group_num = objs_1['group'].max(skipna=True)
    D = get_dist_between_objs(objs_1, objs_2)
    min_weight_max_cliques_1 = get_groups(D, add_to_index_1=max(objs_2.index.values)+1)

    d = get_dist_between_objs(objs_1)
    min_weight_max_cliques_2 = get_groups(d)

    for gr in min_weight_max_cliques_2[0]:
        groups = objs_1.loc[gr, 'group']
        gr_num = groups.min(skipna=True)

        if pd.isnull(gr_num):
            group_num += 1
            gr_num = group_num

        objs_1.loc[objs_1['group'].isin(groups), 'group'] = gr_num

    objs_1.index += max(objs_2.index.values) + 1
    concat_objs = pd.concat([objs_2, objs_1])

    for gr in min_weight_max_cliques_1[0]:
        conc_groups = concat_objs.loc[gr, 'group'].dropna()
        conc_groups_val = [g for g in conc_groups if not pd.isnull(g)]
        gr_num = group_num + 1
        for gr in conc_groups_val:
            if pd.isnull(gr):
                continue
            if gr_num > gr:
                gr_num = gr

        concat_objs.loc[concat_objs['group'].isin(conc_groups_val), 'group'] = gr_num

    concat_objs.to_csv('vsegei_new_groups.csv', sep=';')










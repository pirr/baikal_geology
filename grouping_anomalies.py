import pandas as pd
from sklearn.neighbors import DistanceMetric
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from osgeo import ogr
# from matplotlib.ticker import ScalarFormatter


def _get_point(x, y):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    return point

distm = DistanceMetric.get_metric(metric='haversine')

df = pd.read_excel('d:\work\BAIKAL\Geology\каталог\\0209 Реестр проявлений углеводородов.xlsx', 
                   sheetname='реестр', skiprows=2)
cols = [c for c in df.columns[:11]]
cols.extend(df.columns[19:])
reestr = df[11:][cols]

coords = reestr[['y', 'x']].as_matrix()
yx = coords / 57.29578
D = distm.pairwise(yx)

for i in range(len(D)):
    D[i, :i + 1] = np.nan

r = D * 6372795
dists = [100]

for dis in dists:
    rx = np.where(r <= dis, r, np.NAN)
    true_indexes = np.argwhere(~np.isnan(rx))
#    dist = rx[~np.isnan(rx)]
    
    nodes = np.union1d(true_indexes[:, 0], true_indexes[:, 1])
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(true_indexes)
    T = G.copy()
    B = G.copy()
    triangles = [x for x, v in nx.triangles(G).items() if v > 0]
    binaries = [x for x in G.nodes() if x not in triangles]
    T.remove_nodes_from(binaries)
    B.remove_nodes_from(triangles)
    
    colors = ['w', 'r', 'y']
    names = ['cluster', 'binaries', 'triangles']
    names = [n+str(dis) for n in names]
    
    for k, g in enumerate([G, B, T]):
    
        xy_nodes = coords[g.nodes(), ::-1]
        pos = {n: p for n, p in zip(g.nodes(), xy_nodes)}
    
        d = nx.degree(g)
#        nx.draw(g, pos=pos, edge_vmax=1,
#                node_color=colors[k],
#                edge_color=colors[k],
#                node_size=[v * 100 for v in d.values()])
#        nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')
    
        groups = list(nx.connected_components(g))
        groups_dict = {}
        for item in groups:
            i = sorted(list(item))[0]
            num = reestr['№ проявления УВ (после кластеризации)  '].iloc[i]
            if pd.isnull(num):
                num = reestr['№ проявления УВ (после кластеризации)  '].max() + 1
            groups_dict[num] = item
        res = []
        reestr[names[k]] = np.nan
    
        for num, indexes in groups_dict.items():
            reestr[names[k]].iloc[list(indexes)] = num
    
#        plt.ticklabel_format(style='plain', axis='both', useOffset=False)
    
    reestr.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
    reestr_coord = reestr[~pd.isnull(reestr['x'])][['x', 'y']]
    reestr_coord['point'] = reestr_coord.apply(
        lambda row: np.nan if pd.isnull(row['x'])
        else _get_point(row['x'], row['y']), axis=1)
#plt.savefig('cluster.pdf', orientation='album')
    reestr['graph'] = reestr[
    'Группировка (кластеризация) - построение графа (типа "лес" расстояние между вершинами (проявлениями) - не более 100м)']
    gr_col_name = '№ проявления' + str(dis)
    reestr[gr_col_name] = np.nan
    num_gr = 1
    uniq = set()
    for row in reestr.iterrows():
        gr = row[1]['graph']
    
        if pd.isnull(gr):
            reestr[gr_col_name].ix[row[0]] = num_gr
            num_gr += 1
    
        elif gr in uniq:
            continue
    
        else:
            uniq.add(gr)
            reestr[gr_col_name][reestr['graph'] == gr] = num_gr
            num_gr += 1

reestr.to_csv('0209clusteriz_100.csv', sep=';')
#plt.show()
#reestr.rename(columns={'Unnamed: 30': 'work_section'}, inplace=True)
#zones_dict = {name: i for i, name in enumerate(reestr['Unnamed: 31'].unique())}
#reestr['zone'] = reestr.apply(
#    lambda row: zones_dict[row['Unnamed: 31']], axis=1)
#reestr.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
#reestr['graph'] = reestr[
#    'Группировка (кластеризация) - построение графа (типа "лес" расстояние между вершинами (проявлениями) - не более 100м)']
#reestr['№ проявления'] = np.nan
#reestr['№ проявления в участке'] = np.nan
#

#
#uniq = set()
#for section, section_group in reestr.groupby('work_section'):
#    num_gr = 1
#    print('section -', section)
#    # section_group = section_group[~pd.isnull(section_group['graph'])]
#    section_group.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
#    for row in section_group.iterrows():
#        gr = row[1]['graph']
#        if pd.isnull(gr):
#            reestr['№ проявления в участке'].ix[row[0]] = num_gr
#            num_gr += 1
#
#        elif gr in uniq:
#            continue
#
#        else:
#            uniq.add(gr)
#            reestr['№ проявления в участке'][reestr['graph'] == gr] = num_gr
#            num_gr += 1
        # reestr['№ проявления в участке'][reestr['graph'] == gr] = i + 1

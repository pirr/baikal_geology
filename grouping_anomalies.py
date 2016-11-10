import pandas as pd
from sklearn.neighbors import DistanceMetric
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from osgeo import ogr
from anomaly_searcher import get_min_weight_max_cliques
# from matplotlib.ticker import ScalarFormatter


def _get_point(x, y):
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(x, y)
    return point

distm = DistanceMetric.get_metric(metric='haversine')
#reestr = pd.read_excel('d:\work\BAIKAL\Geology\каталог\\1010Каталог проявлений УВ_раб.xls', 
#                   sheetname='реестр', header=19)
#df = pd.read_excel('d:\work\BAIKAL\Geology\каталог\\1309 Реестр проявлений углеводородов.xlsx', 
#                   sheetname='реестр', skiprows=2)
#cols = [c for c in df.columns[:12]]
#cols.extend(df.columns[19:])
#reestr = df[14:][cols]
#reestr.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
#reestr = reestr[reestr['Физические проявления в водной среде (толща, поверхн.)']=='пропарина']

reestr = pd.read_csv('d:\\work\\BAIKAL\\Geology\\2210catalogUV_GIS.csv', sep=';', encoding='cp1251')
reestr = reestr[~((reestr['Тип строки'] == 'ретро') & (reestr[' + - связь ретро и ФЦП'] == ' +'))]

coords = reestr[['y', 'x']].as_matrix()
yx = coords / 57.29578
D = distm.pairwise(yx)

for i in range(len(D)):
    D[i, :i + 1] = np.nan

r = D * 6372795
dists = [2000,]

for dis in dists:
    
    rx = np.where(r <= dis, r, np.NAN)
    true_indexes = np.argwhere(~np.isnan(rx))
    
    nodes = np.union1d(true_indexes[:, 0], true_indexes[:, 1])
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for index in true_indexes:
        G.add_edge(*index, weight=rx[index[0], index[1]])
    T = G.copy()
    B = G.copy()
    triangles = [x for x, v in nx.triangles(G).items() if v > 0]
    binaries = [x for x in G.nodes() if x not in triangles]
    T.remove_nodes_from(binaries)
    B.remove_nodes_from(triangles)
    
    colors = ['w', 'r']
    names = ['binaries', 'triangles', 'not_spec']
    names = [n+str(dis) for n in names]
    
    num = 1
    for k, g in enumerate([B, T, G]):
        
    
        xy_nodes = coords[g.nodes(), ::-1]
        pos = {n: p for n, p in zip(g.nodes(), xy_nodes)}
    
#        d = nx.degree(g)
#        nx.draw(g, pos=pos, edge_vmax=1,
#                node_color=colors[k],
#                edge_color=colors[k],
#                node_size=[v * 100 for v in d.values()])
#        nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')
    
        reestr[names[k]] = np.nan
    
        for indexes in nx.connected_components(g):
            reestr[names[k]].iloc[list(indexes)] = num
            num += 1
    
    num = 1
    max_clique_name = 'max_clique' + str(dis)
    reestr[max_clique_name] = np.nan
    min_weight_max_cliques = get_min_weight_max_cliques(G)
    for clique in min_weight_max_cliques:
        reestr[max_clique_name].iloc[clique] = num
        num += 1

#reestr.sort_values(by=['№ проявления УВ (после кластеризации)  '], inplace=True)
reestr.to_csv('0711clusteriz_2000.csv', sep=';')

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

dist = DistanceMetric.get_metric(metric='haversine')

df = pd.read_excel('testdata//1608_Реестр проявлений углеводородов.xlsx')
cols = [c for c in df.columns[:10]]
cols.extend(df.columns[16:])
reestr = df[7:][cols]
coords = reestr[['y', 'x']].as_matrix()
yx = coords / 57.29578
D = dist.pairwise(yx)

for i in range(len(D)):
    D[i, :i + 1] = np.nan

r = D * 6372795

rx = np.where(r <= 100, r, np.NAN)
true_indexes = np.argwhere(~np.isnan(rx))
dist = rx[~np.isnan(rx)]

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

for k, g in enumerate([G, B, T]):

    xy_nodes = coords[g.nodes(), ::-1]
    pos = {n: p for n, p in zip(g.nodes(), xy_nodes)}

    d = nx.degree(g)
    nx.draw(g, pos=pos, edge_vmax=1,
            node_color=colors[k],
            edge_color=colors[k],
            node_size=[v * 100 for v in d.values()])
    nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')

    groups = list(nx.connected_components(g))
    groups_dict = {}
    for item in groups:
        i = list(item)[0]
        num = int(reestr['№ 2015'].iloc[i])
        groups_dict[num] = item
    res = []
    reestr[names[k]] = np.nan

    for num, indexes in groups_dict.items():
        reestr[names[k]].iloc[list(indexes)] = num

    plt.ticklabel_format(style='plain', axis='both', useOffset=False)

reestr.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
reestr_coord = reestr[~pd.isnull(reestr['x'])][['x', 'y']]
reestr_coord['point'] = reestr_coord.apply(
    lambda row: np.nan if pd.isnull(row['x'])
    else _get_point(row['x'], row['y']), axis=1)

plt.show()

reestr.rename(columns={'Unnamed: 30': 'work_section'}, inplace=True)
zones_dict = {name: i for i, name in enumerate(reestr['Unnamed: 31'].unique())}
reestr['zone'] = reestr.apply(
    lambda row: zones_dict[row['Unnamed: 31']], axis=1)
# reestr.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
reestr['graph'] = reestr[
    'Группировка (кластеризация) - построение графа (типа "лес" расстояние между вершинами (проявлениями) - не более 100м)']
reestr['№ проявления'] = np.nan
reestr['№ проявления в участке'] = np.nan

X = reestr[~pd.isnull(reestr['graph'])]
X.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
for i, gr in enumerate(X['graph'].unique()):
    reestr['№ проявления'][reestr['graph'] == gr] = i + 1


for section, section_groups in reestr.groupby('work_section'):
    print('section -', section)
    section_groups = section_groups[~pd.isnull(section_groups['graph'])]
    section_groups.sort_values(by=['y', 'x'], ascending=[0, 1], inplace=True)
    for i, gr in enumerate(section_groups['graph'].unique()):
        reestr['№ проявления в участке'][reestr['graph'] == gr] = i + 1

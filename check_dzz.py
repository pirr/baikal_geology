# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 10:28:38 2016

@author: Smaga
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import DistanceMetric
import networkx as nx

from anomaly_searcher import get_min_weight_max_cliques



distm = DistanceMetric.get_metric(metric='haversine')

reestr = pd.read_excel('d:\work\BAIKAL\Geology\каталог\\1010Каталог проявлений УВ_раб.xls', 
                   sheetname='реестр', header=19)
ved_dzz = pd.read_excel('d:\work\BAIKAL\Geology\каталог\\ведомость_пропарины2016.xls', 
                   header=4)
R_num_dict = {v: k for k, v in enumerate(ved_dzz['Повторяемость'].unique())}
ved_dzz['R_num'] = ved_dzz['Повторяемость'].apply(lambda x: R_num_dict[x])
coords_reestr = reestr[['y', 'x']].as_matrix()


#cliques_dict = {}
#for tp, group in ved_dzz.groupby('Повторяемость'):
   
coords_ved_dzz = ved_dzz[['Y_DD', 'X_DD']].as_matrix()
coords = np.concatenate((coords_reestr, coords_ved_dzz), axis=0)

yx = coords / 57.29578
D = distm.pairwise(yx)

for i in range(len(D)):
    D[i, :i + 1] = np.nan

r = D * 6372795
dists = [50, ]

for dis in dists:
    gr_num = reestr['№ проявления УВ  '].max() + 1
    rx = np.where(r <= dis, r, np.NAN)
    true_indexes = np.argwhere(~np.isnan(rx))
    
    nodes = np.union1d(true_indexes[:, 0], true_indexes[:, 1])
    
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for index in true_indexes:
        G.add_edge(*index, weight=rx[index[0], index[1]])
    
    min_weight_max_cliques = get_min_weight_max_cliques(G)
    dzz_ind = []
    for clique in min_weight_max_cliques:
        
        reestr_ind = [ind for ind in clique if ind < len(reestr)]
        ved_ind = [(ind - len(reestr)) for ind in clique if ind not in reestr_ind]
        dzz_ind.extend(ved_ind)
        if ved_ind and reestr_ind:
            ved_r_fid = ved_dzz.loc[ved_ind, 'N п/п'].values
            ved_dzz_rnum = set(ved_dzz.loc[ved_ind, 'R_num'].values)
            dzz_ind
            if len(ved_dzz_rnum) > 1:
                print(ved_r_fid, reestr.loc[reestr_ind, '№ проявления УВ  '])
            else:
                negative = ''
                ved_dzz_R_num = ','.join([str(x) for x in ved_dzz_rnum])
                if ved_dzz_R_num == '0':
                    negative = 'не подтв.'
                reestr.loc[reestr_ind, 'ved_dzz_prot_dzz_num'] = negative + '2016-1-d-' + ','.join([str(x) for x in ved_r_fid])
                reestr.loc[reestr_ind, 'ved_dzz_R_num'] = ved_dzz_R_num
                ved_dzz.loc[ved_ind, 'reestr_num'] = ','.join(str(x) for x in set(reestr.loc[reestr_ind, '№ проявления УВ  '].values))

        elif ved_ind and not reestr_ind:
            ved_dzz.loc[ved_ind, 'reestr_num'] = gr_num
            gr_num += 1
              
reestr.to_csv('1110_check_dzz.csv', sep=';')
ved_dzz.to_csv('1110ved_dzz_with_reestr_num.csv', sep=';')
            
     
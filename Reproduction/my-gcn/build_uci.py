import numpy as np
import pandas as pd
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import torch 


def build_uci_graph(dataset):
    g = dgl.DGLGraph()
    l1 = len(dataset['age'])
    g.add_nodes(l1)

    edges = []

    print(dataset['age'])

    for i in range(len(dataset['age'])):
        for j in range(i, len(dataset['age'])):
            if abs(dataset['age'][i] - dataset['age'][j]) <= 1:
                edges.append((i,j))



    # print(edges) 

    src , dst = tuple(zip(*edges))
    g.add_edges(src,dst)
    g.add_edges(dst,src)
    
    return g



# fig, ax = plt.subplots()
# fig.set_tight_layout(False)
# nx_G = g.to_networkx().to_undirected()
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[0.9,0.9,0.]])
# plt.show()
# plt.savefig('./graph_uci.png')

def build_feature_matrix(dataset):

    dataset = pd.get_dummies(dataset, columns = ['sex','cp','fbs','restecg','exang','slope','ca','thal'])
    dataset.info()
    standardScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
    dataset.info()
    dataset = dataset.drop(columns = ['id','age'])

    my_data = torch.from_numpy(dataset.values).float()
    # print(my_data)

    return my_data
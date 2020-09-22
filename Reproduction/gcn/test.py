import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

names = ['x','y','tx','ty','allx','ally','graph']
objects =[]
for i in range(len(names)):
    with open("./gcn/data/ind.{}.{}".format('cora',names[i]),'rb') as f:
        # print(sys.version_info)
        objects.append(pkl.load(f, encoding='latin1'))
        # print((objects[0]))

x, y, tx, ty, allx, ally, graph = tuple(objects)
print(x.shape)
print(y.shape)
print(tx.shape)
# # print(tx[:,:])
print(ty.shape)
# print(ty)
print(allx.shape)
print(ally.shape)
# # for i in range(len(graph)):
# #     print(graph[i])
# print((graph))
print(len(graph))
test_idx_reorder = parse_index_file("./gcn/data/ind.{}.test.index".format("cora"))
print(len(test_idx_reorder))
test_idx_range = np.sort(test_idx_reorder)
#print(test_idx_range)

# if dataset_str == 'citeseer':
#     # Fix citeseer dataset (there are some isolated nodes in the graph)
#     # Find isolated nodes, add them as zero-vecs into the right position
#     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#     tx_extended[test_idx_range-min(test_idx_range), :] = tx
#     tx = tx_extended
#     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#     ty_extended[test_idx_range-min(test_idx_range), :] = ty
#     ty = ty_extended
features = sp.vstack((allx,tx)).tolil()
# print(features)
# 调整顺序
features[test_idx_reorder, :] = features[test_idx_range, :]
# print(features)

# 建立连接矩阵
adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
print(adj.shape)

labels = np.vstack((ally,ty))
# 调整顺序
labels[test_idx_reorder,:] = labels[test_idx_range,:]
# print(test_idx_reorder[0:10])
# print(test_idx_range[0:10])

idx_test = test_idx_range.tolist()
idx_train = range(len(y))
idx_val = range(len(y),len(y)+500)


train_mask = sample_mask(idx_train, labels.shape[0])
print(len(train_mask))
val_mask = sample_mask(idx_val, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])

y_train = np.zeros(labels.shape)
print(y_train.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
print(y_train)
y_train[train_mask, :] = labels[train_mask, :]
print(y_train)
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]
print("==========================")
print(features.shape)
print(train_mask.shape)
print(val_mask.shape)
print(adj[100,100])
print(features.shape)

rowsum = np.array(features.sum(1))
print(len(rowsum))


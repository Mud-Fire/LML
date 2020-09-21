import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import matplotlib.animation as animation 
import matplotlib.pyplot as plt

from model import GCN
from build_graph import build_karate_club_graph

import warnings
warnings.filterwarnings('ignore')



net = GCN(34, 6, 2)
print(net)
G = build_karate_club_graph()
print(G)

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])
print(labeled_nodes)
labels = torch.tensor([0, 1])
print(labels)

optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
all_logits = []


# def gcn_message(edges):
#     """
#     compute a batch of message called 'msg' using the source nodes' feature 'h'
#     :param edges:
#     :return:
#     """
#     return {'msg': edges.src['h']}

# def gcn_reduce(nodes):
#     """
#     compute the new 'h' features by summing received 'msg' in each node's mailbox.
#     :param nodes:
#     :return:
#     """
#     return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

# G.ndata['h'] = inputs
# print(G)
# print(G.nodes[1].data['h'])
# G.send_and_recv(G.edges(), gcn_message, gcn_reduce)
# h = G.ndata.pop('h')
# print(nn.Linear(h))
# print(G)
# print(G.edges().src['h'])
# print(G.nodes().src['h'])
# print("=====================")

# fig, ax = plt.subplots()
# fig.set_tight_layout(False)
# nx_G = G.to_networkx().to_undirected()
# pos = nx.kamada_kawai_layout(nx_G)
# nx.draw(nx_G, pos, with_labels=True, node_color=[[0.9,0.9,0.]])
# plt.show()
# plt.savefig('./graph1.png')


for epoch in range(20):
    logits = net(G, inputs)
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    print(logp)

    loss = F.nll_loss(logp[labeled_nodes],labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('epoch %d | Loss: %.4f' % (epoch, loss.item()))

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    # ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors, with_labels=True, node_size=300, ax=ax)

nx_G =G.to_networkx().to_undirected()
fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()

ani = animation.FuncAnimation(fig, draw,frames = len(all_logits),interval = 200)
ani.save('basic_animation2.gif', writer='imagemagick', fps=1)
plt.show()


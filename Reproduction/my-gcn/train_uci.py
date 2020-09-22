import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
import matplotlib.animation as animation 
import matplotlib.pyplot as plt
import pandas as pd
from model import GCN
from build_uci import build_uci_graph
from build_uci import build_feature_matrix
import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("dataset.csv")
dataset.info()
dataset.describe(include = 'all').to_csv("./doc/data_describe.csv")

l1 = len(dataset['age'])
print(l1)

net = GCN(30, 10, 2)
print(net)
G = build_uci_graph(dataset)
print(G)



inputs = build_feature_matrix(dataset)
print(inputs.shape)
inputs_1 = torch.eye(l1)
print(inputs_1)


labeled_nodes = torch.tensor([0,1,5,10,30,50,60,150,160,180,190,200,220,240,250,270,280,290,300])
print(labeled_nodes)
labels = torch.tensor([0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
print(labels)

labeled_nodes1 = torch.tensor([range(100)])
labeled_nodes2 = torch.tensor([range(167,250)])

labeled_nodes =torch.cat((labeled_nodes1, labeled_nodes2), 1)[0]
print(labeled_nodes)
labels1 = torch.zeros(100)
labels2 = torch.ones(83)
labels = torch.cat((labels1,labels2), 0)
print(labels)



optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)
all_logits = []


for epoch in range(200):
    logits = net(G, inputs)
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)

    # print(logp)

    loss = F.nll_loss(logp[labeled_nodes],labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('epoch %d | Loss: %.4f' % (epoch, loss.item()))


print(all_logits[0][5])
print(all_logits[19][5])

dead = []
for i in range(l1):
    if all_logits[19][i].argmax():
        dead.append(i)

print(dead)
print(len(dead))
print(dataset['target'])


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(l1):
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
ani.save('basic_animation3.gif', writer='imagemagick', fps=1)
plt.show()


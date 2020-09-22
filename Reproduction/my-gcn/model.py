import torch
import torch.nn as nn

def gcn_message(edges):
    """
    compute a batch of message called 'msg' using the source nodes' feature 'h'
    :param edges:
    :return:
    """
    # print({'msg': edges.src['h']})
    return {'msg': edges.src['h']}

def gcn_reduce(nodes):
    """
    compute the new 'h' features by summing received 'msg' in each node's mailbox.
    :param nodes:
    :return:
    """
    return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}

class GCNLayer(nn.Module):
    
    # Define GCNLayer

    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
    
    # def forward(self, g, inputs):
    #     # g - graph || inputs - node features
    #     g.ndata['h'] = inputs
    #     g.send(g.edges(), gcn_message)
    #     # trigger aggregation at all nodes
    #     g.recv(g.nodes(), gcn_reduce)
    #     h = g.ndata.pop('h')

    #     return self.linear(h)

    def forward(self, g, inputs):
        g.ndata['h'] = inputs
        g.send_and_recv(g.edges(), gcn_message, gcn_reduce)
        h = g.ndata.pop('h')

        return self.linear(h)

class GCN(nn.Module):
    
    # Define a 2-layer GCN model.

    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g,h)

        return h



# if __name__ == "__main__":
#     net = GCN(34, 5 ,2)

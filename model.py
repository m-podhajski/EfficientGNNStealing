import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import rand
import dgl.nn as dglnn
import torch


def attrMask(x, mask_probabilityattr):
    mask = rand((x.shape[1],)) < mask_probabilityattr
    x = x * mask.to(x.device)
    return x


class SAGEEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout
                 ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))

        self.layers.append(dglnn.SAGEConv(n_hidden, n_output_dim, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x, train, adj, prob_attr):
        if train:
            x = attrMask(x, prob_attr)
        h = x

        for i in range(0, self.n_layers):
            if train:
                edge_weights = torch.from_numpy(adj.toarray()[blocks[i].srcdata['_ID'].cpu(), :][:, blocks[i].dstdata['_ID'].cpu(
                )][blocks[i].edges()[0].cpu(), blocks[i].edges()[1].cpu()]).float().to(x.device)
            else:
                edge_weights = None
            h = self.layers[i](blocks[i], h, edge_weight=edge_weights)
            h = self.activation(h)
            if i != self.n_layers - 1:
                h = self.dropout(h)

        return h


class GATEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 num_heads,
                 num_workers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.num_workers = num_workers
        self.layers.append(dglnn.GATConv((in_feats, in_feats), n_hidden, num_heads=num_heads,
                           feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_hidden,
                               num_heads=num_heads, feat_drop=0., attn_drop=0., activation=activation, negative_slope=0.2))
        self.layers.append(dglnn.GATConv((n_hidden * num_heads, n_hidden * num_heads), n_output_dim,
                           num_heads=num_heads, feat_drop=0., attn_drop=0., activation=None, negative_slope=0.2))

    def forward(self, blocks, x, train, adj, prob_attr):
        if train:
            x = attrMask(x, prob_attr)
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if train:
                edge_weights = torch.from_numpy(adj.toarray()[block.srcdata['_ID'].cpu(), :][:, block.dstdata['_ID'].cpu(
                )][block.edges()[0].cpu(), block.edges()[1].cpu()]).float().to(x.device)
            else:
                edge_weights = None
            h_dst = h[:block.number_of_dst_nodes()]
            if l < self.n_layers - 1:
                h = layer(block, (h, h_dst),
                          edge_weight=edge_weights).flatten(1)
            else:
                h = layer(block, (h, h_dst), edge_weight=edge_weights)
        h = h.mean(1)
        return h


class GINEMB(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_output_dim,
                 n_classes,
                 n_layers,
                 activation,
                 batch_size,
                 num_workers,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_output_dim = n_output_dim
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        linear = nn.Linear(in_feats, n_hidden)
        self.layers.append(dglnn.GINConv(linear, 'mean'))
        for i in range(1, n_layers-1):
            linear = nn.Linear(n_hidden, n_hidden)
            self.layers.append(dglnn.GINConv(linear, 'mean'))
        linear = nn.Linear(n_hidden, n_output_dim)
        self.layers.append(dglnn.GINConv(linear, 'mean'))

        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def forward(self, blocks, x, train, adj, prob_attr):
        if train:
            x = attrMask(x, prob_attr)
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            if train:
                edge_weights = torch.from_numpy(adj.toarray()[block.srcdata['_ID'].cpu(), :][:, block.dstdata['_ID'].cpu(
                )][block.edges()[0].cpu(), block.edges()[1].cpu()]).float().to(x.device)
            else:
                edge_weights = None
            h = layer(block, h, edge_weight=edge_weights)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = F.elu(self.fc1(x))
        return z


class Model(nn.Module):

    def __init__(self, in_dim, out_dim, temp, n_classes, architecture, prob_attr):
        super(Model, self).__init__()
        if architecture == 'gat':
            self.encoder = GATEMB(in_dim, 128, out_dim,
                                  n_classes, 2, 4, 4, F.relu, 0.5)
        elif architecture == 'sage':
            self.encoder = SAGEEMB(in_dim, 128, out_dim,
                                   n_classes, 2, F.relu, 1000, 4, 0.5)
        elif architecture == 'gin':
            self.encoder = GINEMB(in_dim, 128, out_dim,
                                  n_classes, 2, F.relu, 1000, 4, 0.5)

        self.temp = temp

        self.proj = MLP(out_dim, 128)
        self.prob_attr = prob_attr

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)

        s = th.mm(z1, z2.t())
        return s

    def get_loss(self, z1, z2):
        def f(x): return th.exp(x / self.temp)

        refl_sim = f(self.sim(z1, z1)).cuda()
        between_sim = f(self.sim(z1, z2)).cuda()

        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()
        loss = -th.log(between_sim.diag() / x1)

        return loss

    def get_embedding(self, graph, feat):
        self.encoder.eval()
        h = self.encoder([graph, graph], feat, False, False, self.prob_attr)

        return h.detach()

    def forward(self, graphs, feat1, embds, adj):
        h1 = self.encoder(graphs, feat1, True, adj, self.prob_attr)

        z1 = self.proj(h1)

        z2 = self.proj(embds)

        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) / 2

        return ret.mean()

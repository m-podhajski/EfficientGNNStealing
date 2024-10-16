import argparse
import warnings

import numpy as np
import torch as th
from eval import label_classification
from model import Model
import dgl
from graphgallery.datasets import NPZDataset
import networkx as nx

import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import scipy.sparse as sp


warnings.filterwarnings("ignore")


def dropEdge(graph, drop_ratio=0.05):
    num_edges = graph.number_of_edges()
    num_edges_to_drop = int(num_edges * drop_ratio)
    indices_to_drop = torch.randperm(num_edges)[:num_edges_to_drop].tolist()
    graph.remove_edges(indices_to_drop)
    return graph


def update(theta, epoch, total):
    theta = theta - theta * (epoch / total)
    return theta


def sinkhorn(K, dist, sin_iter):
    # make the matrix sum to 1
    u = np.ones([len(dist), 1]) / len(dist)
    K_ = sp.diags(1.0 / dist) * K
    dist = dist.reshape(-1, 1)
    ll = 0
    for it in range(sin_iter):
        u = 1.0 / K_.dot(dist / (K.T.dot(u)))
    v = dist / (K.T.dot(u))
    delta = np.diag(u.reshape(-1)).dot(K).dot(np.diag(v.reshape(-1)))
    return delta


def plug(theta, laplace, delta_add, delta_dele, epsilon, dist, sin_iter, c_flag=False):
    C = (1 - theta) * laplace.A
    if c_flag:
        C = laplace.A

    K_add = np.exp(2 * (C * delta_add).sum() * C / epsilon)
    K_dele = np.exp(-2 * (C * delta_dele).sum() * C / epsilon)

    delta_add = sinkhorn(K_add, dist, sin_iter)

    delta_dele = sinkhorn(K_dele, dist, sin_iter)
    return delta_add, delta_dele


def projection(
    X, y, transform_name="TSNE", show_figure=False, gnn="Graphsage", dataset="Cora"
):

    if transform_name == "TSNE":
        transform = TSNE
        trans = transform(n_components=2, n_iter=3000, n_jobs=-1)
        emb_transformed = pd.DataFrame(trans.fit_transform(X))
    elif transform_name == "PCA":
        transform = PCA
        trans = transform(n_components=2)
        emb_transformed = pd.DataFrame(trans.fit_transform(X))

    return emb_transformed.iloc[:, [0, 1]]


def compute_fidelity(pred_surrogate, pred_target):
    _surrogate = th.argmax(pred_surrogate, dim=1)
    _target = th.argmax(pred_target, dim=1)
    _fidelity = (_surrogate == _target).float().sum() / len(pred_surrogate)
    return _fidelity.clone().detach().cpu().item()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(np.abs(adj.A).sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def load_graphgallery_data(dataset):
    # set `verbose=False` to avoid additional outputs
    data = NPZDataset(dataset, verbose=False)
    graph = data.graph
    nx_g = nx.from_scipy_sparse_array(graph.adj_matrix)

    for node_id, node_data in nx_g.nodes(data=True):
        node_data["features"] = graph.node_attr[node_id].astype(np.float32)
        if dataset in ["blogcatalog", "flickr"]:
            node_data["labels"] = graph.node_label[node_id].astype(np.long) - 1
        else:
            node_data["labels"] = graph.node_label[node_id].astype(np.long)

    dgl_graph = dgl.from_networkx(nx_g, node_attrs=["features", "labels"])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    return dgl_graph, len(np.unique(graph.node_label))


def split_graph(g, frac_list=[0.2, 0.3, 0.5]):

    val_subset, train_subset, test_subset = dgl.data.utils.split_dataset(
        g.nodes(), frac_list=frac_list, shuffle=True, random_state=12
    )

    train_g = g.subgraph(train_subset.indices)
    val_g = g.subgraph(val_subset.indices)
    test_g = g.subgraph(test_subset.indices)

    if not "features" in train_g.ndata:
        train_g.ndata["features"] = train_g.ndata["feat"]
    if not "labels" in train_g.ndata:
        train_g.ndata["labels"] = train_g.ndata["label"]

    if not "features" in val_g.ndata:
        val_g.ndata["features"] = val_g.ndata["feat"]
    if not "labels" in train_g.ndata:
        val_g.ndata["labels"] = val_g.ndata["label"]

    if not "features" in test_g.ndata:
        test_g.ndata["features"] = test_g.ndata["feat"]
    if not "labels" in train_g.ndata:
        test_g.ndata["labels"] = test_g.ndata["label"]
    return val_g, train_g, test_g


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="citeseer_full")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument(
    "--epochs", type=int, default=200, help="Number of training periods."
)
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay.")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature.")

parser.add_argument("--act_fn", type=str, default="relu")
parser.add_argument("--task", type=str, default="embedding")
parser.add_argument("--surrogate", type=str, default="gat")
parser.add_argument("--target", type=str, default="gin")

parser.add_argument("--turn", type=int, default=20)
parser.add_argument("--type", type=str, default="i")
parser.add_argument("--ratio_q", type=float, default=1.0)
parser.add_argument("--noise", type=float, default=-1.0)


args = parser.parse_args()

args.prob_attr = {"gin": 0.5, "sage": 0.5, "gat": 0.5}[args.target]


if args.gpu != -1 and th.cuda.is_available():
    args.device = "cuda:{}".format(args.gpu)
else:
    args.device = "cpu"


def main():
    lr = args.lr

    temp = args.temp
    epochs = args.epochs
    wd = args.wd

    g, n_classes = load_graphgallery_data(args.dataset)
    val_g, train_g, test_g = split_graph(g, frac_list=[0.2, 0.3, 0.5])
    train_size = round(train_g.number_of_nodes() * args.ratio_q)
    in_dim = train_g.ndata["features"].shape[-1]
    if args.type == "i":
        pred = torch.load(
            f"./target_outputs/query_preds-{args.dataset}-{args.target}.pth"
        )[:train_size]
        embs = torch.load(
            f"./target_outputs/query_embs-{args.dataset}-{args.target}.pth"
        )[:train_size]
        test_pred_target = torch.load(
            f"./target_outputs/test_pred-{args.dataset}-{args.target}.pth"
        )
        tsne_embs = projection(
            embs.clone().detach().cpu().numpy(),
            train_g.ndata["labels"],
            transform_name="TSNE",
            gnn="sage",
        )
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(args.device)
        pred = pred.to(args.device)
        embs = embs.to(args.device)
    elif args.type == "ii":
        sparse_adj = sp.load_npz(
            f"./target_outputs/idgl-adj-{args.dataset}-{args.target}.npz",
        )
        train_g2 = dgl.from_scipy(sparse_adj)
        train_g2.ndata["features"] = train_g.ndata["features"]
        train_g2.ndata["labels"] = train_g.ndata["labels"]
        train_g2 = dgl.add_self_loop(train_g2)
        pred = torch.load(
            f"./target_outputs/idgl-query_preds-{args.dataset}-{args.target}.pth"
        )[:train_size]
        print(pred.shape)
        embs = torch.load(
            f"./target_outputs/idgl-query_embs-{args.dataset}-{args.target}.pth"
        )[:train_size]
        test_pred_target = torch.load(
            f"./target_outputs/idgl-test_pred-{args.dataset}-{args.target}.pth"
        )
        tsne_embs = projection(
            embs.clone().detach().cpu().numpy(),
            train_g.ndata["labels"],
            transform_name="TSNE",
            gnn="sage",
        )
        tsne_embs = torch.from_numpy(tsne_embs.values).float().to(args.device)
        pred = pred.to(args.device)
        embs = embs.to(args.device)
        train_g = train_g2
    train_g = train_g.subgraph(train_g.nodes()[:train_size])
    if args.task == "embedding":
        queries = embs
    elif args.task == "prediction":
        queries = pred
    elif args.task == "projection":
        queries = tsne_embs

    if args.noise > 0:
        queries = queries + (torch.randn(queries.shape)
                             * args.noise).to(args.device)

    model = Model(
        in_dim, queries.shape[-1], temp, n_classes, args.surrogate, args.prob_attr
    )
    model = model.to(args.device)

    optimizer = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    theta = 1
    delta = np.ones((train_g.num_nodes(), train_g.num_nodes())) * 0.5
    delta_add = delta
    delta_dele = delta
    scope = torch.stack(train_g.edges(), dim=0)
    adj = train_g.adj_external(scipy_fmt="coo")
    laplace = sp.eye(adj.shape[0]) - normalize_adj(adj)
    scope_matrix = sp.coo_matrix(
        (np.ones(scope.shape[1]), (scope[0, :], scope[1, :])), shape=adj.shape
    ).A
    dist = adj.A.sum(-1) / adj.A.sum()
    new_adj = adj.tocsc()

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in "10,50".split(",")]
    )

    for epoch in range(epochs):
        if args.target == "sage":
            model.prob_attr = torch.rand(())
        if epoch % args.turn == 0:
            theta = update(1, epoch, epochs)
            if epoch == 0:
                delta_add, delta_dele = plug(
                    theta, laplace, delta_add, delta_dele, 0.1, dist, 3, True
                )
            else:
                delta_add, delta_dele = plug(
                    theta, laplace, delta_add, delta_dele, 0.1, dist, 3, False
                )
            delta = (delta_add - delta_dele) * scope_matrix
            delta = 0.1 * normalize_adj(delta)

            new_adj = adj + delta
            new_graph = dgl.from_scipy(new_adj)
            new_graph.ndata["features"] = train_g.ndata["features"]
            new_graph.ndata["labels"] = train_g.ndata["labels"]
            new_graph = dropEdge(new_graph)
            dataloader = dgl.dataloading.DataLoader(
                new_graph.add_self_loop(),
                new_graph.nodes(),
                sampler,
                batch_size=1000,
                shuffle=True,
                drop_last=False,
                num_workers=4,
                device=args.device,
            )

        model.train()
        optimizer.zero_grad()
        loss_epoch = 0

        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            loss = model(
                blocks,
                blocks[0].srcdata["features"],
                queries[blocks[-1].dstdata["_ID"]],
                new_adj,
            )
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
        print(f"Epoch={epoch:03d}, loss={loss.item():.4f}")
        if epoch % 100 == 0:

            model.eval()

            graph = train_g.add_self_loop()
            graph = graph.to(args.device)
            feat = train_g.ndata["features"].to(args.device)
            embeds = model.get_embedding(graph, feat)

            graph = test_g.add_self_loop()
            graph = graph.to(args.device)
            feat = test_g.ndata["features"].to(args.device)
            test_embeds = model.get_embedding(graph, feat)

            detached_classifier, acc = label_classification(
                embeds, test_embeds, train_g, test_g
            )
            _predicts = detached_classifier.predict_proba(
                test_embeds.clone().detach().cpu().numpy()
            )
            _fidelity = compute_fidelity(
                torch.from_numpy(_predicts).to(args.device),
                test_pred_target.to(args.device),
            )

    return acc, _fidelity


if __name__ == "__main__":
    accs = []
    fidels = []
    for _ in range(3):
        acc, fidel = main()
        accs.append(acc)
        fidels.append(fidel)
    print(
        args.dataset,
        args.target,
        args.surrogate,
        args.task,
        args.type,
        args.ratio_q,
        np.array(accs).mean(),
        np.array(accs).std(),
        np.array(fidels).mean(),
        np.array(fidels).std(),
        args.noise,
        file=open("results.txt", "a"),
    )

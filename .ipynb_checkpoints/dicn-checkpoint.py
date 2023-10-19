import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
import networkx as nx


from SuperGAT.model import SuperGATNet
from SuperGAT.arguments import get_args
from SuperGAT.data import getattr_d, get_dataset_or_loader
from SuperGAT.layer import SuperGAT
from SuperGAT.layer_cgat import CGATConv

from torch_geometric.datasets import Planetoid

import utils
from utils import data_preprocessing, get_dataset
from model import GCNModelGumbel
from evaluation import eva
import community


class DAEGC(nn.Module):
    def __init__(self, device, dropout, edge_num, embedding_vg, num_features, node_num, embedding_size, alpha,
                 num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.node_num = node_num
        self.gat = SuperGATNet(args=sgat_args, dataset_or_loader=train_d)

        self.vGraph = GCNModelGumbel(node_num, embedding_vg, num_clusters, dropout, device)
        # self.encode1=nn.Linear(num_clusters,num_clusters) # (7,7)
        # self.dropout=nn.Dropout(0.5)
        # self.encode3=nn.Linear(embedding_size,embedding_size)  # (128,128)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))  # （7,128）
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.zeros_(m.bias)

    def pre_forward(self, dataset):
        z = self.gat(dataset.x, dataset.edge_index,
                     batch=getattr(dataset, "batch", None),
                     attention_edge_index=getattr(dataset, "train_edge_index", None))

        return z

    def forward(self, dataset, w, c, temp):

        z = self.gat(dataset.x, dataset.edge_index,
                     batch=getattr(dataset, "batch", None),
                     attention_edge_index=getattr(dataset, "train_edge_index", None))
        q = self.get_Q(z)

        recon_c, q_vg, prior, node_embeddings, community_embeddings = self.vGraph(w, c, temp)

        res = torch.zeros([args.node_num, args.n_clusters], dtype=torch.float32).to(device)
        for idx, e in enumerate(args.train_edges):
            res[e[0], :] += q_vg[idx, :]
            res[e[1], :] += q_vg[idx, :]
        from torch.nn.functional import normalize
        Q_to = q + 0.5 * res

        # Q_to=normalize(q, p=1.0, dim = 1)+0.5*normalize(res, p=1.0, dim = 1)
        # Q_to=q+torch.sigmoid(self.dropout(self.encode1(res)))
        Q_to = normalize(Q_to, p=1, dim=1)

        ebs = node_embeddings.weight
        ebs_c = community_embeddings
        return z, q, Q_to, prior, recon_c, q_vg, ebs, ebs_c

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def load_data(dataset):  # Planetoid to ours
    adj=dataset.adj_label.numpy()

    adj_=adj-np.eye(adj.shape[0])

    label=dataset.y.numpy()
    G=nx.from_numpy_array(adj_)
    membership=[label[i] for i in range(G.number_of_nodes())]
    return G,nx.adjacency_matrix(G), membership


def loss_function_v(recon_c, q_y, prior, c, norm=None, pos_weight=None):
    BCE = F.cross_entropy(recon_c, c, reduction='sum') / c.shape[0]
    # BCE = F.binary_cross_entropy_with_logits(recon_c, c, pos_weight=pos_weight)
    # return BCE

    log_qy = torch.log(q_y  + 1e-20)
    KLD = torch.sum(q_y*(log_qy - torch.log(prior)),dim=-1).mean()

    ent = (- torch.log(q_y) * q_y).sum(dim=-1).mean()
    # return BCE
    return BCE + KLD


def get_assignment(G, device,model, dataset,num_classes=5, tpe=0):
    model.eval()
    edges = [(u,v) for u,v in G.edges()]
    batch = torch.LongTensor(edges).to(device)  # （5278,2）

    _, _,_,_,_,q, _,_ = model(dataset,batch[:, 0], batch[:, 1], 1.)

    num_classes = q.shape[1]
    q_argmax = q.argmax(dim=-1)

    assignment = {}

    n_nodes = G.number_of_nodes()

    res = np.zeros((n_nodes, num_classes))
    for idx, e in enumerate(edges):
        if tpe == 0:
            res[e[0], :] += q[idx, :].cpu().data.numpy()
            res[e[1], :] += q[idx, :].cpu().data.numpy()
        else:
            res[e[0], q_argmax[idx]] += 1
            res[e[1], q_argmax[idx]] += 1

    res = res.argmax(axis=-1)
    assignment = {i : res[i] for i in range(res.shape[0])}
    return res, assignment


def trainer(dataset, device):

    ANNEAL_RATE = 0.00003
    temp_min = 0.3
    temp = 1.

    model = DAEGC(device, args.dropout, edge_num=args.edge_size, embedding_vg=args.embedding_vg,
                  num_features=args.input_dim, node_num=args.node_num,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)

    dataset_lt = data_preprocessing(dataset)

    data1 = torch.Tensor(dataset_lt.x).to(device)
    data = np.load('./pretrain/EV22NSO8_cora_dotproducted_vectors_128.npy')
    z = torch.Tensor(data)
    y = dataset_lt.y.cpu().numpy()

    G, adj, gt_membership = load_data(dataset_lt)

    train_edges = [(u, v) for u, v in G.edges()]
    args.train_edges = train_edges
    batch = torch.LongTensor(train_edges)
    assert batch.shape == (len(train_edges), 2)

    w = torch.cat((batch[:, 0], batch[:, 1])).to(device)
    c = torch.cat((batch[:, 1], batch[:, 0])).to(device)

    categorical_dim = len(set(gt_membership))
    n_nodes = G.number_of_nodes()
    dataset = dataset.to(device)
    dataset_lt = dataset_lt.to(device)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, nmi, ari, f1 = eva(y, y_pred, 'pretrain')

    model.vGraph.node_embeddings.weight = Parameter(z.to(device))

    model.vGraph.community_embeddings.weight = Parameter(model.cluster_layer)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    z_to = z
    best_acc, best_nmi, best_modu = 0., 0., 0.
    nmi_acc, nmi_ari, nmi_f1 = 0., 0., 0.
    best_acc_epoch, best_nmi_epoch, best_modu_epoch = 0, 0, 0
    nmi_lst = []

    for epoch in range(args.max_epoch):
        model.train()

        if epoch % args.update_interval == 0:
            z, Q, Q_to, prior, recon_c, q_vg, z_v, z_c = model(dataset_lt, w, c, temp)
            # p = target_distribution(Q.detach())
            p_to = target_distribution(Q_to.detach())
            Q_v = model.get_Q(z_v)
        else:
            z, Q, Q_to, prior, recon_c, q_vg, z_v, z_c = model(dataset_lt, w, c, temp)
            # p_to = target_distribution(Q_to.detach())
            Q_v = model.get_Q(z_v)

        # vGraph loss
        res = torch.zeros([n_nodes, categorical_dim], dtype=torch.float32).to(device)
        for idx, e in enumerate(train_edges):  #
            res[e[0], :] += q_vg[idx, :]
            res[e[1], :] += q_vg[idx, :]

        loss_v = loss_function_v(recon_c, q_vg, prior, c.to(device), None, None)

        # Clustering loss
        kl_loss = F.kl_div(Q_to.log(), p_to, reduction='batchmean')

        q_to = Q_to.detach().data.cpu().numpy().argmax(1)

        q_z = Q.detach().data.cpu().numpy().argmax(1)
        q_vg = res.detach().data.cpu().numpy().argmax(1)

        loss_gat = SuperGAT.mix_supervised_attention_loss_with_pretraining(
            loss=kl_loss * 100,
            model=model.gat,
            mixing_weight=sgat_args.att_lambda,
            criterion=sgat_args.super_gat_criterion,
            current_epoch=epoch,
            pretraining_epoch=sgat_args.total_pretraining_epoch,
        )

        trade_off_loss = F.mse_loss(z_v, z)

        loss = loss_gat + loss_v + 200 * trade_off_loss

        acc, nmi, ari, f1 = eva(y, q_to, epoch)
        nmi_lst.append(nmi)

        acc_z, nmi_z, ari_z, f1_z = eva(y, q_z, epoch)

        acc_vg, nmi_vg, ari_vg, f1_vg = eva(y, q_vg, epoch)

        if nmi > best_nmi:
            best_nmi = nmi
            best_nmi_epoch = epoch
            nmi_acc, nmi_ari, nmi_f1 = acc, ari, f1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        if epoch % 100 == 0:
            temp = np.maximum(temp * np.exp(-ANNEAL_RATE * epoch), temp_min)


    max_nmi = np.max(nmi_lst)
    print(nmi_lst.index(max_nmi), max_nmi)

    print(f"the epoch of the best nmi:{best_nmi_epoch}and other indicators:{nmi_acc},nmi:{best_nmi},ari:{nmi_ari},f1:{nmi_f1}")
    return acc, nmi, ari, f1


import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # Random model
    setup_seed(12345)

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Cora')
    parser.add_argument('--max_epoch', type=int, default=201)
    parser.add_argument('--max_d_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dv', type=float, default=0.5)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=5, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=128, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--embedding_vg', type=int, default=128)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--lamda', type=float, default=0.1)
    parser.add_argument('--lamda2', type=float, default=10.0)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')

    args = parser.parse_args(args=[])
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = get_dataset(args.name)
    dataset = datasets[0]
    args.edge_size = dataset.num_edges
    args.node_num=dataset.num_nodes
    if args.name == 'Citeseer':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 6
    elif args.name == 'Cora':
        args.lr = 0.0001
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    # the neighborhood parameters
    sgat_args = get_args(model_name="GAT",
                         dataset_class="Planetoid",
                         dataset_name="Cora",
                         custom_key="EV13NSO8")
    sgat_args.num_hidden_features=128
    sgat_args.outsize = 128
    dataset_kwargs = {}
    if sgat_args.dataset_class == "ENSPlanetoid":
        dataset_kwargs["neg_sample_ratio"] = sgat_args.neg_sample_ratio
    if sgat_args.dataset_class == "WikiCS":
        dataset_kwargs["split"] = sgat_args.seed % 20
    train_d, val_d, test_d = get_dataset_or_loader(
        sgat_args.dataset_class, sgat_args.dataset_name, sgat_args.data_root,
        batch_size=sgat_args.batch_size, seed=sgat_args.seed, num_splits=sgat_args.data_num_splits,
        **dataset_kwargs,
    )

    args.lr=0.001

    sgat_args.att_lambda=1
    print(args)
    acc, nmi, ari, f1 = trainer(dataset,device)
    print(acc, nmi, ari, f1)
